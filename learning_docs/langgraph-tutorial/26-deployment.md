# 第 26 章：部署方案

本章把图真正部署上线：镜像构建、Cloud vs 自托管选型、Docker Compose/K8s 实操、认证、自定义 API 与 Webhook。

## 26.1 选型：Cloud vs 自托管

| | LangGraph Cloud | 自托管（Self-hosted） |
|---|---|---|
| 部署动作 | 代码推上去，控制台点发布 | `langgraph build` 出镜像，自己运维 |
| 基础设施 | 全托管（含 Postgres/Redis） | 自备 Postgres + Redis + K8s/VM |
| 数据边界 | LangSmith 云账户内 | **完全在自己网络**（合规刚需） |
| 弹性/升级 | 平台自动 | 自己管（HPA、滚动发布） |
| 适合 | 快速上线、海外业务 | 金融/政企/内网、数据敏感 |

两者 API 与 SDK 完全一致（第 24 章讲过的 Control/Data Plane 分离：Cloud 托管 Control Plane 时，你的业务数据仍在你的库）。

## 26.2 自托管第一步：构建镜像

```bash
pip install "langgraph-cli[inmem]"
langgraph build -t my-agent:1.0.0        # 读取 langgraph.json 构建生产镜像
docker run -p 8123:8000 my-agent:1.0.3   # 容器内监听 8000
```

镜像内含：LangGraph Server API（`/threads` `/runs` ...）+ 你的图代码 + 依赖。`langgraph.json` 的 `dockerfile_lines` 可追加系统依赖（如 `RUN apt-get install ...`）。

**生产依赖**：镜像本身无状态，需要外接：
- **Postgres**：checkpoints、threads、runs、store（记忆）
- **Redis**：任务队列、流式发布订阅

## 26.3 Docker Compose：单机生产可用版

```yaml
# docker-compose.yml
services:
  langgraph-api:
    image: my-agent:1.0.0
    ports: ["8123:8000"]
    environment:
      REDIS_URI: redis://redis:6379
      DATABASE_URI: postgres://postgres:postgres@postgres:5432/langgraph
      LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
      LANGCHAIN_END_POINT: https://api.smith.langchain.com
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      LANGGRAPH_CLOUD_LICENSE_KEY: ${LANGGRAPH_CLOUD_LICENSE_KEY}  # 自托管需要许可
    depends_on: [postgres, redis]
    restart: always

  postgres:
    image: pgvector/pgvector:pg16        # 带 pgvector，Store 语义检索要用
    environment: {POSTGRES_PASSWORD: postgres, POSTGRES_DB: langgraph}
    volumes: ["pgdata:/var/lib/postgresql/data"]

  redis:
    image: redis:7
    command: redis-server --appendonly yes
    volumes: ["redisdata:/data"]

volumes: { pgdata: {}, redisdata: {} }
```

启动后 `docker compose up -d`，API 就绪于 `http://localhost:8123`。单机多副本（`--scale langgraph-api=3`）即可水平扩展——状态都在库里。

## 26.4 Kubernetes 部署要点

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: langgraph-api }
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: api
          image: my-agent:1.0.0
          ports: [{ containerPort: 8000 }]
          env:
            - { name: REDIS_URI,   value: redis://redis-master:6379 }
            - { name: DATABASE_URI, value: postgres://...@pg:5432/langgraph }
          resources:
            requests: { cpu: "500m", memory: 1Gi }
            limits:   { cpu: "2",   memory: 4Gi }
          readinessProbe: { httpGet: { path: /ok, port: 8000 } }
          livenessProbe:  { httpGet: { path: /ok, port: 8000 } }
```

清单：
- 无状态副本 + HPA（按 CPU 或自定义 QPS 扩缩）
- **优雅停机**：preStop 钩子 + 足够 terminationGracePeriod——等在跑的 superstep 落盘（未完成的 run 会由其他副本续跑）
- Redis 用托管版或 Sentinel/Cluster；Postgres 用云 RDS + 读写分离按需
- 镜像走私有仓库 + CI 自动构建（git tag → build → push → deploy）

## 26.5 自定义认证（auth）

对外暴露必须加认证。在项目里写 auth 处理模块，`langgraph.json` 声明：

```json
{ "auth": { "path": "src/auth:auth" } }
```

```python
# src/auth.py
from langgraph_sdk.auth import Auth

auth = Auth()

@auth.authenticate
async def authenticate(authorization: str | None) -> AuthUser:
    """每个请求先过这里：校验 token，返回用户身份"""
    token = (authorization or "").removeprefix("Bearer ").strip()
    user = await verify_jwt(token)             # 你的 JWT/OAuth/内网 SSO
    return AuthUser(
        identity=user.id,
        display_name=user.name,
        metadata={"tenant": user.tenant_id, "roles": user.roles},
    )

@auth.on.threads.create
async def on_thread_create(ctx, thread):
    """资源级权限：谁能建/访问什么 thread"""
    if ctx.permissions.get("role") != "admin" and thread["metadata"].get("tenant") != ctx.identity:
        raise AuthError("无权访问该会话")
```

两层粒力：**路径级**（@auth.authenticate 统一鉴权）+ **资源级**（`@auth.on.<resource>.<action>` 精确到 thread/run/assistant/store）。第 15 章审批接口的权限校验就落在这里。

## 26.6 自定义 REST API 与 Webhook

### 图旁挂业务端点

```
src/api/
└── routes.py
```

```python
# src/api/routes.py —— FastAPI 写法，与图的 API 同进程部署
from fastapi import APIRouter
router = APIRouter()

@router.get("/my-app/health-detail")
async def health_detail():
    return {"status": "ok", "version": "1.0.0"}
```

部署后可直接调 `GET /my-app/health-detail`。适合：管理后台接口、给非 Agent 业务用的数据端点、审批回调。

### Webhook：run 状态回调业务系统

```json
// langgraph.json
{ "http": { "webhooks": [{ "path": "/webhooks/run-status", "target": "src.webhooks:handler" }] } }
```

或使用托管 Webhook：run 状态变化（success/error/interrupted）时 POST 到你配置的 URL——工单系统据此流转状态。

## 26.7 前端静态资源托管

Server 支持挂静态文件目录（`public_dir` 配置）：聊天前端构建产物直接由 Agent 服务托管，**一个域名同时服务页面与 SSE 流**，省掉 CORS 与一层网关。

## 26.8 上线检查清单

- [ ] `langgraph build` 镜像在**目标环境**跑通冒烟（模型 Key、内网代理）
- [ ] Postgres/Redis 已建、`setup()` 表结构初始化（首启自动）
- [ ] 认证模块启用（禁匿名）
- [ ] 资源配额与 HPA 配好；`/ok` 探针接 K8s
- [ ] LangSmith 项目名固定 + 环境标签（staging/prod）
- [ ] Webhook/告警接通知渠道
- [ ] 备份策略：Postgres 快照；线程合规删除流程
- [ ] 密钥管理：镜像内零明文密钥，全部环境变量/密管注入

## 本章小结

- Cloud 全托管 vs 自托管全掌控，API 同构
- 自托管三件套：镜像（langgraph build）+ Postgres(pgvector) + Redis；无状态副本随便扩
- K8s：/ok 探针、优雅停机等落盘、HPA
- auth 两层：路径级 authenticate + 资源级 on.resource.action
- 自定义 API、Webhook、静态托管让 Agent 服务成为完整产品后端

> 下一章：LangSmith——上线之后"看得见"的部分。
