# Langfuse 接入、使用与数据分析完全指南

> 适用对象：AI Agent / LLM 应用开发工程师
> 版本基线：Langfuse v4（2026 年，全面基于 OpenTelemetry 重写）+ 自托管/Cloud 双形态
> 阅读时长：约 40 分钟

---

## 目录

1. [Langfuse 是什么 & 为什么 Agent 开发必须用](#1-langfuse-是什么)
2. [核心概念与数据模型](#2-核心概念与数据模型)
3. [部署方式：Cloud vs 自托管](#3-部署方式)
4. [安装与认证](#4-安装与认证)
5. [接入方式（重点）](#5-接入方式重点)
6. [日常使用：UI 里看什么](#6-日常使用ui-里看什么)
7. [评分与反馈（Scoring）](#7-评分与反馈scoring)
8. [提示词管理（Prompt Management）](#8-提示词管理prompt-management)
9. [数据集与评估（质量闭环）](#9-数据集与评估质量闭环)
10. [数据分析（重点）](#10-数据分析重点)
11. [生产最佳实践](#11-生产最佳实践)
12. [v3 → v4 改名对照 & 常见坑](#12-v3--v4-改名对照--常见坑)
13. [附录：完整多步 Agent 接入示例](#13-附录完整多步-agent-接入示例)

---

## 1. Langfuse 是什么

Langfuse 是**开源（MIT）的 LLM/AI Agent 应用可观测平台**。它和传统 APM（Datadog、Prometheus）最大的区别是：专门为"非线性的多步 LLM 执行"设计——一次 Agent 运行可能包含数十轮模型调用、工具调用、检索、中间推理，Langfuse 把它们组织成一棵 **Trace 树**，让你能看清每一步的输入输出、token、耗时、成本、质量分。

**解决的真实痛点：**

| 没有可观测 | 用了 Langfuse |
|-----------|--------------|
| Agent 答错了，翻 raw 日志猜哪步出错 | 在 Trace 树里点开每一步，看 prompt/输出/工具返回 |
| 月底才发现 LLM 账单爆了 | 每个 observation 的 token/成本归因，哪个环节最烧钱一眼可见 |
| "模型好像变差了"凭感觉 | Dataset + LLM-as-Judge 自动打分，跑回归对比版本 |
| 改完 prompt 心里没底 | Prompt 版本管理 + A/B，直接量化质量/成本差异 |

**两种形态：**

- **Langfuse Cloud**：托管 SaaS。免费 Hobby 档（约 50K units/月）→ Core $29/月 → Pro $199/月 → Enterprise。
- **自托管**：MIT 协议，**核心永久免费、无 Trace 数量/席位/留存限制**。唯一区别只是把 `LANGFUSE_HOST` 改成你自己的地址，其余代码完全一致。

> 2026 年 1 月 ClickHouse 收购 Langfuse，但**协议、自托管路径、SDK 全部不变**。

---

## 2. 核心概念与数据模型

### v4 数据模型（observation-centric）

v4 把所有数据（LLM 调用、工具执行、Agent 步骤）写入**一张宽的、基本不可变的 ClickHouse 表**，去掉了读取时的 join 和去重，仪表盘加载在大项目上快约 10×。

### 四个核心对象

| 对象 | 含义 | 类比 |
|------|------|------|
| **Trace** | 一次完整请求/会话的根 | 一次 HTTP 请求的完整生命周期 |
| **Observation** | Trace 下的节点，有三种类型 | span |
| ├─ Span | 普通步骤（工具调用、检索、业务逻辑） | 普通 span |
| ├─ Generation | 一次模型调用（自动记录模型/token/成本） | LLM span |
| └─ Event | 轻量事件（打点、状态变更） | event |
| **Session** | 同一用户的多轮对话归组 | 会话 |
| **Score** | 挂在某条 Trace/Observation 上的评分 | 标签/分数 |

### 关键属性

- **trace_id**：串联整条链路（也建议作为你 API 响应头返回，方便用户反馈时关联）
- **observation_id**：单个节点
- **user_id / session_id / tags**：用于过滤、分组、权限隔离
- **metadata**：任意 JSON，存模型名、上下文窗口占用率等业务信息
- **usage / cost / latency**：Generation 自动计算，Span 可手动填

---

## 3. 部署方式

### 3.1 Langfuse Cloud（最快上手）

1. 访问 `https://cloud.langfuse.com`（EU）或 `https://us.cloud.langfuse.com`（US）
2. 建 Organization → Project
3. 在 Project Settings 拿到 `pk-lf-...`（public）和 `sk-lf-...`（secret）

### 3.2 自托管 Docker Compose（生产推荐）

**最小依赖版本（v4）：** ClickHouse ≥ 25.12（推荐 26.4）、Postgres ≥ 16、Redis ≥ 7.2。

**第一步：生成密钥（切勿复用，用密码学安全源生成）**

```bash
NEXTAUTH_SECRET=$(openssl rand -base64 32)
SALT=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)   # 必须正好 64 个 hex 字符（256-bit）
POSTGRES_PASSWORD=$(openssl rand -base64 24)
CLICKHOUSE_PASSWORD=$(openssl rand -base64 24)
```

**第二步：`docker-compose.yml`（生产倾向版，pin 镜像 + 命名卷）**

```yaml
services:
  langfuse-web:
    image: langfuse/langfuse:3
    environment:
      DATABASE_URL: postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/postgres
      NEXTAUTH_URL: https://langfuse.internal.example.com
      NEXTAUTH_SECRET: ${NEXTAUTH_SECRET}
      SALT: ${SALT}
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD}
      REDIS_CONNECTION_STRING: redis://redis:6379
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
    ports:
      - "3000:3000"
    depends_on: [postgres, clickhouse, redis]

  langfuse-worker:
    image: langfuse/langfuse-worker:3
    environment:
      DATABASE_URL: postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/postgres
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD}
      REDIS_CONNECTION_STRING: redis://redis:6379
    depends_on: [postgres, clickhouse, redis]

  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes: ["pgdata:/var/lib/postgresql/data"]

  clickhouse:
    image: clickhouse/clickhouse-server:26.4
    environment:
      CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD}
      CLICKHOUSE_USER: clickhouse
    volumes: ["chdata:/var/lib/clickhouse"]
    ports: ["8123:8123"]

  redis:
    image: redis:7.2
    volumes: ["redisdata:/data"]

  minio:   # 事件对象存储（也可换阿里云 OSS / S3）
    image: minio/minio
    command: server /data --console-address ":9001"
    volumes: ["miniodata:/data"]

volumes:
  pgdata:
  chdata:
  redisdata:
  miniodata:
```

> 容器数量：生产环境约 6 个（web / worker / postgres / clickhouse / redis / 对象存储）。开发/试用 4GB 内存即可跑起来。

**第三步：启动并建项目**

```bash
docker compose up -d
# 打开 http://localhost:3000，第一个注册用户即 admin
# 建 Organization → Project → 在 Project Settings 复制 public/secret key
```

拿到两个 key 后，下面所有 SDK 代码用这两个 key 认证——**自托管和 Cloud 的唯一差别就是 `LANGFUSE_HOST`**。

---

## 4. 安装与认证

```bash
pip install langfuse openai   # openai 仅示例用，可用任意模型 SDK
```

**环境变量（推荐放 `.env`）**

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # 自托管改成 http://localhost:3000
```

**v4 客户端初始化**

```python
import os
from langfuse import get_client

langfuse = get_client()   # v4 入口：读取环境变量，返回共享单例
print("Connected:", langfuse.auth_check())   # True 即连接成功
```

> ⚠️ **v3 旧写法 `Langfuse(public_key=..., secret_key=...)` 仍能运行，但在 v4 UI 里新 observation 最多有约 10 分钟延迟。** 新项目一律用 `get_client()`。

---

## 5. 接入方式（重点）

下面按"侵入度从低到高"给出五种接入姿势，覆盖绝大多数 Agent 栈。

### 5.1 最快：`@observe` + OpenAI 原生 drop-in

**两行代码看见一切**：把 `from openai import OpenAI` 换成 `from langfuse.openai import OpenAI`，每个 `chat.completions.create` 自动变成一条 Generation（模型、消息、输出、token、成本全记录）；外层用 `@observe` 包成一条 Trace。

```python
from langfuse import get_client, observe
from langfuse.openai import OpenAI   # 注意：是 langfuse.openai，不是 openai

langfuse = get_client()
client = OpenAI()

@observe()
def answer(question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content

answer("What is the capital of France?")
langfuse.flush()   # 短脚本务必 flush，否则最后几条 trace 会丢
```

UI 里看到：根为 `answer` 函数，子节点为 LLM Generation，usage/cost 已自动填好。

### 5.2 Anthropic / 其他模型（重要：无原生 wrapper）

> 🚨 **Langfuse 没有 `from langfuse.anthropic import ...`，这是 AI 生成代码最常见的幻觉。** 只有 OpenAI 有原生 drop-in。

对 Claude 等非 OpenAI 模型，两条正路：

**方案 A：继续用 `@observe`（记录函数的输入/输出）**

```python
import anthropic
from langfuse import get_client, observe

langfuse = get_client()
client = anthropic.Anthropic()

@observe()
def answer(question: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text
```

**方案 B：用 Anthropic 的 OpenTelemetry instrumentor**（v4 是 OTel 底座，任何 OTel 发射的 span 自动进 Langfuse）

```python
# pip install opentelemetry-instrumentation-anthropic
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
AnthropicInstrumentor().instrument()
```

### 5.3 手动构建 Span 树（多步 Agent 推荐）

装饰器适合单调用；**多步 Agent（检索→思考→调工具→回答）值得显式建树**，才能看清延迟和错误落在哪一步。v4 用 `start_as_current_observation`，靠 `as_type` 区分节点类型：

```python
from langfuse import get_client

langfuse = get_client()

def research_agent(query: str):
    with langfuse.start_as_current_observation(
        as_type="span", name="research-agent", input={"query": query}
    ) as root:
        # ---- 子节点 1：检索 ----
        with langfuse.start_as_current_observation(
            as_type="span", name="retrieve", input={"query": query}
        ) as retr:
            context = vector_search(query)
            retr.update_trace(output=context)   # 记录输出

        # ---- 子节点 2：LLM 生成 ----
        with langfuse.start_as_current_observation(
            as_type="generation", name="llm-answer",
            input={"context": context, "query": query},
            model="gpt-4o",
        ) as gen:
            resp = call_llm(context, query)
            gen.update_trace(output=resp)

        root.update_trace(output=resp)
        return resp
```

### 5.4 LangChain / LangGraph 集成

```python
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

langfuse = get_client()

@observe()
def process(query: str):
    handler = CallbackHandler()   # 自动继承当前 @observe 的 trace 上下文
    llm = ChatOpenAI(model_name="gpt-4o")
    chain = ChatPromptTemplate.from_template("Respond to: {input}") | llm
    result = chain.invoke({"input": query}, config={"callbacks": [handler]})
    return result.content

process("What is the capital of France?")
```

### 5.5 自定义 Trace 级属性（用户/会话/标签）

```python
# v4：.propagate_attributes() 替代 v3 的 update_current_trace()
from langfuse import get_client

langfuse = get_client()
with langfuse.start_as_current_observation(
    as_type="span", name="agent-run", input={"q": query}
) as span:
    langfuse.propagate_attributes(
        user_id="user_5678",
        session_id="session-1234",
        tags=["prod", "support-bot"],
    )
    # ...业务逻辑...
```

> 预定义 trace_id（用于和外部系统关联）：`Langfuse.create_trace_id(seed="req_12345")`，再通过 `start_as_current_observation(trace_context={"trace_id": ...})` 传入。

### 5.6 队列与 flush

SDK 后台批量队列上报，**长驻进程无需额外配置**；**短脚本/Serverless 末尾务必 `langfuse.flush()`**，否则最后几条 trace 静默丢失。

---

## 6. 日常使用：UI 里看什么

1. **Tracing 页**：每次运行一条 Trace。点开看执行树：
   - 哪个 observation 耗时最长（优化延迟）
   - 哪个工具调用返回异常（Agent 失措根因）
   - 检索回来的文档是否相关（RAG 质量）
2. **Sessions 页**：同一 `session_id` 的多轮对话归组，回放完整上下文——调试"它怎么忘了前面说过的话"。
3. **Dashboards 页**：token 消耗、延迟、错误率、成本的时序趋势，可设阈值告警。
4. **Filters**：按日期、score 区间、延迟分位数、`user_id`、`tags` 过滤。

---

## 7. 评分与反馈（Scoring）

分数是把"质量"变成可分析数据的核心。三种数据类型：`NUMERIC`（0–1 或任意数值）、`CATEGORICAL`、`BOOLEAN`。

**在当前 trace 上下文内打分（推荐）**

```python
langfuse.score_current_trace(
    name="user_feedback",
    value=1.0,
    data_type="NUMERIC",
    comment="用户点了赞",
)
```

**在上下文外用 trace_id 打分（如用户迟到的反馈）**

```python
langfuse.create_score(
    trace_id=trace_id,
    name="quality",
    value=0.8,
    data_type="NUMERIC",
    comment="LLM-as-Judge 评分",
    score_id="stable-id-123",   # 传稳定 id 可幂等，重跑不重复
)
```

**给单个 observation 打分（如检索相关性）**

```python
# 在 start_as_current_observation 上下文内
span.score(name="retrieval_relevance", value=0.9, data_type="NUMERIC")
```

---

## 8. 提示词管理（Prompt Management）

把 prompt 抽到 Langfuse 的 Prompt 模块做版本管理，代码里**拉取**而非硬编码，改 prompt 不用发版：

```python
prompt = langfuse.get_prompt("support-system-prompt")   # 取最新版
# 或指定版本
prompt_v2 = langfuse.get_prompt("support-system-prompt", version=2)

messages = prompt.compile(user_question="如何退款")   # 支持变量填充
```

在 UI 里可以对比不同版本的质量/成本，直接 A/B。

---

## 9. 数据集与评估（质量闭环）

这是 Agent 开发者最该养成、却最容易忽略的习惯：**每次改 prompt / 换模型，都跑同一批测试集，量化"这次是变好还是变坏"。**

### 9.1 建数据集

```python
langfuse.create_dataset(
    name="support-eval-set",
    description="客服 Agent 回归测试集",
)
langfuse.create_dataset_item(
    dataset_name="support-eval-set",
    input={"question": "订单 12345 到哪里了？"},
    expected_output="已发货，预计明天到达",
)
```

### 9.2 跑实验 + 评估器（v4：`run_experiment`）

```python
from langfuse import get_client, Evaluation

langfuse = get_client()

def task(*, item, **kwargs):
    q = item.input["question"]
    resp = OpenAI().chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": q}]
    )
    return resp.choices[0].message.content

# 评估器接收 input/output/expected_output，返回 Evaluation
def accuracy(*, input, output, expected_output, **kwargs):
    if expected_output and expected_output[:6] in output:
        return Evaluation(name="accuracy", value=1.0, comment="命中关键字")
    return Evaluation(name="accuracy", value=0.0, comment="未命中")

# LLM-as-Judge 也只是"会调模型的评估器"
def judge(*, input, output, **kwargs):
    score = call_judge_model(input, output)   # 你自己的打分模型
    return Evaluation(name="helpfulness", value=score)

result = langfuse.run_experiment(
    name="support-baseline",
    data=[   # 也可换成 langfuse.get_dataset("support-eval-set").run_experiment(...)
        {"input": {"question": "订单 12345 到哪里了？"}, "expected_output": "已发货"},
    ],
    task=task,
    evaluators=[accuracy, judge],
)
print(result.format())   # 打印每条的延迟/成本/分数
```

> **在线 vs 离线**：`run_experiment` 里的评估器**同步、内联**执行，适合 CI 卡点；Langfuse 仪表盘里配置的托管 LLM-as-Judge 则是**异步**对线上 trace 持续打分。两者互补。

### 9.3 对比视图 & 回归

同一数据集多次实验（不同 prompt/模型）可在 UI Compare 视图按延迟、成本、分数维度对比，直观看到回归。

### 9.4 CI/CD 评估门禁（GitHub Action，2026 Launch Week 5）

把评估变成 PR 合并拦截器，分数低于阈值直接 fail：

```yaml
# .github/workflows/eval.yml
- uses: langfuse/experiment-action@v1
  with:
    experiment_path: ./evals/my_experiment.py
    dataset_name: golden_set_v2
    dataset_version: "3"
    langfuse_public_key: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
    langfuse_secret_key: ${{ secrets.LANGFUSE_SECRET_KEY }}
    github_token: ${{ secrets.GITHUB_TOKEN }}
```

> 要求 Python SDK **v4.6.0+** 或 JS SDK **v5.3.0+**。实验脚本里对不达标分数 `raise RegressionError(...)`，Action 会把它挂到 PR 评论并 fail job。
>
> 配套还有 **Code Evaluators**：在 UI 里写确定性 `evaluate()`（如"输出是否合法 JSON""action 字段是否只含审批值"），零 token 成本、完全确定性，处理客观检查层。

---

## 10. 数据分析（重点）

"数据分析"分四层，按需求深度递增：

### 10.1 UI 内置分析（零代码）

- **成本/延迟/质量趋势**：Dashboards 页直接看，可按 `user_id`、`tags`、模型名切片。
- **Trace 列表 + 过滤器**：按 score 区间、延迟分位、时间筛选，定位"质量最差的那批 trace"。
- **会话回放**：Session 维度复盘多轮对话。

### 10.2 用 Python SDK 拉取数据做自定义分析

```python
from langfuse import get_client

langfuse = get_client()

# 拉取带过滤的 traces（v4 走 /api/public/v2/observations）
traces = langfuse.fetch_traces(
    user_id="user_5678",
    tags=["prod"],
    from_timestamp="2026-08-01",
    to_timestamp="2026-08-17",
)

for t in traces.data:
    print(t.id, t.latency, t.total_cost, t.scores)
```

> v4 公共 API 端点变化：`GET /api/public/v2/observations`、`/api/public/v3/scores`、`/api/public/v2/metrics`。旧的 `/api/public/traces`、`/api/public/generiments` 等在 v4 预览部署里返回 404。

### 10.3 自托管直连 ClickHouse 跑 SQL（最强分析力）

自托管最大红利：**原始数据用 SQL 随便查**。v4 是宽 `observations` 表；v3 是 `traces` + `observations` 两张表。

```sql
-- v4：按 observation 类型统计成本/延迟/错误率
SELECT
    name,
    count()                                            AS n,
    round(avg(total_cost), 5)                          AS avg_cost,
    round(avg(latency) / 1000, 1)                      AS avg_latency_ms,
    round(sumIf(1, level = 'ERROR') / count(), 3)      AS error_rate
FROM observations
WHERE project_id = '<your-project-id>'
  AND start_time > now() - INTERVAL 7 DAY
GROUP BY name
ORDER BY n DESC
LIMIT 50;
```

```sql
-- 找出质量分最低的生成节点（定位 prompt 回归）
SELECT
    o.name,
    o.trace_id,
    s.value AS quality
FROM observations o
JOIN scores s ON s.observation_id = o.id
WHERE s.name = 'helpfulness'
  AND s.value < 0.5
  AND o.start_time > now() - INTERVAL 1 DAY
ORDER BY s.value ASC
LIMIT 20;
```

> 自托管时 ClickHouse 客户端直接连 `clickhouse:8123`（或 9000 原生端口），凭据即 compose 里的 `CLICKHOUSE_USER/PASSWORD`。

### 10.4 导出到数仓 / Parquet（BI 集成）

- **Blob Storage Export**：配置 S3/OSS/MinIO，自动把数据 dump 成 JSONL / Parquet，喂给 Spark / DuckDB / 数仓做离线分析。
- **Web Analytics API**：把指标嵌到你自己的内部看板。
- 自托管还可直接把 ClickHouse 当分析库，省去 ETL。

---

## 11. 生产最佳实践

| 主题 | 建议 |
|------|------|
| **采样** | 全量成本高，生产设采样率（如 10%–20%），关键/失败请求全采 |
| **PII 脱敏** | 在 SDK 层对用户输入/输出做脱敏再记录；或用 `metadata` 存非敏感上下文，正文打码 |
| **性能** | 后台批量队列已优化；短脚本/Serverless 结尾务必 `flush()` |
| **告警** | 在 Dashboards 对成本突增、延迟超 SLA、质量分跌破阈值设告警 |
| **权限** | 用 `user_id`/`session_id` 隔离租户；Enterprise 支持 SSO/RBAC/审计日志 |
| **合规** | 数据敏感 → 自托管 + 全量 air-gapped；Langfuse 有 SOC 2 Type II + ISO 27001 |
| **版本** | 新项目用 v4 SDK（`get_client`）；老项目升级参考官方 v3→v4 migration 指南 |

---

## 12. v3 → v4 改名对照 & 常见坑

| 用途 | v3（旧） | v4（新，推荐） |
|------|---------|---------------|
| 客户端 | `Langfuse(...)` | `get_client()` |
| 追踪函数 | `@observe`（相同） | `@observe`（相同） |
| 手动建 span | `start_span` / `start_generation` | `start_as_current_observation(as_type=...)` |
| Trace 属性 | `update_current_trace(...)` | `propagate_attributes(...)` |
| 上下文工具 | `from langfuse.decorators import langfuse_context` | 直接用 client 方法 |
| 评估 | `evaluate(...)` | `run_experiment(...)` + `Evaluation` |

**常见坑（复制旧教程会踩）：**

1. ❌ `from langfuse.anthropic import ...` —— **不存在**，Claude 用 `@observe` 或 OTel instrumentor。
2. ❌ 短脚本忘了 `langfuse.flush()` —— 最后几条 trace 静默丢失。
3. ❌ v3 老代码直连 v4 预览部署 —— 旧公共端点（`/api/public/traces` 等）返回 404，需走 v2/v3 新端点。
4. ❌ 把 Langfuse 当"普通日志"——它价值在**结构化 Trace 树 + 评分闭环**，请务必用 `@observe` 包出层级。
5. ❌ 只看错误率——LLM 应用"没报错但答错"才是主战场，必须上 Score + 评估。

---

## 13. 附录：完整多步 Agent 接入示例

下面是一份可直接跑的骨架（OpenAI 版），覆盖：Trace 树、Generation、工具调用 Span、评分、自定义属性。

```python
# pip install langfuse openai
import os
from langfuse import get_client, observe
from langfuse.openai import OpenAI

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"   # 自托管

langfuse = get_client()
client = OpenAI()

# ---------- 工具 ----------
@observe()
def search_docs(query: str) -> str:
    # 真实场景这里连向量库
    return f"[检索结果] 关于「{query}」的官方文档片段..."

@observe()
def call_tool(tool_name: str, params: dict):
    if tool_name == "search_docs":
        return search_docs(params["query"])
    return "unknown tool"

# ---------- Agent 主流程 ----------
@observe()
def agent(user_input: str, session_id: str, user_id: str) -> str:
    # 1) 规划（LLM）
    plan = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是客服 Agent，按需调用工具。"},
            {"role": "user", "content": user_input},
        ],
    ).choices[0].message.content

    # 2) 执行工具（包成 Span）
    tool_out = call_tool("search_docs", {"query": user_input})

    # 3) 综合回答（LLM Generation）
    answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"参考:{tool_out}"},
            {"role": "user", "content": user_input},
        ],
    ).choices[0].message.content

    # 4) 自定义属性 + 评分
    langfuse.propagate_attributes(user_id=user_id, session_id=session_id, tags=["prod"])
    langfuse.score_current_trace(name="answered", value=1.0, data_type="BOOLEAN")
    return answer

if __name__ == "__main__":
    result = agent("我的订单到哪了？", session_id="s-001", user_id="u-001")
    print(result)
    langfuse.flush()
```

跑完打开 Langfuse UI，你会看到一条名为 `agent` 的 Trace，下面挂着 `search_docs` → `call_tool` → 两次 LLM Generation 的子树，每个节点的 token/成本/延迟都可直接下钻。

---

## 参考资源

- 官方文档：https://langfuse.com/docs
- GitHub：https://github.com/langfuse/langfuse
- 数据集与实验：https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk
- v4 自托管预览：https://github.com/orgs/langfuse/discussions/14157
- CI/CD 评估门禁：`langfuse/experiment-action@v1`

> 文档基于 2026 年 8 月 Langfuse v4 公开信息整理。SDK 仍在快速迭代，落地前请以官方文档最新版为准。
