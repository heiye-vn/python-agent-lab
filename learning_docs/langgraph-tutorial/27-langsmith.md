# 第 27 章：可观测性与质量保障（LangSmith）

Agent 系统的上线不是终点——**没有观测的 Agent 系统等于盲飞**。LangSmith 是 LangChain 官方的 LLM 观测/评估/Prompt 管理平台，与 LangGraph 原生集成。本章讲 tracing、评估（evals）、Prompt 管理与 Engine。

## 27.1 Tracing：零代码接入

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2-xxx
export LANGSMITH_PROJECT=my-agent-prod     # 项目隔离（按环境/产品线）
```

之后每次 invoke/run 自动上报。Trace 结构与图结构同构：

```
Trace（一次 invoke / 一个 run）
 ├─ span: agent 节点           输入状态 → 输出状态
 │   └─ span: ChatOpenAI       prompt、补全、token、费用、延迟
 ├─ span: tools 节点
 │   ├─ span: get_weather      args → result
 │   └─ span: search
 └─ span: human_approval       interrupt 详情（第 14 章）
```

**排障三连**：
1. 列表页按 `error=true` 过滤失败 run
2. 打开 trace 看**第一个红色 span**——定位到具体节点/工具
3. 看该 span 的原始输入输出（prompt 全文、工具返回），90% 问题（工具返回超长、参数错了、提示冲突）一眼可见

### 给 trace 打业务标签

```python
result = graph.invoke(inputs, {
    "configurable": {"thread_id": t_id},
    "metadata": {                      # → LangSmith 可按此过滤
        "user_tier": "pro", "env": "prod", "experiment": "prompt-v3",
    },
    "tags": ["customer-service", "vip"],
})
```

### 反馈回流（质检数据）

```python
from langsmith import Client
ls = Client()
ls.create_feedback(run_id=run_id, score=1, key="user_thumbs_up")   # 用户点赞
ls.create_feedback(run_id=run_id, score=0, key="human_review", comment="答非所问")
```

## 27.2 离线评估：数据集 + Evaluator

### 建数据集

```python
from langsmith import Client
ls = Client()

dataset = ls.create_dataset("客服工具选择", description="意图→工具路由回归集")
ls.create_examples(
    dataset_id=dataset.id,
    inputs=[
        {"messages": [{"role": "user", "content": "退款"}]},
        {"messages": [{"role": "user", "content": "怎么开发票"}]},
    ],
    outputs=[                                  # 期望（参考答案）
        {"expected_tool": "create_refund"},
        {"expected_tool": "query_invoice"},
    ],
)
```

### 跑评估

```python
from langsmith import evaluate

def target(inputs: dict) -> dict:              # 被测目标：调你的图
    result = graph.invoke(inputs, config={"recursion_limit": 10})
    return {"messages": result["messages"]}

def tool_correct(run, example) -> dict:        # 评估器1：确定性断言
    tools = [tc["name"] for m in run.outputs["messages"]
             for tc in (getattr(m, "tool_calls", None) or [])]
    return {"score": example.outputs["expected_tool"] in tools}

def helpfulness(run, example) -> dict:         # 评估器2：LLM-as-judge
    answer = run.outputs["messages"][-1].content
    judge = judge_llm.invoke(
        f"判断回答是否准确解决了问题（0/1）：\n问题：{example.inputs}\n回答：{answer}"
    )
    return {"score": int("1" in judge.content)}

results = evaluate(
    target,
    data="客服工具选择",                       # 数据集名
    evaluators=[tool_correct, helpfulness],
    experiment_prefix="prompt-v3",             # 实验名前缀（对比不同提示/模型）
)
```

**工作流**：改提示/换模型 → 跑 evaluate → 实验对比面板看分数差异 → 达标才发版。这是 Agent 的"单元回归测试"。

### 评估器工具箱

| 类型 | 示例 | 适用 |
|---|---|---|
| 确定性 | 工具命中、JSON schema、关键词 | 路由、结构化输出 |
| LLM 判官 | 有用性/正确性/语气 0-1 打分 | 开放式回答 |
| 自定义 | 与参考答案的编辑距离、延迟/成本阈值 | 任何指标 |

## 27.3 在线监控与告警

LangSmith 面板提供：
- **延迟/token/费用**：p50/p95、按节点分解（哪个节点最烧钱一目了然）
- **错误率**：按异常类型聚合
- **反馈率**：点赞点踩趋势
- **告警**：错误率/延迟阈值触发邮件、Webhook（接企业 IM）

生产 SLO 建议：错误率 < 2%、首 token 延迟 p95 < 3s、单会话 token 预算告警。

## 27.4 Prompt Hub：提示即资产

```python
from langchain import hub

# 推送（带 git 语义：版本、说明）
hub.push("my-team/customer-service-prompt",
         SystemMessage("你是金牌客服……"),
         description="v3: 增加退款话术约束")

# 拉取（可锁版本）
prompt = hub.pull("my-team/customer-service-prompt")            # 最新
prompt = hub.pull("my-team/customer-service-prompt:latest.stages.fixed")
```

团队协作：提示改动走 Hub 评审、代码里只引用名字——**提示版本与代码版本解耦**，热修提示不用发版。

## 27.5 LangSmith Engine：从 trace 到修复建议

Engine（2026 年推出的重点能力）在 tracing 之上自动做"质量工程"：

- 持续扫描项目 trace，聚类识别问题模式（某类问题总答错、某工具总被误调、某节点超时）
- 关联 Prompt Hub 中对应提示，**生成修改建议并开 PR**
- 团队评审后合入，回到 27.2 的评估集验证

价值：把"上线后靠人肉翻日志"变成"自动发现→建议→回归→合并"的闭环。

## 27.6 LangSmith Fleet（一图流了解）

无代码 Agent 构建与托管：在 LangSmith 界面里组装 Agent（选模型、配置工具、写提示）直接发布运行。定位是非工程团队的自服务场景；工程主导的生产系统仍以 LangGraph 代码开发为主，两者可共存（Fleet 的 Agent 同样被 tracing 覆盖）。

## 27.7 观测落地路线图

```
第 1 天    开 tracing + 项目/环境标签 + 错误告警
第 1 周    接入用户反馈回流 + 建 20 条核心场景数据集
第 1 月    每次 prompt/模型变更跑 evaluate；上线延迟费用看板
持续       badcase → 数据集 → 回归；启用 Engine 自动诊断
```

## 本章小结

- Tracing 零代码接入；trace 结构与图同构，排障三连定位问题
- metadata/tags 打业务维度；用户反馈 create_feedback 回流质检
- evaluate：数据集 + 确定性/LLM 判官评估器 = Agent 回归测试
- Prompt Hub 提示版本化，与代码解耦
- Engine 自动发现 trace 问题并提议修复；Fleet 面向无代码场景

> 下一章：生产最佳实践总集——模型、成本、安全、可靠性。
