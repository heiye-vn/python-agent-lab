# 第 23 章：本地开发与调试

从本章起进入企业级部署篇。先讲本地开发体验三件套：**脚手架、LangGraph Server（本地版）、LangGraph Studio**，以及怎么给图写测试。

## 23.1 项目脚手架：langgraph new

```bash
pip install langgraph-cli[inmem]

langgraph new my-agent        # 交互式选择模板（new-langgraph-project-python 等）
cd my-agent
```

生成的标准结构：

```
my-agent/
├── src/agent.py          # 你的图（langgraph.json 里指向它）
├── langgraph.json        # 项目配置（核心）
├── .env                  # 环境变量
├── pyproject.toml
└── README.md
```

## 23.2 langgraph.json：项目清单

```json
{
  "dependencies": ["."],                  # 依赖来源："."= 当前项目(pip install -e .)，或 requirements 路径
  "graphs": {
    "my_agent": "./src/agent.py:graph"    # 暴露的图：名字 -> 模块:变量
  },
  "env": ".env",                          # 运行时加载的环境变量文件
  "python_version": "3.11",
  "dockerfile_lines": []                  # 构建镜像时的额外指令
}
```

一个项目可暴露多个图（`graphs` 多个条目），Server 上每个图成为一个"agent"端点。

## 23.3 langgraph dev：本地 Server + 热重载

```bash
langgraph dev
# 输出:
# - API: http://127.0.0.1:2024
# - Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
# - API Docs: http://127.0.0.1:2024/docs
```

它启动了一个**完整的 LangGraph Server（内存模式）**：
- REST API 与生产部署**完全一致**（第 25 章）——本地联调前端 = 生产同款协议
- 代码热重载：改代码自动重建图
- 自带 OpenAPI 文档（`/docs`）

后端联调示例（还没讲 SDK，先感受一下）：

```bash
# 创建 thread
curl -X POST http://127.0.0.1:2024/threads
# → {"thread_id": "abc-123", ...}

# 跑图
curl -X POST http://127.0.0.1:2024/threads/abc-123/runs \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "my_agent", "input": {"messages": [{"role":"user","content":"你好"}]}}'
```

## 23.4 LangGraph Studio：可视化调试台

浏览器打开 Studio 链接（本地免费，无需付费 LangSmith 账户也可用基础功能）：

| 功能 | 操作 |
|---|---|
| **图结构** | 左侧自动渲染 mermaid 图，节点可点击 |
| **对话测试** | 输入框直接聊，右侧看每步状态 |
| **流式** | token / updates 实时滚动 |
| **HITL 演练** | 图 interrupt 后直接在 UI 里输入 resume 值继续 |
| **时间旅行** | 检查点时间线上任选快照 → 查看状态 → **修改字段 → Fork/Replay**（第 13 章） |
| **Store 检查** | 查看长期记忆条目 |

调试复杂 Agent 的标准姿势：**Studio 里跑 + LangSmith 里看 trace**，两者配合覆盖"状态视角"与"执行视角"。

## 23.5 测试策略

### 单元层：节点/工具直接测

```python
import pytest

def test_search_node():
    state = {"query": "langgraph", "docs": [], "rounds": 0}
    result = search_node(state)
    assert len(result["docs"]) > 0
    assert result["rounds"] == 1
```

### 图层：Fake 模型跑全图（不花 token）

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

def make_graph_with_fake_llm(fake_responses: list):
    fake = GenericFakeChatModel(messages=iter(fake_responses))
    return builder.compile()   # 把 llm 注入方式设计成可替换（依赖注入）

def test_agent_calls_tool_then_answers():
    fake_llm = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "get_weather",
                                           "args": {"city": "上海"},
                                           "id": "c1"}]),
        AIMessage(content="上海 25 度，晴。"),
    ]))
    graph = build_agent(llm=fake_llm)
    result = graph.invoke({"messages": [("user", "上海天气？")]})
    assert "25" in result["messages"][-1].content
```

要点：**把 llm、工具依赖做成可注入参数**——这也是为什么推荐"类节点/工厂函数"写法。

### 断言"调了哪个工具"（Agent 行为测试）

```python
def test_refund_intent_routes_to_refund_agent():
    result = graph.invoke({"messages": [("user", "我要退款")]})
    tool_names = [tc["name"] for m in result["messages"]
                  for tc in (getattr(m, "tool_calls", None) or [])]
    assert "create_refund" in tool_names
```

### 回归层：LangSmith 数据集 + 评估（第 27 章展开）

把线上 badcase 存成数据集，每次改提示/换模型跑一遍自动评分。

## 23.6 推荐开发循环

```
1. 写/改图 → langgraph dev 热重载
2. Studio 对话冒烟（看状态流转对不对）
3. 断点/时间旅行定位问题节点
4. LangSmith trace 看该步的完整输入输出
5. 沉淀为 fake-llm 单测 → 进 CI
6. badcase 进 LangSmith 数据集
```

## 本章小结

- `langgraph new` 起项目，`langgraph.json` 是清单（graphs 暴露多个图）
- `langgraph dev` = 本地全功能 Server（2024 端口）+ 热重载 + OpenAPI
- Studio：图结构、对话、HITL 演练、时间旅行、Store——调试主战场
- 测试三层：节点单测 / fake-llm 图测 / LangSmith 回归数据集
- 设计可注入的 llm 与工具依赖是可测试性的关键

> 下一章：LangGraph Platform 的运行时架构——threads/runs/assistants 到底是什么。
