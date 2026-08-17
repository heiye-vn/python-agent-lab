# 附录 A：LangGraph JS/TS 版差异速览

LangGraph 有官方的 JS/TS 实现（`@langchain/langgraph`），核心概念与 Python 版完全一致（本书全部心智模型通用），但 API 细节有差异。本附录给 Python 读者一份对照速查，足以快速上手 TS 项目。

## A.1 安装与包结构

```bash
npm install @langchain/langgraph @langchain/core @langchain/openai
```

| 能力 | Python 包 | JS 包 |
|---|---|---|
| 核心 | `langgraph` | `@langchain/langgraph` |
| 预构建 Agent | `langgraph.prebuilt` | `@langchain/langgraph/prebuilt` |
| 持久化 | `langgraph-checkpoint-*` | `@langchain/langgraph-checkpoint-*` |
| SDK 客户端 | `langgraph-sdk` | `@langchain/langgraph-sdk` |
| supervisor/swarm | `langgraph-supervisor` 等 | `@langchain/langgraph-supervisor` 等 |
| deepagents | `deepagents` | `deepagents`（JS 版同步提供） |

## A.2 核心概念对照表

| 概念 | Python | JS/TS |
|---|---|---|
| 状态注解 | `Annotated[list, add_messages]` | `{ messages: addMessages.annotations }`（Annotation.Root） |
| 状态定义 | `TypedDict` / Pydantic | `Annotation.Root({...})` 或 StateGraph 泛型 + zod |
| 入口/出口 | `START` / `END` | `START` / `END`（同） |
| 条件边 | `add_conditional_edges(n, fn, map)` | `addConditionalEdges(n, fn, map)` |
| Command | `from langgraph.types import Command` | `import { Command } from "@langchain/langgraph"` |
| Send | `Send("node", input)` | `new Send("node", input)` |
| interrupt | `interrupt(payload)` / `Command(resume=)` | `interrupt(payload)` / `new Command({ resume })` |
| 检查点内存 | `InMemorySaver` | `@langchain/langgraph-checkpoint` 的 `MemorySaver` |
| 长期记忆 | `InMemoryStore` | `InMemoryStore`（同） |
| 流式 | `stream(input, {streamMode})` | `stream(input, { streamMode })` |
| entrypoint/task | `@entrypoint` / `@task` 装饰器 | `entrypoint()` / `task()` 高阶函数包装 |

## A.3 最小对照示例

**Python（本书第 2 章）**：

```python
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
graph = builder.compile(checkpointer=InMemorySaver())
graph.invoke({"messages": [("user", "hi")]}, config)
```

**JS/TS 等价**：

```typescript
import { StateGraph, MessagesAnnotation, START, END, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

const chatbot = async (state: typeof MessagesAnnotation.State) => {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
};

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("chatbot", chatbot)
  .addEdge(START, "chatbot")
  .addEdge("chatbot", END);

const graph = workflow.compile({ checkpointer: new MemorySaver() });

const result = await graph.invoke(
  { messages: [{ role: "user", content: "你好" }] },
  { configurable: { thread_id: "t-1" } },
);
```

状态定义的 Annotation 风格：

```typescript
import { Annotation, messagesStateReducer } from "@langchain/langgraph";

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({ reducer: messagesStateReducer, default: () => [] }),
  query: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
});
```

（对照 Python 的 `Annotated[list, add_messages]` 与默认覆盖语义——reducer 显式写出来，语义一致。）

## A.4 主要差异清单

1. **命名风格**：snake_case → camelCase（`add_edge`→`addEdge`、`stream_mode`→`streamMode`、`thread_id`→`threadId`）
2. **状态定义**：没有 Pydantic；用 `Annotation.Root` 或 zod schema；reducer 必须显式（含覆盖型）
3. **异步为主**：JS 版 API 天然 Promise 化，`await` 是日常
4. **预构建**：`createReactAgent` 在 `@langchain/langgraph/prebuilt`，参数同构（`prompt` 可为函数）
5. **Server/Studio/SDK**：完全同套（`langgraph.json` 通用，`node` 项目用 `langgraph-cli` 同样 `langgraph dev/build`）
6. **生态位差**：个别新特性/中间件首发 Python，JS 随后跟进（选型时查 changelog）
7. **功能面差异持续缩小**：1.x 后两版本 API 对齐是官方明确目标

## A.5 选型建议

- 团队主技术栈 Node/TS、要做全栈一体化 → JS 版（前后端同语言，SSE 处理自然）
- 数据科学/算法侧协作、用 Python 生态工具（pandas、向量库客户端）多 → Python 版
- **混编很常见**：模型/数据处理用 Python 图，BFF 层通过 LangGraph Server 的 REST/SDK 调用——Server 与语言无关
