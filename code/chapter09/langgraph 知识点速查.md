# LangGraph 知识点速查清单

> 定位：概念梳理用速查手册，不含代码示例。每个知识点 1-2 句话精炼概括，适合快速回忆复习。
>
> LLM 对接说明：文中涉及 LLM 调用的部分均以阿里云百炼 qwen3.7-max 为例，通过 `init_chat_model` + `ALI_BAILIAN_API_KEY` 环境变量初始化。

---

## 一、核心概念层

### 1.1 LangGraph 与 LangChain 的关系

LangGraph 是 LangChain 团队推出的状态图编排框架，负责多节点间的状态流转和循环控制；LangChain / LCEL 负责单节点内的具体逻辑（如调用 LLM、解析输出）。两者互补不互斥：图的节点内部可以用 Runnable / LCEL chain 实现。

### 1.2 StateGraph（状态图）

LangGraph 的核心抽象。用 `StateGraph(State)` 创建，通过 `add_node()` 添加节点、`add_edge()` / `add_conditional_edges()` 连接边，最后 `compile()` 编译为可执行图。

### 1.3 State（状态）

图在节点间传递的数据结构。通常用 `TypedDict` 或 Pydantic Model 定义。每个字段可搭配 reducer 控制合并策略。

### 1.4 Reducer（状态合并策略）

默认策略是"覆盖"——后一个节点的返回值直接替换 state 中对应字段。对于需要累加的字段（如对话历史 messages），需用 `Annotated[list, add_messages]` 声明为"追加"策略。自定义 reducer 可实现更复杂的合并逻辑。

### 1.5 Node（节点）

图中的执行单元，本质是一个函数。接收当前 state 作为参数，返回一个 dict 表示对 state 的更新（只需包含要修改的字段，非全量替换）。

### 1.6 Edge（边）

节点间的连接，定义执行流转方向。分三类：普通边（固定跳转）、条件边（根据 state 动态决定下一节点）、入口/出口边（从 START 出发、到 END 结束）。

### 1.7 START 与 END

两个特殊节点。START 是图的唯一入口，END 是图的终止。图必须包含从 START 出发的边和（至少一条）通向 END 的边。

### 1.8 compile（编译）

`graph.compile()` 将 StateGraph 编译为可执行对象。编译时可传入 Checkpointer（持久化）、interrupt_before/after（人机协作暂停点）、recursion_limit（递归限制）等参数。

### 1.9 invoke / stream / astream

编译后图的执行方式。`invoke(input)` 一次性执行并返回最终结果；`stream(input)` 流式返回每步的 state 更新；`astream(input)` 异步流式，适合配合 FastAPI 等 async 框架。

### 1.10 Stream Mode

`stream()` / `astream()` 支持指定模式：`values`（返回每步完整 state）、`updates`（返回每步增量更新）、`messages`（返回 LLM 的 token 流）、`debug`（返回最详细的调试信息，含每步输入输出）。

---

## 二、Agent 基础模式层

### 2.1 ReAct 模式

最经典的 Agent 架构：LLM 思考 → 调用工具 → 观察结果 → 再思考，循环直到 LLM 认为不需要再调工具。LangGraph 的条件边天然适合实现这个循环。

### 2.2 Tool Node（工具节点）

封装工具调用的节点。通常用 `ToolNode(tools)` 创建，它会从 state 中取出 LLM 的 `tool_calls`，执行对应工具，将结果以 `ToolMessage` 形式放回 state。

### 2.3 bind_tools

将工具定义绑定到 LLM 上，使 LLM 能在回复中生成结构化的 `tool_calls`。对应百炼 qwen3.7-max，通过 `init_chat_model(...).bind_tools(tools)` 实现。

### 2.4 tools_condition（条件路由）

ReAct 模式的路由核心：检查 LLM 回复中是否包含 `tool_calls`——有则路由到工具节点，无则路由到 END。LangGraph 提供现成的 `tools_condition` 函数可直接使用。

### 2.5 create_react_agent

LangGraph 提供的快捷函数，一行代码创建完整的 ReAct Agent。内部自动搭建 LLM 节点 → 工具节点 → 条件边的循环。适合快速上手，但理解原理后建议手搭一遍。

### 2.6 Checkpointer（状态持久化）

负责在每步执行后保存图的状态快照，支持断点续传和多轮对话记忆。`MemorySaver` 用于开发（内存，重启丢失），`SqliteSaver` / `PostgresSaver` 用于生产（持久化到数据库）。

### 2.7 thread_id（会话标识）

配合 Checkpointer 使用，通过 `config={"configurable": {"thread_id": "xxx"}}` 区分不同会话。相同 thread_id 的多次调用会自动加载历史 state，实现多轮对话。

### 2.8 Memory Saver vs Production Saver

开发阶段用 `MemorySaver`（纯内存，简单但不持久）；上线后需切换到 `SqliteSaver`（单机文件）或 `PostgresSaver`（多实例共享、生产级）。切换只需改 compile 时传入的 checkpointer 实例。

### 2.9 Recursion Limit（递归限制）

图执行时节点间循环的最大次数限制，默认 25。防止 Agent 陷入死循环。通过 `compile(recursion_limit=N)` 或运行时 config 调整。超限会抛出 `GraphRecursionError`。

---

## 三、高级模式层

### 3.1 Human-in-the-loop（人机协作）

在指定节点暂停图执行，等待人工确认/修改后再继续。通过 `compile(interrupt_before=["node_name"])` 或 `interrupt_after` 实现。暂停后用 `graph.update_state()` 修改状态，再 `graph.invoke(None, config)` 续传。

### 3.2 interrupt（主动中断）

LangGraph 较新版本提供的 `interrupt()` 函数，可在节点函数内部主动触发暂停，将值传给调用方并等待人工输入后返回。比 `interrupt_before/after` 更灵活，适合需要节点内部动态决定是否暂停的场景。

### 3.3 多 Agent 架构总览

将多个 Agent 组成协作网络。核心问题是：谁来决定调用哪个子 Agent、Agent 间如何传递消息。常见模式有 Supervisor（主管调度）和 Hierarchical（层级调度）。

### 3.4 Supervisor 模式

一个"主管"Agent 负责接收用户请求、决定交给哪个子 Agent 处理、汇总子 Agent 结果后返回。适合任务边界清晰、子 Agent 各司其职的场景。

### 3.5 Hierarchical 模式

多层调度结构：顶层主管 → 中层主管 → 底层执行 Agent。适合复杂任务需要多级分解的场景。本质是 Supervisor 模式的递归嵌套。

### 3.6 子图（Subgraph）

将一段完整流程封装成子图，作为单个节点嵌入更大的图中。子图有自己的 State 和边定义，通过 `add_node("subgraph_name", subgraph_compiled)` 嵌入。实现流程复用和复杂图的组织。

### 3.7 并行执行

从一个节点引出多条边指向不同节点，这些节点会并行执行。并行节点的结果通过 reducer 合并回 State。需注意并行节点间不应有数据依赖（否则需通过 state 显式传递）。

### 3.8 Send（动态路由）

`Send(node, state)` 机制允许条件边动态返回多个目标节点及各自独立的 state，实现 map-reduce 式的并行扇出。与固定并行边不同，Send 的目标数量和内容在运行时才确定。

### 3.9 Command（指令对象）

LangGraph 较新版本的 `Command` 对象，让节点函数可以同时声明"返回的状态更新 + 要跳转的下一个节点"，取代传统的 `add_conditional_edges`。使节点的状态输出和路由逻辑内聚在一处。

### 3.10 流式输出的三个层级

节点级流式（`stream_mode="updates"` 返回每个节点的输出增量）、消息级流式（`stream_mode="messages"` 返回 LLM 生成的 token 流，可配合 `astream_events` 做更细粒度控制）、事件级流式（`astream_events(version="v2")` 返回所有内部事件，最细粒度，适合 UI 展示）。

### 3.11 Subgraph Streaming

子图内部的状态流转默认对外不可见。通过 `stream_mode` 传入列表如 `["updates", "messages"]` 或使用 `subgraphs=True` 参数，可以在主图流式输出中看到子图内部每步的更新。

---

## 四、工程实战层

### 4.1 State 设计原则

字段尽量精简，只放真正需要在节点间传递的数据；对话历史等需累加的字段必须配 reducer；复杂状态可拆分为多个子状态分别管理。State 设计直接影响图的清晰度和性能。

### 4.2 Annotated 类型标注

`Annotated[list, add_messages]` 是最常用的标注方式，声明该字段使用追加策略而非覆盖。也可以用 `operator.add`（列表拼接/数字累加）或自定义函数。忘记标注是初学者最常见的 bug 来源。

### 4.3 错误处理

节点函数抛出的异常会中断整个图。处理方式：在节点函数内部 try/except 并将错误信息写入 state；或使用 fallback 节点（条件边判断 state 中是否有 error 字段，有则路由到错误处理节点）。

### 4.4 Retry 策略

LLM 调用失败（如 API 限流、超时）时需要自动重试。LangGraph 层面无内建重试，需在节点函数内部用 tenacity 等库实现，或在 LLM 初始化时配置 `max_retries` 参数。百炼 qwen3.7-max 的 `init_chat_model` 支持透传 retry 配置。

### 4.5 LangSmith 集成

LangSmith 是 LangChain 生态的可观测性平台。设置 `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` 后，图的每一步执行（节点输入输出、LLM 调用、工具执行）都会被自动 trace，可在 LangSmith 面板查看和调试。

### 4.6 部署：LangGraph Server / Cloud

LangGraph 官方部署方案，提供 REST API、异步任务队列、持久化存储、流式 SSE 等能力。适合需要生产级 Agent 服务的场景。也可以自行用 FastAPI 包装编译后的图，灵活但需自己处理持久化和并发。

### 4.7 自部署：FastAPI + LangGraph

用 FastAPI 将编译后的图包装为 HTTP 服务。核心要点：Checkpointer 用持久化后端（Sqlite/Postgres）、流式输出转为 SSE（Server-Sent Events）、thread_id 作为 API 参数传入。适合不想用 LangGraph Cloud 或有定制化需求的场景。

### 4.8 异步执行

图可以同步（`invoke` / `stream`）或异步（`ainvoke` / `astream`）执行。异步模式下节点函数需定义为 `async def`，内部 LLM 调用用 `ainvoke`。生产环境推荐异步以支持并发请求。注意：百炼 qwen3.7-max 通过 `init_chat_model` 初始化后同时支持同步和异步调用。

### 4.9 条件边的实现方式

`add_conditional_edges(source, router_function, path_map)` 中，`router_function` 接收 state 返回字符串（目标节点名），`path_map` 是可选的映射字典。新版 LangGraph 也支持 `Command` 对象在节点内部声明路由，省去条件边的定义。

### 4.10 图的可视化

`graph.get_graph().draw_mermaid()` 可生成 Mermaid 格式的图结构描述，方便在 Markdown 中嵌入预览。也可用 `draw_png()` / `draw_mermaid_png()` 直接输出图片（需安装相关可选依赖）。

---

## 五、百炼 qwen3.7-max 对接速查

### 5.1 初始化方式

通过 `langchain.chat_models.init_chat_model` 初始化，模型名指定为 `qwen3.7-max`（或平台对应标识），API Key 从 `.env` 文件的 `ALI_BAILIAN_API_KEY` 读取。初始化后得到的 model 对象可直接作为图节点的 LLM。

### 5.2 绑定工具

`model.bind_tools(tools)` 后，LLM 回复中会包含结构化的 `tool_calls` 字段，供 ToolNode 和 `tools_condition` 消费。百炼 qwen3.7-max 支持 OpenAI 兼容的 function calling 格式。

### 5.3 流式输出

`model.stream()` / `model.astream()` 返回 token 级流式输出。在 LangGraph 中配合 `stream_mode="messages"` 可以让图的流式输出包含 LLM 生成的实时 token，适合做打字机效果的前端展示。

### 5.4 环境变量

核心环境变量：`ALI_BAILIAN_API_KEY`（百炼 API Key）、`LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY`（LangSmith 追踪，可选但推荐）。均在项目 `.env` 文件中配置，通过 `python-dotenv` 加载。

---

## 六、学习路径建议

1. 手搭最简两节点图，跑通 state 流转（理解 Node / Edge / State / compile）
2. 手搭 ReAct Agent，不用 `create_react_agent`（理解条件边循环 + Tool Node + tools_condition）
3. 加 Checkpointer 做多轮对话记忆（理解 thread_id + 状态持久化）
4. 加 Human-in-the-loop 暂停（理解 interrupt_before + update_state + 续传）
5. 尝试多 Agent Supervisor 架构（理解子图 + Agent 间消息传递）
6. 配置 LangSmith 追踪 + 异步执行（进入工程化阶段）
7. 用 FastAPI 自部署或接入 LangGraph Server（完成生产化闭环）
