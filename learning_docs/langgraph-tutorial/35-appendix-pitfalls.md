# 附录 C：常见报错与踩坑集锦

按"报错信息 / 现象"组织，遇到直接查。每条给原因和解法。

## C.1 状态与 Reducer 类

**`InvalidUpdateError: Invalid concurrent update to same channel/action`**
- 原因：同一 superstep 内多个并行节点写了**没有 reducer** 的同一字段（第 3 章）
- 解法：该字段加 reducer（`operator.add` / 自定义可交换函数），或用边串行化写入

**消息只剩最后一条，历史"丢失"**
- 原因：状态里 `messages: list` 没加 `add_messages`，每节点返回覆盖
- 解法：`messages: Annotated[list, add_messages]` 或直接用 `MessagesState`

**节点返回了 dict，但状态没变**
- 原因一：返回的 key 与 State schema 字段名不一致（拼写错误不报错！）
- 原因二：Functional API 里忘了 `.result()`
- 解法：打印 `graph.get_state(config).values` 核对 key；检查 schema

**Pydantic State 报 ValidationError（字段缺失）**
- 原因：invoke 输入没提供必填字段；Pydantic 校验的是**输入合并后**的完整状态
- 解法：给字段默认值，或保证输入完整；或用 input schema 分离必填项（第 3 章）

## C.2 图结构类

**`GraphRecursionError: Recursion limit of 25 reached`**
- 原因：循环没有终止条件（或模型永远"再想一步"）
- 解法：路由条件里加轮数上限；`config={"recursion_limit": N}` 调兜底；捕获异常降级（第 7 章）

**`ValueError: ... must be reached from START`（或悬空节点告警）**
- 原因：节点没接入图（add_edge 漏了或名字拼错）
- 解法：compile 前自查边；用 `draw_mermaid()` 可视化核对

**条件边返回的节点名不存在（运行时 KeyError）**
- 原因：路由函数返回值与实际节点名不一致
- 解法：路径映射 + 白名单校验 + 默认兜底（第 7 章）；路由用结构化输出枚举（第 19 章）

**compile 后修改图报错**
- 原因：图不可变
- 解法：重新 build/compile；需要动态行为用 Command/Send/Functional API

## C.3 持久化与 HITL 类

**配了 checkpointer 但"没记忆"**
- 原因一：每次 invoke 用了**不同的 thread_id**
- 原因二：新建了 checkpointer 实例（InMemorySaver 的数据在对象里，进程内要复用同一实例；跨进程换 DB 实现）
- 解法：固定 thread_id；checkpointer 全局单例（第 11 章）

**`interrupt() called outside of a node/graph`**
- 原因：在工具内部/节点外直接调 interrupt（工具是独立函数栈）
- 解法：HITL 移到节点或 pre_model_hook 里（第 14/17 章）

**resume 之后好像"从头跑了一遍"**
- 原因：正常现象的一半——**恢复时所在节点整体重跑**，interrupt 之前的历史值自动回放（第 14 章）
- 解法：节点内副作用放 interrupt 之后且幂等；多轮 interrupt 用循环写法（LangGraph 按调用序号匹配历史值）

**`get_state(config).next` 为空但图"没跑完"的感觉**
- 原因：图实际已到 END（next=() 就是结束）；或没配 checkpointer 时根本查不到
- 解法：看 `values` 与 `messages`；确认 checkpointer

**SqliteSaver 并发写锁错误（database is locked）**
- 原因：SQLite 单写者，多线程/多进程争用
- 解法：开发用没事，生产换 PostgresSaver（第 11 章选型）

## C.4 工具与 Agent 类

**模型从不调用工具**
- 排查顺序：① 模型是否支持 function calling ② `bind_tools` 是否生效 ③ 工具 docstring 是否描述清楚"什么时候用" ④ prompt 是否明确"遇到 X 应调用工具"
- 解法：先在纯 LCEL 里手动验证 `llm.bind_tools(...).invoke(...)` 的 tool_calls（第 16 章）

**工具抛异常 → 整个 run 崩了**
- 原因：ToolNode 外的自定义节点里调工具
- 解法：用 ToolNode（默认把错误转 ToolMessage 回传模型）；或 try/except 后返回错误字符串（第 16 章"错误还给模型"）

**Agent 循环调同一个工具停不下来**
- 原因：工具结果让模型不满意（返回错误/空值），或提示鼓励"多验证"
- 解法：工具错误信息给出"下一步建议"；限制重试轮数；recursion_limit 兜底

**response_format 结果缺失**
- 原因：模型在产出结构化结果前又发起了 tool_call；或该模型不支持结构化输出
- 解法：prompt 明确"信息足够后先停止工具调用，再输出最终结果"；换支持的模型

## C.5 流式与 Server 类

**stream 拿不到 token（只有最终结果）**
- 原因：mode 用了 values/updates（节点级）；或 Server 侧没开流式 run（用了 `runs.create` 而非 `runs.stream`）
- 解法：`stream_mode="messages"`；Server 用 stream/join（第 10/25 章）

**前端 SSE 乱序/断线丢消息**
- 解法：断线用 `join_stream` 重连续流（第 25 章）；消息带 node/step 元数据排序

**`langgraph dev` 启动报图加载失败**
- 原因：`langgraph.json` 的 graphs 路径/变量名写错；依赖没装进 CLI 的环境
- 解法：`"graphs": {"name": "./src/agent.py:graph"}` 核对；本地 `pip install -e .` 后重启（第 23 章）

**Studio 打不开/连不上**
- 解法：确认 `langgraph dev` 在跑且端口 2024；baseUrl 参数与之一致；浏览器 Mixed Content（https Studio 连 http 本地——官方已处理，若拦截改用本地代理或 http 入口）

**自托管容器起来就退出**
- 排查：环境变量缺 `REDIS_URI` / `DATABASE_URI`；许可证缺失；看 `docker logs`（第 26 章清单）

## C.6 性能与成本类

**越聊越慢、token 费暴涨**
- 原因：消息历史无限增长（没裁剪/摘要）
- 解法：trim + SummarizationMiddleware（第 11/18 章）；LangSmith 按节点看 token 曲线（第 27 章）

**Send 并行触发了供应商限流（429）**
- 解法：并发上限（信号量/`max_concurrency`）；错峰重试 RetryPolicy（第 28 章）

**延迟高但不知瓶颈**
- 解法：LangSmith 按 span 看 p95——通常是个慢工具而不是模型（第 28 章）
