# 附录 D：高频面试题与答案要点

按主题分组，每题给**要点式答案**（面试时展开成口述）。标注对应章节便于回看。

## D.1 基础概念

**Q1：LangGraph 和 LangChain 的区别？为什么需要 LangGraph？**
- 要点：LangChain 是接口层（模型/工具标准化），LangGraph 是编排与运行时层（状态机图 + 持久化执行）
- 核心动机：链（DAG）表达不了 Agent 的**循环**；图可以。再加运行时四能力：持久化、流式、HITL、恢复（第 1 章）
- 加分：LangGraph 可脱离 LangChain 单独用；执行模型是 Pregel 式 superstep

**Q2：解释 State、Node、Edge 的关系，状态更新是怎么发生的？**
- 要点：State 全图共享；节点返回 **partial update**；按字段的 **reducer 合并**（默认覆盖 / add_messages 追加 / operator.add 拼接 / 自定义需满足交换律）
- 加分：无 reducer 字段禁止并行写（InvalidUpdateError）；节点级私有状态（类型注解裁剪）（第 3、4 章）

**Q3：superstep 是什么？并行是怎么发生的？**
- 要点：一轮内所有被触发节点并发执行，全部完成后合并状态进入下一轮；静态多出边 = fan-out；Send = 运行时动态 map-reduce（每实例独立输入、同轮并发、reducer 汇聚）（第 7 章）

**Q4：Command 和条件边怎么选？**
- 要点：条件边=图结构层纯路由；Command=节点内"更新+跳转"一体（goto 支持 Send/PARENT），1.x 官方推荐；简单静态分支保留条件边利于可视化（第 5 章）

## D.2 持久化与记忆

**Q5：讲讲 Checkpointer 的原理和作用。**
- 要点：每 superstep 把状态快照写入 thread 的检查点链 → 支撑记忆/恢复/HITL/时间旅行；崩溃后 `invoke(None, config)` 从最后完整快照续跑；**节点是重试原子单位**，节点内副作用必须幂等（第 11 章）
- 加分：选型 InMemory/SQLite/Postgres/Redis；thread 是会话坐标

**Q6：短期记忆和长期记忆的区别与实现？**
- 要点：短期=thread 内对话历史（checkpointer），长期=跨 thread（Store，namespace+key+value，可配向量索引语义检索）；长期记忆三类型 semantic/episodic/procedural；写入两策略（热路径工具 + 会话后提炼）（第 12 章）
- 加分：长话治理 trim + 滚动摘要；记忆注入做预算控制

**Q7：时间旅行怎么实现？有什么用？**
- 要点：检查点链即历史；replay=同 thread + checkpoint_id + invoke(None)（可先 update_state）；fork=新 thread + 历史 checkpoint_id（平行宇宙）
- 用途：断点重跑省 token、A/B 对比、撤销回滚、事故复盘（第 13 章）

## D.3 HITL

**Q8：LangGraph 的 interrupt 是怎么工作的？暂停时系统资源占用？**
- 要点：interrupt(payload) → 状态落盘（含"停在哪个节点"）→ 本次调用正常返回；**不阻塞不占资源**，等几天都行；恢复=`invoke(Command(resume=...), 同 thread)`，resume 值成为 interrupt() 返回值；恢复时节点重跑、历史 interrupt 自动回放
- 加分：与 interrupt_before/after 的区别；NodeInterrupt 条件异常打断；工程闭环（通知/审批 UI/审计/超时升级）（第 14、15 章）

## D.4 Agent 与多 Agent

**Q9：create_react_agent 的执行流程？如何扩展它？**
- 要点：模型→(tool_calls? tools→模型循环):END；扩展四插槽：动态 prompt（注入上下文）、pre/post_model_hook（HITL/裁剪）、state_schema（业务字段）、middleware（guardrails/权限/模型路由）；response_format 结构化终局输出（第 17、18 章）

**Q10：Supervisor 和 Swarm 的区别？**
- 要点：Supervisor 中心路由（结构化输出派活+汇总），可控、适合工作流任务；Swarm 对等 handoff（交接工具），去中心、适合对话分诊；成本：supervisor 每轮多一次调用 vs swarm 有乒乓风险（第 21 章）

**Q11：什么时候该用多 Agent？收益的本质是什么？**
- 要点：两大真实收益=职责聚焦（提示/工具子集）+ **上下文隔离**（子 Agent 消耗自己窗口只回传结论）；触发条件是"指令冲突/上下文爆炸成为实际瓶颈"；警惕过度设计（成本×N、调试×N）（第 20 章）
- 加分：通信两机制（共享状态 vs 消息 handoff）；"确定性骨架 + 智能决策点"架构观

**Q12：Deep Agents 是什么，适合什么任务？**
- 要点：create_react_agent 之上内置规划（todos）、子 Agent（task）、虚拟文件系统、上下文管理；适合小时级长任务（深度研究/报告/代码分析）；生产配预算+产出校验+持久化（第 22 章）

## D.5 部署与生产

**Q13：LangGraph Server 的核心对象？为什么不能 FastAPI 自己包一层？**
- 要点：Agent→Assistant→Thread→Run + Store + Cron；Run 三模式（后台/流式/无状态）；暂停 run 零资源占用；状态外置 DB → 水平扩展 + 断线 join 重连；自建会撞的墙：长任务连接占用、多实例状态共享、任务队列、统一 API（第 24 章）

**Q14：生产上如何保证 Agent 系统的安全？**
- 要点：注入防御五层（输入过滤/最小权限/数据标记/输出校验/业务层二次鉴权）；核心心法：**模型输出只是建议，权限在代码**；身份信息运行时注入（InjectedState/config），密钥零入图、trace 脱敏（第 28 章）

**Q15：Agent 上线后怎么评估和迭代？**
- 要点：LangSmith tracing（按节点定位）+ 数据集回归（确定性断言：工具命中；LLM 判官：回答质量）+ Prompt Hub 版本化 + Assistant 灰度 + Engine 自动诊断；变更三分离（代码/提示/配置）（第 27 章）

**Q16：成本失控怎么治？**
- 要点：模型分级路由、fallbacks 降级链、trim+摘要、CachePolicy/FAQ 缓存、prompt caching、工具结果限量、预算硬上限+告警（第 28 章）

## D.6 设计与场景题

**Q17：设计一个"退款客服机器人"，讲出关键组件。**
- 答题骨架：create_react_agent + 订单/政策/退款工具（JSON 限量返回）→ 危险工具 pre_model_hook interrupt 审批（resume 驱动）→ PostgresSaver 记忆 + 长期偏好 Store → Server 化（thread 关联工单、Webhook 通知审批、Cron 超时）→ LangSmith 工具命中率回归（完整对照第 29 章）

**Q18：一个流程 80% 步骤确定、20% 需要智能判断，怎么设计？**
- 要点：确定性 StateGraph 做骨架，智能点嵌 create_react_agent 子图/结构化输出节点；LLM 决策全部枚举+白名单；关键门禁用规则硬卡（第 32 章审批流就是范本）

**Q19：RAG 用固定管线还是 Agent 化？**
- 要点：固定管线=低延迟可控，Agent 化=自适应多轮检索/改写 query/自知不知道；高频入口管线+缓存，对话式助手 Agent 化，可路由混合（第 30 章）

**Q20：手写一个最小 ReAct Agent（白板题）**
- 答题骨架（~15 行）：bind_tools → call_model 节点 → `should_continue`（tool_calls? tools:END）→ ToolNode → 边回模型 → compile(checkpointer)（第 16 章末尾原文可背）
