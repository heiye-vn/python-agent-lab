# 第 20 章：多 Agent 架构总览

多 Agent 不是玄学，本质一句话：**把大任务拆给多个有独立上下文/工具/提示的角色，用一个编排层协调它们**。本章建立正确的架构观，第 21 章动手实现。

## 20.1 为什么需要多 Agent（真实收益）

| 痛点 | 多 Agent 解法 |
|---|---|
| 单 Agent 提示塞入所有职责 → 指令互相干扰、准确率下降 | 每个角色一份聚焦的小提示 |
| 工具太多 → 模型选择困难（第 16 章） | 每个角色只带自己的工具子集 |
| 长任务上下文爆炸 | **子 Agent 消耗自己的上下文，只回传结论**（最大收益） |
| 需要"分头查、汇总答" | 并行执行天然提速 |
| 不同步骤要不同模型（贵模型做规划，便宜模型做抽取） | 按角色配模型 |

**同时警惕过度设计**：能用一个好提示 + 三个工具解决的单 Agent，不要上多 Agent。多 Agent 的成本：调试复杂度×N、token 消耗×N、延迟叠加。判定标准：**当"职责冲突"或"上下文隔离"成为实际瓶颈时才拆**。

## 20.2 五种基础拓扑

```
1. Supervisor（主管）          2. Swarm（对等交接）
      ┌─ worker A                  A ⇄ B ⇄ C
boss ─┼─ worker B                  （无中心，互相 handoff）
      └─ worker C

3. Network（网状）             4. Hierarchical（层级）
   A→B, B→A, A→C, C→A…            supervisor
                                      ├─ sub-team supervisor
   （灵活但难控，慎用）                 │    ├─ worker D
                                      │    └─ worker E
                                      └─ worker F

5. Pipeline（流水线）
   A → B → C（固定顺序，本质是图编排，不算真"多 Agent"）
```

选型速查：

| 需求 | 推荐 |
|---|---|
| 任务异构、需要中心调度决策 | Supervisor |
| 对话型场景、角色间直接转接用户 | Swarm |
| 组织型大任务（团队套团队） | Hierarchical |
| 步骤固定 | 别用多 Agent，StateGraph 流水线 |

## 20.3 两种通信机制（实现层面的根本分野）

### 方式一：共享状态（shared state）

所有 Agent 读写同一份图状态（或消息列表）：

```python
builder.add_node("researcher", researcher_agent)   # 都挂在同一张图上
builder.add_node("writer", writer_agent)
# researcher 写 state["findings"]，writer 读它
```

- ✅ 简单直接、信息全量可见
- ❌ 缺乏封装；角色互相耦合状态字段；上下文不隔离

### 方式二：消息传递 + handoff（交接）

每个 Agent 是**独立子图**（通常各自是 create_react_agent），用 `Command(goto=...)` 把控制权和"任务说明消息"交给下一个：

```python
# 在 Agent A 的工具里触发交接
transfer_to_writer → Command(goto="writer", update={"messages": [交接说明]})
```

- ✅ 上下文隔离（A 的 50 页阅读记录不进 B 的窗口）、角色可独立开发部署
- ❌ 信息靠消息传递，要设计交接协议

**行业趋势**：复杂系统用方式二（隔离收益大），方式一用于小组件内部。

## 20.4 Supervisor 的核心：路由决策

Supervisor 本质是一个"只做决策不做执行"的 LLM 节点：

```python
class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "critic", "FINISH"]
    task: str = Field(description="交给该成员的具体任务说明")

supervisor_llm = llm.with_structured_output(RouteDecision)

def supervisor(state) -> Command:
    decision = supervisor_llm.invoke([
        {"role": "system", "content": SUPERVISOR_PROMPT},
        *state["messages"],
    ])
    if decision.next == "FINISH":
        return Command(goto=END, update={"messages": [final_summary(state)]})
    # 交接：任务说明作为新消息交给成员
    return Command(
        goto=decision.next,
        update={"messages": [HumanMessage(content=decision.task, name="supervisor")]},
    )
```

要点：
- `Literal` 枚举 = 白名单路由（第 7 章健壮性）
- 交接消息带**具体任务说明**（不要只说"给 writer"，要说"基于 findings 写一份 500 字结论"）
- FINISH 是显式出口；supervisor 提示里要写清完成标准

## 20.5 多 Agent 与 LangGraph 概念的映射

| 多 Agent 概念 | LangGraph 机制 |
|---|---|
| 编排层 | 父图（StateGraph） |
| 单个 Agent | 子图（通常是 create_react_agent） |
| 交接/转接 | `Command(goto=..., update=交接消息)` |
| 子 Agent 结果上报 | `Command(goto=Command.PARENT, ...)` |
| 全局黑板 | 共享 State 字段 / Store |
| 并行分头工作 | 父图 fan-out 或 Send |

至此你会发现：**多 Agent 没有新机制，全是第 5、8、16、17 章已有能力的组合**。这正是 LangGraph 设计的优雅之处。

## 20.6 什么时候不用多 Agent（反面清单）

- 单角色 + 10 个以内工具能搞定 → 单 Agent
- 流程完全确定 → 普通 StateGraph（不需要 LLM 决策路由的"假多 Agent"）
- 为了"看起来高级" → 评审时会被打回（真实成本：token、延迟、可调试性）
- 需要极致延迟的对话场景 → 多跳交接的累积延迟往往不可接受

## 本章小结

- 拆多 Agent 的两大真实收益：职责聚焦 + 上下文隔离
- 五种拓扑：Supervisor / Swarm / Network / Hierarchical / Pipeline，各有适用
- 通信两机制：共享状态（简单）vs 消息 handoff（隔离，主流）
- Supervisor = 结构化输出路由节点；交接要带具体任务说明
- 多 Agent 无新机制，全是已有图能力的组合

> 下一章：用官方库把 Supervisor、Swarm 完整搭出来。
