# 第 8 章：子图 Subgraph 与图组合

编译后的图**本身就是一个节点**——这是 LangGraph 最强大的组合机制，也是多 Agent 系统的基石。

## 8.1 基本用法：图作为节点（共享 State）

父子图共享同一个 State schema，直接挂：

```python
# 子图：一个完整的"检索子流程"
child_builder = StateGraph(SharedState)
child_builder.add_node("web_search", web_search)
child_builder.add_node("rerank", rerank)
child_builder.add_edge(START, "web_search")
child_builder.add_edge("web_search", "rerank")
child_builder.add_edge("rerank", END)
child_graph = child_builder.compile()

# 父图：把整个子图当一个节点
parent_builder = StateGraph(SharedState)
parent_builder.add_node("retrieval_unit", child_graph)   # ← 图即节点
parent_builder.add_node("answer", answer_node)
parent_builder.add_edge(START, "retrieval_unit")
parent_builder.add_edge("retrieval_unit", "answer")
parent_builder.add_edge("answer", END)
parent = parent_builder.compile(checkpointer=InMemorySaver())
```

执行时：进入 `retrieval_unit` → 完整跑一遍子图（web_search → rerank）→ 返回父图继续。
对外部调用者来说，子图内部结构是黑盒，LangSmith tracing 里可以展开细看。

## 8.2 不同 State schema：写"适配器"节点

父子 State 不一致时（多数真实场景），用一个普通函数做转换，函数内部调子图：

```python
# 子图用自己的状态（比如一个通用 RAG 组件，不想耦合父图 schema）
class RagState(TypedDict):
    question: str
    docs: list
    answer: str

rag_graph = StateGraph(RagState) ... .compile()


# 父图节点：翻译进 → 调子图 → 翻译出
def rag_node(parent_state: ParentState) -> dict:
    result = rag_graph.invoke({
        "question": parent_state["user_input"],
        "docs": [],
        "answer": "",
    })
    return {"final_answer": result["answer"]}   # 只把父图需要的字段带回去
```

这是**组件化复用**的标准姿势：把通用能力（RAG、代码执行、审阅流程）封装成子图，父图只对接输入输出。

## 8.3 子图拥有独立 checkpointer

```python
child = child_builder.compile(checkpointer=child_checkpointer)  # 可独立持久化
```

用途：
- 子图线程独立可回放（比如"重跑某次检索流程"）
- 已有部署好的子图服务，直接作为"远程节点"接入
- 父子可用不同存储（父图 Postgres，子图内存）

注意：子图独立 checkpointer 时，父图的 thread 概念对子图不可见，跨层时间旅行需要分别操作。

## 8.4 Command.PARENT：子图反向控制父图

子图节点可以把"下一步去哪"的决定权交给父图：

```python
# 子图内部
def escalate(state: ChildState) -> Command:
    return Command(
        goto="human_review",          # 父图中的节点名！
        update={"needs_human": True},
        graph=Command.PARENT,         # 关键：跳转作用于父图
    )
```

这正是多 Agent "handoff（交接）"的底层机制：子 Agent 完成或放弃时，直接把控制权交回父图并附带结果（第 21 章大量使用）。

## 8.5 子图与外层的交互规则汇总

| 能力 | 行为 |
|---|---|
| 状态 | 共享 schema 直通；不同 schema 需适配函数 |
| 流式输出 | 默认折叠为子图整体；`subgraphs=True` 展开（第 10 章） |
| interrupt | 子图内的 `interrupt()` 会正常冒泡暂停整棵树（第 14 章） |
| 递归 | 子图可以嵌套子图，无层数限制（别嵌太深，调试痛苦） |
| langsmith | trace 中子图是可展开的嵌套 span |

## 8.6 设计模式：什么时候拆子图

**该拆**：
- 一段流程要**复用**（多个父图/多个项目用）
- 一段流程要**独立演进**（RAG 组件单独团队维护）
- 图太大画不出来了（子图是"函数"概念的图版）

**不该拆**：
- 只是"几个节点一组"——用注释和文件组织即可，拆了反而 trace 跳来跳去难调试
- 父子状态需要高频细粒度交互——说明它们本来就是一个图

经验：**先写在一张图里，出现复用需求时再抽子图**。 premature abstraction 在图编程里同样成立。

## 本章小结

- 编译后的图可以整体作为节点：共享 schema 直通、不同 schema 写适配器
- 子图可带独立 checkpointer，独立持久化与回放
- `Command(graph=Command.PARENT)` 是子图反向控制父图、实现 handoff 的机制
- 子图的 interrupt / streaming 会正确冒泡与展开
- 先直写后抽取，避免过度组件化

> 下一章：Functional API——不用画图也能享受持久化运行时。
