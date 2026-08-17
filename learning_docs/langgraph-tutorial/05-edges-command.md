# 第 5 章：边 Edge 与 Command —— 两种控制流范式

LangGraph 1.x 有两套等价的路由方式：**条件边（老范式）** 和 **节点内 Command（新范式，官方推荐）**。都要会：读老代码用前者，写新代码优先后者。

## 5.1 静态边

```python
from langgraph.graph import StateGraph, START, END

builder.add_edge(START, "a")        # 入口
builder.add_edge("a", "b")          # a 完成后必走 b
builder.add_edge("b", END)          # 结束
```

**多条出边 = 静态并行（fan-out）**：

```python
builder.add_edge("a", "b")
builder.add_edge("a", "c")   # a 完成后，b 和 c 在同一个 superstep 并行执行
builder.add_edge("b", "join")   # join 等所有入边来源完成（下一轮执行）
builder.add_edge("c", "join")
```

## 5.2 条件边（老范式）

路由函数接收 state，返回"下一个节点名"：

```python
def route_by_intent(state: State) -> str:
    if state["intent"] == "refund":
        return "refund_agent"
    return "faq_agent"

builder.add_conditional_edges(
    "classifier",              # 从哪个节点出来后做判断
    route_by_intent,           # 路由函数
    {                          # 路径映射（可选，用于可视化与校验）
        "refund_agent": "refund_agent",
        "faq_agent": "faq_agent",
    }
)
```

要点：
- 返回值是**节点名**；返回列表 = 动态 fan-out（并行走多条）
- 路由函数也可以是 `(state, config) -> str` 双参数
- 路由函数**不应该有副作用**（不写库不发请求），它只做纯判断
- 缺点：跳转逻辑在图定义处，离节点代码远；无法在跳转同时更新状态

## 5.3 Command（新范式，1.x 推荐）

`Command` 让节点在返回时**同时声明"更新什么"和"跳去哪"**：

```python
from langgraph.types import Command

def classify_and_route(state: State) -> Command:
    if state["intent"] == "refund":
        return Command(
            goto="refund_agent",
            update={"priority": "high"},   # 跳转 + 状态更新一步完成
        )
    return Command(goto="faq_agent")
```

`goto` 的四种形态：

```python
Command(goto="node_a")                       # 单个节点
Command(goto=["node_a", "node_b"])           # 并行多个（fan-out）
Command(goto=Send("worker", {"task": 1}))    # 动态分发（第 7 章）
Command(goto=Command.PARENT)                 # 子图跳回父图的节点（第 8 章）
```

在子图中向父图汇报并跳转：

```python
# 子图节点内
return Command(
    goto=Command.PARENT,            # 控制权交还父图
    update={"subgraph_done": True}, # 更新父图的状态
    graph=Command.PARENT,           # 声明作用目标是父图
)
```

## 5.4 Command vs 条件边：怎么选

| 维度 | 条件边 | Command |
|---|---|---|
| 跳转时机 | 节点执行完，由图判断 | 节点内部主动声明 |
| 同时更新状态 | ❌ 需要额外节点 | ✅ `update=` 一步完成 |
| 可读性 | 分散（节点+图定义） | 集中在节点里 |
| 调试 | LangSmith 中是独立 step | 与节点执行在一起 |
| 动态/程序化跳转 | 弱（只能返回名字） | 强（可编排 Send、PARENT） |

官方建议：**新代码一律用 Command**；条件边用于非常简单、且想在图结构上清晰表达分支的场合（比如纯静态的 if-else 路由，画出来的图更直观）。

## 5.5 常用控制流模板速查

### 模板一：Router 节点模式（Command 版）

```python
builder.add_node("router", route_fn)          # 返回 Command
builder.add_edge(START, "router")
builder.add_edge("refund_agent", END)
builder.add_edge("faq_agent", END)
```

### 模板二：循环直到满足条件

```python
def should_continue(state) -> str:
    if state["enough_info"]:
        return "finalize"
    if state["search_rounds"] >= 3:
        return "give_up"
    return "search"          # 回到 search，形成循环

builder.add_conditional_edges("evaluate", should_continue)
```

### 模板三：带状态的跳转（等价于"函数调用后返回"）

```python
def call_subagent(state) -> Command:
    return Command(
        goto="expert_agent",
        update={"task": state["hard_question"]},
    )

def expert_agent(state):
    result = llm.invoke(...)
    return Command(goto="merge", update={"expert_answer": result.content})
```

## 5.6 特殊保留字与规则

- `START` / `END`：`langgraph.graph` 导出的保留节点名，不能用作自定义节点名
- 多个节点可以同时 `add_edge(START, x)` —— 多入口并行起点
- 到 `END` 不是必须的：Agent 类应用常常永远循环（靠 `recursion_limit` 或工具里的 `Command(goto=END)` 控制），但批处理图必须可达 `END`
- 条件边返回的节点名不存在 → 运行时才报错，compile 不查（Command 同理），注意别拼错

## 本章小结

- 静态边连固定流程；多出边即并行；多入边即汇聚
- 条件边：图结构层路由，纯判断、可画出漂亮流程图
- Command：节点内路由 + 状态更新，1.x 官方推荐
- `goto` 支持单节点 / 列表 / Send / Command.PARENT 四种形态

> 下一章：图的编译参数、校验与可视化。
