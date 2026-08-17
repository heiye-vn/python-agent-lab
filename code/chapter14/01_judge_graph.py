from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END


# 定义结构化状态
class MyState(BaseModel):
    x: int
    result: str | None = None


# 构建图
builder = StateGraph(MyState)


# 定义各节点处理逻辑
def check_x(state: MyState) -> MyState:
    print(f"[check_x] Received state: {state}")
    return state


def handle_even(state: MyState) -> MyState:
    print(f"[handle_even] x 是偶数")
    return MyState(x=state.x, result="even")


def handle_odd(state: MyState) -> MyState:
    print(f"[handle_odd] x 是奇数")
    return MyState(x=state.x, result="odd")


builder.add_node("check_x", check_x)
builder.add_node("handle_even", handle_even)
builder.add_node("handle_odd", handle_odd)


# 奇偶判断
def is_even(state: MyState) -> bool:
    return state.x % 2 == 0


# 添加条件分支，add_conditional_edges(起始节点, 条件判断函数, 条件路径映射字典)，条件路径映射可以有多条处理路径（类似 match...case）
builder.add_conditional_edges(
    "check_x", is_even, {True: "handle_even", False: "handle_odd"}
)

# 衔接起始和结束，add_edge(起始节点, 结束节点)
builder.add_edge(START, "check_x")
builder.add_edge("handle_even", END)
builder.add_edge("handle_odd", END)

# 创建图
graph = builder.compile()

mermaid_code = graph.get_graph().draw_mermaid()
print("\n--- Mermaid 语法 ---")
print(mermaid_code)


print("\n测试 x=4（偶数）")
graph.invoke(MyState(x=4))

print("\n测试 x=3（奇数）")
graph.invoke(MyState(x=3))
