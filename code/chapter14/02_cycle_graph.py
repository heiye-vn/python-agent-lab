from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END


# 定义结构化状态
class CycleState(BaseModel):
    x: int


# 构建图
builder = StateGraph(CycleState)


# 定义递增函数节点
def increment(state: CycleState) -> CycleState:
    print(f"[increment] 当前 x = {state.x}")
    return CycleState(x=state.x + 1)


builder.add_node("increment", increment)

builder.add_edge(START, "increment")

builder.add_conditional_edges(
    "increment", lambda state: state.x > 10, {True: END, False: "increment"}
)

# 创建图
graph = builder.compile()

mermaid_code = graph.get_graph().draw_mermaid()
print("\n--- Mermaid 语法 ---")
print(mermaid_code)


print("\n 执行循环直到 x > 10")
final_state = graph.invoke(CycleState(x=6))
print(f"[最终结果] -> x = {final_state['x']}")
