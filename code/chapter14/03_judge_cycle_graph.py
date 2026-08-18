from pydantic import BaseModel
from langgraph.graph import START, StateGraph, END


class BranchCycleState(BaseModel):
    x: int
    done: bool | None = False  # 字段名尽量和节点名称不要重复，容易混淆


def check_x(state: BranchCycleState) -> BranchCycleState:
    print(f"[check_x] 当前 x = {state.x}")
    return state


def is_even(state: BranchCycleState) -> bool:
    return state.x % 2 == 0


def increment(state: BranchCycleState) -> BranchCycleState:
    print(f"[increment] x 是偶数，执行 +1 → {state.x + 1}")
    return BranchCycleState(x=state.x + 1, done=False)


def done(state: BranchCycleState) -> BranchCycleState:
    print(f"[done] x 是奇数，流程结束")
    return BranchCycleState(x=state.x, done=True)


# 构建图
builder = StateGraph(BranchCycleState)

builder.add_node("check_x", check_x)
builder.add_node("increment", increment)
builder.add_node("done_node", done)

builder.add_edge(START, "check_x")
builder.add_conditional_edges(
    "check_x", is_even, {True: "increment", False: "done_node"}
)

builder.add_edge("increment", "check_x")
builder.add_edge("done_node", END)

graph = builder.compile()

mermaid_code = graph.get_graph().draw_mermaid()
# print("\n--- Mermaid 语法 ---")
# print(mermaid_code)

# 测试执行
print("\n初始 x=6（偶数，进入循环）")
final_state1 = graph.invoke(BranchCycleState(x=6))
print("[最终结果1] ->", final_state1)

print("\n初始 x=3（奇数，直接 done）")
final_state2 = graph.invoke(BranchCycleState(x=3))
print("[最终结果2] ->", final_state2)
