"""
from pydantic import BaseModel

class MyState(BaseModel):
    x: int
    y: str = "default"   # 设置默认值

# 自动校验
state = MyState(x=1)
print(state.x)       # 输出 1
print(state.y)       # 输出 default

# 错误类型会报错
state = MyState(x="abc")
"""

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END


# 定义结构化状态模型
class CalcState(BaseModel):
    x: int = Field(
        default=15,
        description="初始状态",
    )


# 定义节点函数，接收并返回 CalcState
def addition(state: CalcState) -> CalcState:
    print(f"【addition】初始状态: {state}")
    return CalcState(x=state.x + 1)


def subtraction(state: CalcState) -> CalcState:
    print(f"【subtraction】接收到状态: {state}")
    return CalcState(x=state.x - 2)


# 构建图
builder = StateGraph(CalcState)

builder.add_node("addition", addition)
builder.add_node("subtraction", subtraction)

builder.add_edge(START, "addition")
builder.add_edge("addition", "subtraction")
builder.add_edge("subtraction", END)

agent = builder.compile()

mermaid_code = agent.get_graph().draw_mermaid()
print("\n--- Mermaid 语法 ---")
print(mermaid_code)

# 执行图：传入结构化状态对象
# initial_state = CalcState(x=10)
initial_state = CalcState()
final_state = agent.invoke(initial_state)

print("\n[最终结果] ->", final_state)
