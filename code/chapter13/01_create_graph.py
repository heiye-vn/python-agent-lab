from langgraph.graph import StateGraph

# 创建一个字典类型的State
builder = StateGraph(dict)


def addition(state):
    print(f"加法节点收到的初始值:{state}")
    return {"x": state["x"] + 1}


def subtraction(state):
    print(f"减法节点收到的初始值:{state}")
    return {"x": state["x"] - 2}


from langgraph.graph import START, END

# 向图中添加两个节点
builder.add_node("addition", addition)
builder.add_node("subtraction", subtraction)

# 构建节点之间的边
builder.add_edge(START, "addition")
builder.add_edge("addition", "subtraction")
builder.add_edge("subtraction", END)

print(builder.nodes)
print(builder.edges)

graph = builder.compile()

# graph.get_graph().print_ascii()

mermaid_code = graph.get_graph().draw_mermaid()
print("\n--- Mermaid 语法 ---")
print(mermaid_code)

initial_state = {"x": 10}

result = graph.invoke(initial_state)

print("最后的结果:", result)
