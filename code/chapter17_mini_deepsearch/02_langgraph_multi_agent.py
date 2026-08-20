import json
import os
import sys
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, parse_obj_as, ValidationError
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_classic.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parent / ".env")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError("未检测到 ALI_BAILIAN_API_KEY，请检查 .env 文件配置！")

llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

# 一、构建规划智能体

PLANNER_INSTRUCTIONS = (
    "You are a helpful research assistant, Given a query, come up with a set of web searches "
    "to perform to best answer the query, Output between 5 and 7 terms to query for."
)

planner_prompt = ChatPromptTemplate.from_messages(
    [("system", PLANNER_INSTRUCTIONS), ("human", "{query}")]
)


class WebSearchItem(BaseModel):
    query: str
    "The search term to use for the web search."
    "用于网络搜索的关键词"

    reason: str
    "You reasoning for why this search is important to the query."
    "为什么这个搜索对于解答该问题很重要的理由"


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    "A list of web searches to perform to best answer the query"
    "为了尽可能全面回答该问题而需要执行的网页搜索列表"


planner_chain = planner_prompt | llm.with_structured_output(WebSearchPlan)


# 二、构建搜索智能体

SEARCH_INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300"
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good"
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)

search_tool = TavilySearch(max_results=5, topic="general")

search_agent = create_agent(
    model=llm, tools=[search_tool], system_prompt=SEARCH_INSTRUCTIONS
)


# 三、构建编写智能体

WRITER_PROMPT = (
    "You are a senior researcher tasked with writing a cohesive report for a research query."
    "You will be provided with the original query and some initial research.\n\n"
    "1. 先给出完整的大纲;\n"
    "2. 然后生成正式报告。\n\n"
    "**写作要求**:\n"
    "。报告使用 Markdown 格式;\n"
    "。 章节清晰，层次分明;\n"
    "markdown_report部分至少包含2000中文字(注意需要用中文进行回复);\n"
    "。内容丰富、论据充分，可加入引用和数据,允许分段、添加引用、表格等;\n"
    "。最终仅返回 JSON:\n"
    '{{"short summary":"."markdown report":"...","follow up questions": ["..."]}}'
)


class ReportData(BaseModel):
    short_summary: str
    """
    A short 2-3 sentence summary of this findings
    一份2-3句话的简短研究结论摘要
    """

    markdown_report: str
    """
    The final report
    最终生成的报告(markdown格式)
    """

    follow_up_questions: list[str]
    """
    Suggested topics to research further
    建议进一步研究的相关主题
    """


writer_prompt = ChatPromptTemplate.from_messages(
    [("system", WRITER_PROMPT), ("human", "{query}")]
)

writer_chain = writer_prompt | llm.with_structured_output(ReportData)


# 四、 封装LangGraph节点


def planner_node(state: MessagesState):
    user_query = state["messages"][-1].content
    raw = planner_chain.invoke({"query": user_query})

    # 这里要注意的是 执行结果可能是WebSearchPlan类型，也可能是字典类型（被python解析了), 为了严谨性，这里加一个捕捉一场逻辑
    try:
        plan = parse_obj_as(WebSearchPlan, raw)
    except ValidationError:
        if isinstance(raw, dict) and isinstance(raw.get("searches"), list):
            plan = WebSearchPlan(
                searches=[WebSearchItem(query=q, reason=r) for q, r in raw["searches"]]
            )
        else:
            raise

    return {
        "plan": plan,  # 保存原生对象到状态中，后面节点也可以直接使用
        "messages": [AIMessage(content=plan.model_dump_json())],
    }


def search_node(state: MessagesState):
    plan_json = state["messages"][-1].content
    plan = WebSearchPlan.model_validate_json(plan_json)

    summaries = []
    for item in plan.searches:
        run = search_agent.invoke({"messages": [HumanMessage(content=item.query)]})
        msgs = run["messages"]
        # 取可读内容：也就是最后一条ToolMessage 或 AIMessage的内容
        readable = next(
            (m for m in reversed(msgs) if isinstance(m, (ToolMessage, AIMessage))),
            msgs[-1],
        )
        summaries.append(f"## {item.query}\n\n{readable.content}")
    combined = "\n\n".join(summaries)
    return {"messages": [AIMessage(content=combined)]}


def writer_node(state: MessagesState):
    original_query = state["messages"][0].content
    combined_summary = state["messages"][-1].content

    writer_input = f"原始问题: {original_query}\n\n" f"搜索摘要：\n{combined_summary}"

    report: ReportData = writer_chain.invoke({"query": writer_input})

    return {"messages": [AIMessage(content=report.model_dump_json())]}


# 构建图
builder = StateGraph(MessagesState)
builder.add_node("planner_node", planner_node)
builder.add_node("search_node", search_node)
builder.add_node("writer_node", writer_node)

builder.add_edge(START, "planner_node")
builder.add_edge("planner_node", "search_node")
builder.add_edge("search_node", "writer_node")
builder.add_edge("writer_node", END)

graph = builder.compile()

mermaid_code = graph.get_graph().draw_mermaid()
print(mermaid_code)

initial_state = {
    "messages": [HumanMessage(content="请生成一份关于人工智能伦理的研究报告")]
}
final_state = graph.invoke(initial_state)

print(final_state["messages"][-1].content)
