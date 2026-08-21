import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_classic.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
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

# ============================================================
# 一、构建规划智能体
# ============================================================

PLANNER_INSTRUCTIONS = (
    "You are an expert research planner. Today's date is {current_date} (Year: {current_year}).\n"
    "Given a user query, formulate a comprehensive, up-to-date search plan to gather all necessary information.\n"
    "Requirements:\n"
    "1. Time Sensitivity: Current time is {current_year}. When dealing with queries containing words like '当下', '最新', '当前', '趋势' or time-sensitive topics, "
    "always formulate search queries targeting {current_year} or latest information. STRICTLY DO NOT use past years (like 2023, 2024) to represent 'current/now'.\n"
    "2. Generate 5 to 7 specific and diverse search terms.\n"
    "3. Match the language of search queries with the user query (e.g., use Chinese if query is in Chinese).\n"
    "4. Provide a clear reasoning for why each search term is essential."
)
planner_prompt = ChatPromptTemplate.from_messages(
    [("system", PLANNER_INSTRUCTIONS), ("human", "{query}")]
)


class WebSearchItem(BaseModel):
    query: str = Field(description="用于网络搜索的关键词，需与用户查询语言保持一致")
    reason: str = Field(description="为什么该搜索词对解答问题至关重要的理由")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="为了全面回答问题而需要执行的网络搜索列表（5-7项）"
    )


planner_chain = planner_prompt | llm.with_structured_output(WebSearchPlan)


# ============================================================
# 二、构建搜索智能体
# ============================================================

_now = datetime.now()
_current_date_str = _now.strftime("%Y-%m-%d")
_current_year_str = str(_now.year)

SEARCH_INSTRUCTIONS = (
    f"You are a research assistant. Current date is {_current_date_str} (Year: {_current_year_str}).\n"
    "Given a search term, search the web and produce a concise, up-to-date summary.\n"
    "Requirements:\n"
    f"1. Prioritize recent and latest facts and data corresponding to the current time context ({_current_year_str}).\n"
    "2. Summary must be 2-3 paragraphs and under 300 words.\n"
    "3. Capture key facts, data, and main points while ignoring fluff.\n"
    "4. Write succinctly in the same language as the search query.\n"
    "5. Do not include introductory or concluding commentary, only the summary itself."
)

search_tool = TavilySearch(max_results=5, topic="general")

search_agent = create_agent(
    model=llm, tools=[search_tool], system_prompt=SEARCH_INSTRUCTIONS
)


# ============================================================
# 三、构建编写智能体
# ============================================================

WRITER_PROMPT = (
    "你是一位资深行业研究员。当前基准时间是 {current_date}（{current_year}年）。\n"
    "负责根据用户的原始问题和多维度的网络搜索摘要，撰写一份结构严谨、内容详实、具有鲜明时效性的专业研究报告。\n\n"
    "**报告要求**：\n"
    "1. **时效性与视角**：报告必须立足于 {current_year} 年的当下视角。对于 2024/2025 年的历史信息请标明为背景或演进历程，重点突出当前（{current_year}年）的市场现状、技术要求与趋势；\n"
    "2. **结构规范**：使用标准 Markdown 格式，报告开头需包含内容大纲，正文层级分明（使用 H1/H2/H3）；\n"
    "3. **内容详实**：充分融合搜索摘要中的事实、数据和论据，支持添加表格、分段及引用；\n"
    "4. **字数与语言**：报告正文（markdown_report）须使用中文撰写，内容充实完整；\n"
    "5. **结构化输出**：请直接填充指定的输出结构，无需在文本外包裹多余说明。"
)


class ReportData(BaseModel):
    short_summary: str = Field(description="2-3句话的核心研究结论摘要")
    markdown_report: str = Field(
        description="完整的研究报告正文（Markdown格式，包含大纲和详细章节）"
    )
    follow_up_questions: list[str] = Field(
        description="建议进一步深入研究的3-5个相关问题"
    )


writer_prompt = ChatPromptTemplate.from_messages(
    [("system", WRITER_PROMPT), ("human", "{query}")]
)

writer_chain = writer_prompt | llm.with_structured_output(ReportData)


# ============================================================
# 四、封装 LangGraph 节点
# ============================================================


def planner_node(state: MessagesState):
    """规划节点：根据用户问题生成搜索方案"""
    user_query = state["messages"][-1].content
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_year = str(now.year)

    print(f"\n{'='*60}")
    print(f"📋 [规划阶段] 正在为问题制定搜索方案 (基准时间: {current_date})...")
    print(f"   用户问题: {user_query}")
    print(f"{'='*60}")

    start = time.time()
    raw = planner_chain.invoke(
        {
            "query": user_query,
            "current_date": current_date,
            "current_year": current_year,
        }
    )

    # with_structured_output 可能返回 WebSearchPlan 实例或 dict，统一处理
    if isinstance(raw, dict):
        plan = WebSearchPlan.model_validate(raw)
    else:
        plan = raw

    elapsed = time.time() - start
    print(
        f"\n✅ [规划完成] 耗时 {elapsed:.1f}s，共生成 {len(plan.searches)} 条搜索任务："
    )
    for i, item in enumerate(plan.searches, 1):
        print(f"   {i}. {item.query}")
        print(f"      └─ 理由: {item.reason}")

    return {"messages": [AIMessage(content=plan.model_dump_json())]}


def search_node(state: MessagesState):
    """搜索节点：逐条执行搜索任务并汇总摘要"""
    plan_json = state["messages"][-1].content
    plan = WebSearchPlan.model_validate_json(plan_json)

    total = len(plan.searches)
    print(f"\n{'='*60}")
    print(f"🔍 [搜索阶段] 开始执行 {total} 条搜索任务...")
    print(f"{'='*60}")

    summaries = []
    for i, item in enumerate(plan.searches, 1):
        print(f"\n   🔎 [{i}/{total}] 搜索: {item.query} ...", end="", flush=True)
        start = time.time()

        run = search_agent.invoke({"messages": [HumanMessage(content=item.query)]})
        msgs = run["messages"]

        # 优先取最后一条 AIMessage（Agent 的总结），fallback 到 ToolMessage（原始搜索结果）
        readable = next(
            (m for m in reversed(msgs) if isinstance(m, AIMessage)),
            next(
                (m for m in reversed(msgs) if isinstance(m, ToolMessage)),
                msgs[-1],
            ),
        )

        elapsed = time.time() - start
        # 截取摘要前80字符作为预览
        preview = readable.content[:80].replace("\n", " ")
        print(f" 完成 ({elapsed:.1f}s)")
        print(f"      └─ 摘要预览: {preview}...")

        summaries.append(f"## {item.query}\n\n{readable.content}")

    combined = "\n\n".join(summaries)
    print(f"\n✅ [搜索完成] 全部 {total} 条搜索任务已完成")
    return {"messages": [AIMessage(content=combined)]}


def writer_node(state: MessagesState):
    """撰写节点：基于搜索摘要生成研究报告"""
    original_query = state["messages"][0].content
    combined_summary = state["messages"][-1].content
    now = datetime.now()
    current_date = now.strftime("%Y年%m月%d日")
    current_year = str(now.year)

    print(f"\n{'='*60}")
    print(f"📝 [撰写阶段] 正在生成研究报告 (基准时间: {current_date})...")
    print(f"   原始问题: {original_query}")
    print(f"   搜索摘要长度: {len(combined_summary)} 字符")
    print(f"{'='*60}")

    writer_input = f"原始问题: {original_query}\n\n" f"搜索摘要：\n{combined_summary}"

    start = time.time()
    report: ReportData = writer_chain.invoke(
        {
            "query": writer_input,
            "current_date": current_date,
            "current_year": current_year,
        }
    )
    elapsed = time.time() - start

    print(f"\n✅ [撰写完成] 耗时 {elapsed:.1f}s")
    print(f"   核心摘要: {report.short_summary}")
    print(f"   报告长度: {len(report.markdown_report)} 字符")
    print(f"   后续问题: {len(report.follow_up_questions)} 条")

    return {"messages": [AIMessage(content=report.model_dump_json())]}


# ============================================================
# 五、构建 LangGraph 图
# ============================================================

builder = StateGraph(MessagesState)
builder.add_node("planner_node", planner_node)
builder.add_node("search_node", search_node)
builder.add_node("writer_node", writer_node)

builder.add_edge(START, "planner_node")
builder.add_edge("planner_node", "search_node")
builder.add_edge("search_node", "writer_node")
builder.add_edge("writer_node", END)

graph = builder.compile()

if __name__ == "__main__":
    # 打印图结构
    mermaid_code = graph.get_graph().draw_mermaid()
    print("📊 图结构（Mermaid）:")
    print(mermaid_code)

    query = "请生成一份当下中国招聘市场对 Ai Agent 应用开发工程师/岗位的需求分析报告"
    print(f"\n🚀 启动 Mini DeepSearch 多智能体研究流程")
    print(f"   问题: {query}")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()
    initial_state = {"messages": [HumanMessage(content=query)]}
    final_state = graph.invoke(initial_state)
    total_elapsed = time.time() - total_start

    # 解析并展示最终报告
    print(f"\n{'='*60}")
    print(f"🎉 全部流程完成！总耗时 {total_elapsed:.1f}s")
    print(f"{'='*60}")

    try:
        report = ReportData.model_validate_json(final_state["messages"][-1].content)
        print(f"\n📌 核心摘要:\n{report.short_summary}")
        print(f"\n📄 完整报告:\n{report.markdown_report}")
        print(f"\n❓ 建议深入研究的问题:")
        for i, q in enumerate(report.follow_up_questions, 1):
            print(f"   {i}. {q}")
    except Exception:
        # 兜底：直接输出原始内容
        print(final_state["messages"][-1].content)
