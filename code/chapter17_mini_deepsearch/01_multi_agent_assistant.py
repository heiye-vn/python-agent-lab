import os
import sys
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
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

# 定义系统指令，构建提示词
PLANNER_INSTRUCTIONS = f"""你是一名资深的深度研究规划专家。今天是 {datetime.now().strftime('%Y年%m月%d日')}。
你的任务是针对用户给出的【研究主题】，拆解并生成 3 ~ 5 个高质量、正交（互不重叠）的搜索引擎查询关键词（Search Queries）。
### 拆解与规划原则（MECE 原则）：
1. 多维度覆盖：生成的搜索词必须覆盖不同维度，避免同义重复。通常涵盖以下方向：
   - 核心定义与最新现状 / 市场规模（包含最新年份）
   - 关键技术架构与典型应用场景
   - 行业标杆案例与落地实践（包含具体产品/公司）
   - 现存痛点、合规风险或技术瓶颈
   - 未来发展趋势与前沿预测
2. 搜索词工程化（Search Query Engineering）：
   - 采用适合搜索引擎的高信息密度【关键词组合】，避免冗长的自然语言问句。
   - 示例：
     - ❌ 不佳：请问目前AI在医疗领域有哪些最新的应用落地案例
     - ✅ 推荐：AI 医疗 临床诊断 落地案例 2025 2026
3. 语言策略：
   - 若主题具有强国内/本地属性，主要生成中文关键词；
   - 若主题涉及前沿技术或全球趋势，可生成 1-2 个精准的英文核心术语进行全球检索。
4. 提供明确理由：
   - 对每个搜索词说明其研究意图（Reason），解释该搜索词能补充哪一部分关键信息。
"""
planner_prompt = ChatPromptTemplate.from_messages(
    [("system", PLANNER_INSTRUCTIONS), ("human", "{query}")]
)


# 定义结构化输出格式
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

# planner_result = planner_chain.invoke({"query": "请问你对AI+教育有何看法"})

# print(planner_result)


# 二、构建搜索智能体

SEARCH_INSTRUCTIONS = f"""你是一名信息检索与提炼专家。今天是 {datetime.now().strftime('%Y年%m月%d日')}。
你的任务是：针对给定的搜索关键词，必须先调用搜索引擎工具检索真实网页，并提炼出高信息密度的核心事实摘要，供下游报告撰写智能体使用。
### 提炼规范：
1. 工具调用：必须调用搜索工具获取最新网页内容，严禁仅凭预训练记忆胡编乱造。
2. 格式要求：以清晰的无序列表（Bullet Points）输出，总字数控制在 300 ~ 500 字以内。
3. 内容重心：重点提取核心事实、权威数据、百分比、发布时间、典型案例及关键结论；过滤掉广告营销套话、客套废话和冗余修饰。
4. 来源溯源：在每个关键事实或数据后附带来源或域名（例如："- 2025年AI教育市场渗透率达35% [36kr.com]"）。
5. 边界处理：若搜索无有效结果，直接返回 "未检索到有效信息"。
6. 语言要求：摘要内容统一采用流畅、准确的中文输出。
7. 纯粹输出：严禁输出任何开场白、寒暄或总结性套话，仅输出提炼后的事实列表。
"""

search_tool = TavilySearch(max_results=5, topic="general")

search_agent = create_agent(
    model=llm, tools=[search_tool], system_prompt=SEARCH_INSTRUCTIONS
)

# search_agent_res = search_agent.invoke(
#     {"messages": [{"role": "user", "content": planner_result.searches[0].query}]}
# )

# print(search_agent_res['messages'][-1].content)


# 三、构建编写智能体

WRITER_PROMPT = """你是一名资深的行业研究专家，擅长将零散的调研资料整合为结构严谨、见解深刻的深度研究报告。
你将收到用户的【原始研究主题】以及研究助手收集整理的【多维度搜索摘要】。
你的任务是完成一份高质量的研究报告，遵循以下规范：
1. 结构与内容规范：
   - 报告采用专业 Markdown 格式排版，包含标题、层级结构（#、##、###）、数据表格或关键要点列表。
   - 报告结构应涵盖：背景与现状、核心应用场景与案例、技术与商业价值、面临的挑战与瓶颈、未来发展趋势及建议。
   - 篇幅详实，字数建议在 2000 ~ 4000 字之间，严禁泛泛而谈或车轱辘话。
2. 事实与引用：
   - 充分吸收并提炼搜索材料中的事实、数据和观点，基于事实进行客观推理。
   - 保留材料中的关键数据源或案例出处。
3. 输出要求：
   - 全文使用流畅、专业的中文输出，避免机翻腔。
   - 按照指定的 Schema 格式分别提供简明摘要（short_summary）、完整长报告（markdown_report）和后续延伸研究问题（follow_up_questions）。
"""


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


# 四、自定义逻辑串联/LangGraph 图结构串联


def plan_searches(query: str) -> WebSearchPlan:
    print(f"\n[1/3] 正在规划搜索策略: '{query}' ...")
    result = planner_chain.invoke({"query": query})
    print(f"  -> 生成了 {len(result.searches)} 个搜索项:")
    for idx, item in enumerate(result.searches, 1):
        print(f"     {idx}. {item.query} (理由: {item.reason})")
    return result


def search(item: WebSearchItem) -> str | None:
    try:
        final_query = f"Search Item: {item.query}\nReason for searching: {item.reason}"
        result = search_agent.invoke(
            {"messages": [{"role": "user", "content": final_query}]}
        )
        return str(result["messages"][-1].content)
    except Exception as e:  # noqa
        print(f"  [!] 搜索 '{item.query}' 失败: {e}")
        return None


def perform_searches(search_plan: WebSearchPlan):
    print(f"\n[2/3] 正在执行网络搜索与信息提炼 (共 {len(search_plan.searches)} 项)...")
    results = []
    for idx, item in enumerate(search_plan.searches, 1):
        print(f"  [{idx}/{len(search_plan.searches)}] 搜索中: {item.query} ...")
        result = search(item)
        if result is not None:
            results.append(result)
    return results


def write_report(query: str, search_results) -> ReportData:
    print(f"\n[3/3] 正在根据 {len(search_results)} 份搜索摘要撰写完整研究报告...")
    summary = "\n\n".join(search_results)
    final_query = f"Original query: {query}\n Summarized search results: {summary}"
    result = writer_chain.invoke({"query": final_query})
    return result


def deepsearch(query: str) -> ReportData:
    search_plan = plan_searches(query)
    search_results = perform_searches(search_plan)
    report = write_report(query, search_results)

    print("\n" + "=" * 50)
    print("【最终研究报告】\n")
    print(report.markdown_report)
    return report


if __name__ == "__main__":
    deepsearch("AI Agent 应用开发岗位在中国招聘市场的前景")
