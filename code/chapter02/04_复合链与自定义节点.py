"""
04_复合链与自定义节点.py

演示 LangChain (LCEL) 中复合链与自定义节点的完整用法：

知识点 1: 自定义节点 —— 用 RunnableLambda 把任意普通函数包装成链的一环
知识点 2: 自定义节点 —— 继承 Runnable 写一个自己的组件类
知识点 3: 并行复合链 —— RunnableParallel 让同一输入分流到多条子链
知识点 4: 透传与追加 —— RunnablePassthrough.assign 保留原输入、附加新字段
知识点 5: 链中套链 —— 串行复合，一条链的输出直接作为下一条链的输入
知识点 6: 查看链结构 —— get_graph().print_ascii() 可视化整条流水线

场景：输入一篇技术文章，流水线自动完成
    清洗 -> 统计 -> (摘要 || 关键词 || 情感分析) -> 汇总报告
"""

import io
import os
import re
import sys
from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Windows 控制台默认 GBK，中文输出前统一包一层 UTF-8，防止 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

# 待分析的文章（主题与之前学的 RabbitMQ 呼应）
ARTICLE = """
在智能体（Agent）系统中，任务往往需要在多个组件之间流转。如果各个组件直接互相调用，
一旦某个环节处理变慢，整条流程都会被拖垮。消息队列正是解决这个问题的经典方案：
生产者把任务发布到队列，消费者按自己的节奏取走处理，双方互不感知、互不阻塞。
配合确认机制与死信队列，还能保证任务不丢失、失败可重试。可以说，消息队列让
Agent 系统从"紧耦合的同步调用"进化为"可伸缩的异步协作"，是构建生产级多智能体
应用绕不开的一块基石。
"""


# ============================================================
# 知识点 1: 自定义节点 —— RunnableLambda 包装普通函数
# 只要函数满足 "一个输入 -> 一个输出"，就能成为链上的一环
# ============================================================


def normalize_input(raw):
    """自定义节点：把外部输入统一成 dict 结构，方便下游节点使用"""
    if isinstance(raw, str):
        return {"text": raw, "source": "user-input"}
    return raw


def clean_text(data):
    """自定义节点：文本清洗，去掉首尾空白、压缩连续空白"""
    cleaned = re.sub(r"\s+", " ", data["text"]).strip()
    print(f"[clean_text] 清洗完成，文本长度 {len(cleaned)}")
    return {**data, "text": cleaned}


def parse_sentiment(raw: str) -> str:
    """自定义节点：情感结果后处理。模型偶尔会多说几个字，
    这里兜底提取关键词，保证下游拿到的是规范值"""
    for word in ("正面", "负面", "中性"):
        if word in raw:
            return word
    return "未知"


# ============================================================
# 知识点 2: 自定义节点 —— 继承 Runnable 写组件类
# 适合需要复用、带参数、或逻辑较复杂的节点
# ============================================================


class TextStatsNode(Runnable):
    """统计文本的基础指标，输出一个 dict 作为元数据"""

    def invoke(self, input, config=None, **kwargs):
        text = input["text"] if isinstance(input, dict) else str(input)
        return {
            "char_count": len(text),
            "sentence_count": text.count("。") + text.count("！") + text.count("？"),
        }


# ============================================================
# 三条基础子链：prompt | model | parser（链的基本单元）
# ============================================================

summary_chain = (
    ChatPromptTemplate.from_template(
        "用一句话总结下面这篇文章的核心观点，只输出总结：\n\n{text}"
    )
    | model
    | StrOutputParser()
)

keywords_chain = (
    ChatPromptTemplate.from_template(
        "提取下面这篇文章的 3 个关键词，用逗号分隔，只输出关键词：\n\n{text}"
    )
    | model
    | StrOutputParser()
)

# 情感链尾部挂一个自定义节点做兜底清洗，这是 "模型 + 自定义节点" 的常见组合
sentiment_chain = (
    ChatPromptTemplate.from_template(
        "判断下面这篇文章的整体情感倾向，只能从 正面、负面、中性 三个词中选一个输出：\n\n{text}"
    )
    | model
    | StrOutputParser()
    | RunnableLambda(parse_sentiment)
)


# ============================================================
# 知识点 3: 并行复合链 —— RunnableParallel
# 同一份输入同时进入三条子链，并发执行，结果汇总为一个 dict
# ============================================================

analysis_chain = RunnableParallel(
    summary=itemgetter("text") | summary_chain,
    keywords=itemgetter("text") | keywords_chain,
    sentiment=itemgetter("text") | sentiment_chain,
    # meta 分支只是透传，把统计信息带到最后
    meta=itemgetter("meta"),
)
# 补充：直接写 dict 字面量是等价简写，LCEL 会自动转成 RunnableParallel：
# analysis_chain = {"summary": ..., "keywords": ..., "sentiment": ...}


# ============================================================
# 知识点 4 的收尾节点：把并行结果合并成一份报告（纯 Python，不花钱调模型）
# ============================================================


def merge_report(data):
    """自定义节点：汇总所有分支结果，生成最终报告"""
    meta = data["meta"]
    return (
        "================ 文章分析报告 ================\n"
        f"【摘要】{data['summary']}\n"
        f"【关键词】{data['keywords']}\n"
        f"【情感倾向】{data['sentiment']}\n"
        f"【文本统计】共 {meta['char_count']} 字，约 {meta['sentence_count']} 句\n"
        "=============================================="
    )


# ============================================================
# 组装完整流水线：预处理 -> 并行分析 -> 汇总
# 这就是复合链：链里面套着链，自定义节点穿插其中
# ============================================================

full_pipeline = (
    RunnableLambda(normalize_input)  # 自定义节点：规范输入
    | RunnableLambda(clean_text)  # 自定义节点：清洗文本
    | RunnablePassthrough.assign(meta=TextStatsNode())  # 自定义组件类：附加统计元数据
    | analysis_chain  # 并行复合链：三路分流
    | RunnableLambda(merge_report)  # 自定义节点：生成报告
)


# ============================================================
# 知识点 5: 链中套链（串行复合）
# 摘要链的输出是字符串，直接喂给标题链 —— 链可以当作另一条链的节点
# ============================================================

title_prompt = ChatPromptTemplate.from_template(
    "根据下面这句摘要，起一个不超过 15 个字的标题，只输出标题：{summary}"
)
# summary_chain 本身是一条完整的链，这里被当作 title 链的第一个节点
title_chain = summary_chain | title_prompt | model | StrOutputParser()


if __name__ == "__main__":
    print("===== 第一步：运行完整流水线（复合链 + 自定义节点）=====")
    report = full_pipeline.invoke(ARTICLE)
    print(report)

    print("\n===== 第二步：链中套链（摘要链 -> 标题链）=====")
    title = title_chain.invoke({"text": ARTICLE})
    print(f"生成标题：{title}")

    print("\n===== 第三步：查看整条流水线的结构图 =====")
    try:
        # 画图依赖 grandalf 库：pip install grandalf
        full_pipeline.get_graph().print_ascii()
    except ImportError:
        print("(结构图需要 grandalf 库，请先执行 pip install grandalf)")
