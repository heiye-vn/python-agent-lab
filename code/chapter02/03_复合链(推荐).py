# 推荐写法：使用 Pydantic + with_structured_output 方式

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)


# 1. 用 Pydantic 定义需要的结构化输出数据类型
class NewsSummary(BaseModel):
    time: str = Field(description="事件发生的时间")
    location: str = Field(description="事件发生的地点")
    event: str = Field(description="发生的具体事件")


# 子链 1：根据标题生成新闻内容（纯文本链）
news_gen_prompt = PromptTemplate.from_template(
    "请根据以下新闻标题撰写一段简短的新闻内容（100字以内）：\n\n标题：{title}"
)
news_chain = news_gen_prompt | model | StrOutputParser()

# 子链 2：使用现代 with_structured_output 原生绑定抽取模型
structured_extractor = model.with_structured_output(NewsSummary)
extract_prompt = PromptTemplate.from_template(
    "请从下面这段新闻内容中提取关键信息：\n\n{news}"
)

summary_chain = extract_prompt | structured_extractor

# 使用 RunnableParallel / 字典组合，同时保留中间的新闻文本和提取结果
full_chain = {
    "raw_news": news_chain,  # 保存第一步生成的新闻正文
} | RunnablePassthrough.assign(
    summary=lambda x: summary_chain.invoke({"news": x["raw_news"]})  # 第二步提取
)

# 执行复合链
result = full_chain.invoke({"title": "苹果公司在加州发布新款AI芯片"})

print("=== 1. 第一步生成的新闻正文 ===")
print(result["raw_news"])
print("\n=== 2. 第二步提取的结构化结果 ===")
print(result["summary"])
