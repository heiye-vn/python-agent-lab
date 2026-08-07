# 示例代码为旧版写法，目前不推荐，了解其原理即可

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

""" 第一个子链：生成新闻内容 """
# 第一步：根据标题生成新闻正文
news_gen_prompt = PromptTemplate.from_template(
    "请根据以下新闻标题撰写一段简短的新闻内容（100字以内）：\n\n标题：{title}"
)

news_chain = news_gen_prompt | model | StrOutputParser()

# 从正文中提取结构化字段
schemas = [
    ResponseSchema(name="time", description="事件发生的时间"),
    ResponseSchema(name="location", description="事件发生的地点"),
    ResponseSchema(name="event", description="发生的具体事件"),
]
parser = StructuredOutputParser.from_response_schemas(schemas)

""" 第二个子链：生成新闻摘要 """
summary_prompt = PromptTemplate.from_template(
    "请从下面这段新闻内容中提取关键信息，并返回结构化JSON格式：\n\n{news}\n\n{format_instructions}"
)

summary_chain = (
    summary_prompt.partial(format_instructions=parser.get_format_instructions())
    | model
    | parser
)

# 组合成一个复合 chain
full_chain = {"news": news_chain} | summary_chain

result = full_chain.invoke({"title": "苹果公司在加州发布新款AI芯片"})
print(result)
