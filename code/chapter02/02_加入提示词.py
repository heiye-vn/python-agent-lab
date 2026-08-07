import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import BooleanOutputParser

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手，请根据用户的问题给出回答。"),
        ("user", "这是用户的问题：{topic}，请用 yes 或 no 来回答。"),
    ]
)

# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", "你是一名资深的 {role}。"), ("user", "{question}")]
# )

# 直接使用模型 + 输出解析器
# template_qa_chain = prompt_template | model | StrOutputParser()
template_qa_chain = (
    prompt_template | model | BooleanOutputParser()
)  # BooleanOutputParser 该解析器不推荐使用了

result = template_qa_chain.invoke({"topic": "请问 1 + 1 是否大于 2？"})

# result = template_qa_chain.invoke(
#     {"role": "Python 架构师", "question": "什么是装饰器？"}
# )

print(result)
