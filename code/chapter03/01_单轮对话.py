import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

chabot_prompt = ChatPromptTemplate.from_messages([
    ('system', '你叫苍井空，是日本著名女演员。'),
    ('user', '{input}')
])

basic_qa_chain =  chabot_prompt | model | StrOutputParser()

question = "你好，请介绍一下你自己。"
result = basic_qa_chain.invoke(question)
print(result)