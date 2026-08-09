import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你叫马龙，是中国著名的乒乓球运动员。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model | StrOutputParser()

messages_list = []  # 初始化历史
print("🔹 输入 exit 结束对话")
while True:
    user_query = input("👤 你：")
    if user_query.lower() in {"exit", "quit"}:
        break

    # 1) 追加用户消息
    messages_list.append(HumanMessage(content=user_query))

    # 2) 调用模型
    assistant_reply = chain.invoke({"messages": messages_list})
    print("马龙：", assistant_reply)

    # 3) 追加 AI 回复
    messages_list.append(AIMessage(content=assistant_reply))

    # 4) 仅保留最近 50 条
    messages_list = messages_list[-50:]
