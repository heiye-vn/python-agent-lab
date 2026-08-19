import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

# 本地创建一个轻量级 sqlite 数据库文件
db_file = Path(__file__).parent / "sqlite_memory.db"

conn = sqlite3.connect(db_file, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# 绑定 Agent
agent = create_agent(model=llm, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session_1"}}

# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "你好，我叫王林，外号王麻子"}]},
#     config=config,
# )
# print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "你还记得我叫什么名字吗？"}]},
    config=config,
)
print("----------线程1----------")
print(response["messages"][-1].content)

new_config = {"configurable": {"thread_id": "session_2"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "现在你还记得我叫什么名字吗？"}]},
    config=new_config,
)
print("----------线程2----------")
print(response["messages"][-1].content)
