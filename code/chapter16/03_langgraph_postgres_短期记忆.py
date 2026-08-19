import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

db_uri = os.getenv("SUPABASE_DB_URL")
if not db_uri:
    raise ValueError("未检测到 SUPABASE_DB_URL，请检查 code/chapter16/.env 文件配置！")

# 1. 初始化大模型
llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

# 2. 使用 PostgresSaver 连接 Supabase PostgreSQL 数据库
print("正在连接 Supabase 云端 PostgreSQL 数据库...")
with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    # 首次运行时自动在 Supabase 中创建 checkpoints 等数据表
    checkpointer.setup()
    print("✅ 数据库表结构初始化/就绪完成！\n")

    # 3. 绑定 Agent
    agent = create_agent(model=llm, checkpointer=checkpointer)

    # 4. 会话 1：第一轮对话（自我介绍）
    config_1 = {"configurable": {"thread_id": "supabase_session_001"}}
    print(">>> 正在进行【会话1】第一轮提问：自我介绍...")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我是韩立，江湖人称韩跑跑！"}]},
        config=config_1,
    )
    print("Agent 回复：", response["messages"][-1].content)
    print("-" * 50)

    # 5. 会话 1：第二轮对话（测试记忆延续）
    print(">>> 正在进行【会话1】第二轮提问：测试记忆...")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "你还记得我是谁、外号是什么吗？"}]},
        config=config_1,
    )
    print("Agent 回复：", response["messages"][-1].content)
    print("=" * 50)

    # 6. 会话 2：不同 thread_id（测试会话隔离）
    config_2 = {"configurable": {"thread_id": "supabase_session_002"}}
    print(">>> 正在进行【会话2】提问（全新会话隔离测试）：")
    response_2 = agent.invoke(
        {"messages": [{"role": "user", "content": "你还记得我是谁、外号是什么吗？"}]},
        config=config_2,
    )
    print("Agent 回复：", response_2["messages"][-1].content)
