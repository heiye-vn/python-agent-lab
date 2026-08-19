import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.store.sqlite import SqliteStore

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

# 1. 初始化大模型
llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

# 2. 本地 SQLite 长期记忆数据库文件路径
db_file = Path(__file__).parent / "sqlite_store.db"


# 3. 定义供 Agent 使用的长期记忆工具（通过 get_store() 访问）
def get_user_info(config: RunnableConfig) -> str:
    """查找用户的长期记忆画像信息"""
    store = get_store()
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return "未提供 user_id，无法检索长期记忆"

    item = store.get(("users",), user_id)
    return str(item.value) if item else "未找到该用户的长期记忆信息"


def update_user_hobby(hobby: str, config: RunnableConfig) -> str:
    """更新或添加用户的长期偏好爱好"""
    store = get_store()
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return "未提供 user_id，无法保存"

    # 先读取已有数据
    item = store.get(("users",), user_id)
    data = item.value if item else {}
    data["hobby"] = hobby
    # 写入更新
    store.put(("users",), user_id, data)
    return f"已成功将用户 {user_id} 的爱好更新为：{hobby}"


# 4. 使用 SqliteStore 上下文管理器运行
with SqliteStore.from_conn_string(str(db_file)) as store:
    # 首次运行自动初始化表结构
    store.setup()

    # 预先写入初始长期画像数据（命名空间: ("users",), key: "user_123"）
    store.put(
        ("users",),
        "user_123",
        {"name": "刘亦菲", "job": "演员", "hobby": "喜欢喝拿铁和养猫"},
    )
    print(f"✅ SQLite 长期记忆库已就绪，存储文件：{db_file.name}\n")

    # 5. 绑定 Agent 与 store
    agent = create_agent(
        model=llm, tools=[get_user_info, update_user_hobby], store=store
    )

    # 6. 【会话 1】（thread_id="thread_001", user_id="user_123"）
    config_thread_1 = {
        "configurable": {
            "thread_id": "thread_001",
            "user_id": "user_123",
        }
    }
    print(">>> 【会话 1】提问：查询长期记忆中的用户信息")
    response_1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你知道我的名字、职业和爱好吗？"}]},
        config=config_thread_1,
    )
    print("Agent 回复：", response_1["messages"][-1].content)
    print("=" * 60)

    # 7. 【会话 2】（全新的 thread_id="thread_002"，但 user_id 仍然是 "user_123"）
    # 验证核心能力：跨 Thread（跨会话窗口）共享长期记忆！
    config_thread_2 = {
        "configurable": {
            "thread_id": "thread_002",
            "user_id": "user_123",
        }
    }
    print("\n>>> 【会话 2】提问（全新会话窗口，测试跨 Thread 读取长期记忆）：")
    response_2 = agent.invoke(
        {"messages": [{"role": "user", "content": "推荐一部适合我职业或爱好的周末休闲活动吧"}]},
        config=config_thread_2,
    )
    print("Agent 回复：", response_2["messages"][-1].content)
