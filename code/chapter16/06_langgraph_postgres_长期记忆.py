import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.store.postgres import PostgresStore

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


# 2. 定义供 Agent 使用的长期记忆工具（通过 get_store() 访问云端 Postgres）
def get_user_profile(config: RunnableConfig) -> str:
    """从云端 PostgreSQL 长期记忆库中查询用户画像和历史偏好"""
    store = get_store()
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return "未提供 user_id，无法查询"

    item = store.get(("user_profiles",), user_id)
    return str(item.value) if item else "未找到该用户的长期记忆档案"


def save_user_preference(key: str, value: str, config: RunnableConfig) -> str:
    """将用户新的偏好或事实信息持久化保存到云端长期记忆库"""
    store = get_store()
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return "未提供 user_id，无法保存"

    # 读取旧档案并合并新属性
    item = store.get(("user_profiles",), user_id)
    data = item.value if item else {}
    data[key] = value
    # 写入云端 PostgresStore
    store.put(("user_profiles",), user_id, data)
    return f"已成功将用户 {user_id} 的长期偏好【{key}={value}】保存至云端数据库！"


# 3. 使用 PostgresStore 上下文管理器连接 Supabase 云端数据库
print("正在连接 Supabase 云端 PostgreSQL 数据库 (PostgresStore)...")
with PostgresStore.from_conn_string(db_uri) as store:
    # 首次运行自动在 Supabase 中创建 store 数据表
    store.setup()
    print("✅ 云端长期记忆库表结构初始化/就绪完成！\n")

    # 预设一条长期记忆（命名空间: ("user_profiles",), key: "user_888"）
    store.put(
        ("user_profiles",),
        "user_888",
        {
            "name": "韩立",
            "alias": "韩天尊",
            "hobby": "培育灵草、喝灵茶",
            "motto": "杀人放火厉飞雨，万人敬仰韩天尊",
        },
    )

    # 4. 绑定 Agent 与云端 store
    agent = create_agent(
        model=llm,
        tools=[get_user_profile, save_user_preference],
        store=store,
    )

    # 5. 【会话 1】（thread_id="supabase_th_101", user_id="user_888"）
    config_1 = {
        "configurable": {
            "thread_id": "supabase_th_101",
            "user_id": "user_888",
        }
    }
    print(">>> 【会话 1】提问：查询长期档案")
    response_1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "查一下我的长期用户档案，告诉我我是谁、有什么座右铭？",
                }
            ]
        },
        config=config_1,
    )
    print("Agent 回复：", response_1["messages"][-1].content)
    print("=" * 60)

    # 6. 【会话 2】（全新的会话窗口 thread_id="supabase_th_102"，跨 Thread 共享验证）
    config_2 = {
        "configurable": {
            "thread_id": "supabase_th_102",
            "user_id": "user_888",
        }
    }
    print("\n>>> 【会话 2】提问（开启全新会话窗口，测试跨 Thread 读取云端长期档案）：")
    response_2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "在新的聊天窗口里，你还能查到我喜欢什么吗？",
                }
            ]
        },
        config=config_2,
    )
    print("Agent 回复：", response_2["messages"][-1].content)
