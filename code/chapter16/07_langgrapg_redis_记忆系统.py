import json
import os
import sys
from pathlib import Path

import redis
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 避免终端中文乱码
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parent / ".env")


redis_url = os.getenv("UPSTASH_REDIS_URL")
if not redis_url:
    raise ValueError("未在 .env 中检测到 UPSTASH_REDIS_URL，请先配置！")

# 连接 Upstash 云端 Redis
redis_client = redis.from_url(redis_url, decode_responses=True)
print("✅ 成功连接至 Upstash 云端 Redis！")

llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv("ALI_BAILIAN_BASE_URL"),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)


# =====================================================================
# 一、 Redis 长期记忆管理（跨 Thread 共享的用户画像）
# =====================================================================
def save_user_profile(user_id: str, profile_data: dict):
    """把提取出的用户偏好存入 Redis Hash（长期记忆）"""
    key = f"user:profile:{user_id}"
    redis_client.hset(key, mapping=profile_data)
    print(f"📌 [长期记忆] 已将用户 {user_id} 的画像写入 Redis：{profile_data}")


def get_user_profile(user_id: str) -> dict:
    """跨会话读取用户长期记忆"""
    key = f"user:profile:{user_id}"
    return redis_client.hgetall(key)


# =====================================================================
# 二、 Redis 短期记忆管理（会话消息流水账，支持 TTL 自动过期）
# =====================================================================
def save_thread_message(thread_id: str, role: str, content: str, ttl_seconds=3600):
    """将单条对话消息推入 Redis List，并设置 1 小时 TTL 自动过期"""
    key = f"thread:message:{thread_id}"
    redis_client.rpush(key, json.dumps({"role": role, "content": content}))
    redis_client.expire(key, ttl_seconds)  # 设置过期时间，防止内存无限膨胀


def get_thread_messages(thread_id: str) -> list:
    """获取指定 Thread 的短期历史消息"""
    key = f"thread:message:{thread_id}"
    raw_list = redis_client.lrange(key, 0, -1)
    return [json.loads(item) for item in raw_list]


# =====================================================================
# 三、 协同工作演示：短期记忆 + 长期记忆
# =====================================================================
user_id = "user_wanglin_99"

# 1. 模拟写入长期记忆（跨越所有会话）
save_user_profile(
    user_id,
    {
        "name": "王林",
        "alias": "王麻子",
        "hobby": "喜欢喝拿铁",
        "vip_level": "VIP 8",
    },
)

# 2. 会话 1：用户在一个新窗口聊天
thread_1 = "session_2026_001"
profile = get_user_profile(user_id)  # 从 Redis 读取长期记忆

system_prompt = f"你是贴心助手。已知用户长期画像：姓名【{profile.get('name')}】，外号【{profile.get('alias')}】，偏好【{profile.get('hobby')}】。"

user_msg = "帮我点一杯下午茶，照老规矩来！"
save_thread_message(thread_1, "user", user_msg)

# 组合：System Prompt (长期记忆) + Thread 历史 (短期记忆)
chat_prompt = [
    ("system", system_prompt),
    ("user", user_msg),
]
response = llm.invoke(chat_prompt)
print("\n>>> Agent 回复：")
print(response.content)
# 保存回复至短期记忆
save_thread_message(thread_1, "assistant", response.content)


print("\n" + "=" * 50)
print(">>> 验证会话 2 的记忆延续性（Thread ID 不同）")

thread_2 = "session_2026_002"  # 新开一个会话
profile_2 = get_user_profile(user_id)  # 依然能读到长期画像

system_prompt_2 = (
    "你是贴心助手。已知用户长期画像：姓名【"
    f"{profile_2.get('name')}】外号【{profile_2.get('alias')}】"
    f"，偏好【{profile_2.get('hobby')}】。"
)
user_msg_2 = "还记得我之前最喜欢喝拿铁吧？"

save_thread_message(thread_2, "user", user_msg_2)

chat_prompt_2 = [
    ("system", system_prompt_2),
    ("user", user_msg_2),
]
response_2 = llm.invoke(chat_prompt_2)
print("\n>>> Agent 回复（跨 Thread）:")
print(response_2.content)
save_thread_message(thread_2, "assistant", response_2.content)
