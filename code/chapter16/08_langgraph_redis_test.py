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


def get_user_profile(user_id: str) -> dict:
    """跨会话读取用户长期记忆"""
    key = f"user:profile:{user_id}"
    return redis_client.hgetall(key)


def get_thread_messages(thread_id: str) -> list:
    """从 Redis 读取历史对话列表"""
    raw_list = redis_client.lrange(f"thread:message:{thread_id}", 0, -1)
    return [json.loads(item) for item in raw_list]


def save_thread_message(thread_id: str, role: str, content: str):
    key = f"thread:message:{thread_id}"
    redis_client.rpush(key, json.dumps({"role": role, "content": content}))
    redis_client.expire(key, 3600)


# =====================================================================
# 组装短期历史消息进行提问
# =====================================================================
user_id = "user_wanglin_99"
thread_1 = "session_2026_001"  # 沿用之前的会话 1

# A. 读取长期记忆
profile = get_user_profile(user_id)
system_msg = (
    "system",
    f"你是贴心助手。已知用户长期画像：姓名【{profile.get('name')}】，外号【{profile.get('alias')}】。",
)

# B. 读取短期记忆
history_data = get_thread_messages(thread_1)
history_messages = [
    (msg["role"], msg["content"]) for msg in history_data
]  # 将 Redis 存的字典转换为 LangChain 的消息元组格式

current_user_msg = ("user", "那你帮我看下刚刚点了什么东西喝？")

chat_prompt = [system_msg] + history_messages + [current_user_msg]
response = llm.invoke(chat_prompt)
print("\n>>> Agent 回复：")
print(response.content)


# 将当前轮次的问答也追加存入 Redis
save_thread_message(thread_1, "user", current_user_msg[1])
save_thread_message(thread_1, "assistant", response.content)
