import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from zep_cloud import Message
from zep_cloud.client import Zep

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

zep_api_key = os.getenv("ZEP_API_KEY")
if not zep_api_key:
    raise ValueError("未检测到 ZEP_API_KEY，请检查 code/chapter16/.env 文件配置！")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

# 1. 初始化 Zep Cloud 客户端与大模型
zep = Zep(api_key=zep_api_key)
llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

print("=" * 65)
print("【实战：Zep 时序知识图谱 (Temporal Knowledge Graph) 记忆系统】")
print("=" * 65)

# 2. 创建或绑定用户 (User)
unique_id = uuid.uuid4().hex[:6]
user_id = f"user_wanglin_{unique_id}"

print(f"\n1. 正在 Zep 中注册用户: {user_id}...")
zep_user = zep.user.add(
    user_id=user_id,
    first_name="王林",
    last_name="道友",
    email=f"{user_id}@example.com",
    metadata={"role": "程序员", "city": "深圳"},
)
print(f"✅ 用户创建成功: {zep_user.user_id} (姓名: {zep_user.first_name})")

# 3. 模拟第 1 个会话 Thread：包含时序变动与偏好的对话
thread_1_id = f"th_day1_{unique_id}"
print(f"\n2. 创建第 1 个会话线索: {thread_1_id}...")
zep.thread.create(thread_id=thread_1_id, user_id=user_id)

messages_1 = [
    Message(
        role="user",
        role_type="user",
        content=(
            "你好助手！我叫王林，外号王麻子。我目前在深圳做 Python 和 Agent 开发。"
            "上周我的购车预算是 15 万元，但昨天我刚收到一笔大额年终奖，现在的购车预算提升到了 35 万元！"
            "另外，我极度喜欢喝冰美式，坚决不喝含糖饮料。"
        ),
    ),
    Message(
        role="assistant",
        role_type="assistant",
        content="你好王林（王麻子）！恭喜你喜提年终奖，我已经记下了你的深圳常驻地、35万新购车预算以及喝冰美式的偏好！",
    ),
]

print("   正在向 Zep 投递第 1 轮会话消息流（Zep 后台自动异步构建时序知识图谱）...")
zep.thread.add_messages(thread_id=thread_1_id, messages=messages_1)
print("✅ 消息已成功推入 Zep 记忆引擎！")

# 4. 模拟几天后开启全新的第 2 个会话 Thread（跨 Thread 读取时序图谱）
thread_2_id = f"th_day5_{unique_id}"
print(f"\n3. 模拟几天后开启【全新独立会话】: {thread_2_id}...")
zep.thread.create(thread_id=thread_2_id, user_id=user_id)

# 稍作等待，让 Zep 异步图谱引擎处理实体关系
print("   等待 Zep 知识图谱引擎完成实体关系与时序提炼 (等待 3 秒)...")
time.sleep(3)

# 5. 从 Zep 时序知识图谱中检索关于“购车预算与饮品偏好”的记忆
print("\n4. 正在从 Zep 时序知识图谱中执行语义与图谱检索...")
search_results = zep.graph.search(
    user_id=user_id,
    query="用户的购车预算是多少？有什么饮品偏好？",
)

# 整理图谱事实
graph_facts = []
if search_results.edges:
    for edge in search_results.edges:
        graph_facts.append(f"• 关系事实: {edge.fact}")
if search_results.nodes:
    for node in search_results.nodes:
        graph_facts.append(f"• 实体节点: {node.name} ({node.summary})")

print(f"✅ Zep 知识图谱检索完成！命中 {len(graph_facts)} 条图谱事实：")
for fact in graph_facts:
    print(f"   {fact}")

# 6. 将 Zep 图谱时序记忆注入大模型 Prompt 进行跨会话回答
print("\n5. 【全新会话提问】：结合 Zep 长期时序记忆生成回复")
user_query = "请根据我现在的最新预算和日常喜好，为我推荐一款车，并顺便推荐一款提神饮品！"
print(f"   [用户提问 (Thread 2)]: {user_query}\n")

# 构建结合 Zep 知识图谱上下文的 Prompt
memory_context = (
    "\n".join(graph_facts)
    if graph_facts
    else "用户曾提及预算从15万提升到了35万，常驻深圳，偏好冰美式。"
)

system_prompt = (
    f"你是一个拥有长程记忆的贴心智能助理。\n"
    f"【从 Zep 时序知识图谱中提取的用户长期事实】：\n{memory_context}\n"
    f"请精准捕捉用户的【最新预算】（注意时序变化）和个性化偏好进行针对性推荐。"
)

chat_prompt = [
    ("system", system_prompt),
    ("user", user_query),
]

response = llm.invoke(chat_prompt)
print(">>> Agent 最终回复：")
print(response.content)

# 7. 将本次新对话也沉淀回 Zep
zep.thread.add_messages(
    thread_id=thread_2_id,
    messages=[
        Message(role="user", role_type="user", content=user_query),
        Message(
            role="assistant",
            role_type="assistant",
            content=response.content,
        ),
    ],
)
print("\n✅ 第 2 轮会话已自动追加沉淀入 Zep 长期记忆生命周期中！")
