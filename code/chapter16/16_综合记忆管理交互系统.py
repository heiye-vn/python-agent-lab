"""
================================================================================
Chapter 16: 综合记忆管理交互系统
================================================================================
本系统整合了 Chapter 16 的全套核心记忆架构：
1. 【长期记忆 (Long-Term Memory)】: 基于 Supabase PostgreSQL (PostgresStore)，
   跨 Thread 全局持久化存储结构化用户画像与系统偏好，支持多租户命名空间隔离。
2. 【短期工作记忆 (Short-Term Memory)】: 基于 Upstash Redis 实现带 TTL 的 RedisCheckpointSaver，
   实现单个会话内上下文流水账管理，支持动态 TTL 过期自动销毁。
3. 【滑动窗口消息裁剪 (trim_messages)】: 推理前进行 Token / 消息条数安全拦截，
   保障 Prompt 不超过大模型上下文限制。
4. 【滚动摘要提取与历史物理修剪 (Rolling Summary + RemoveMessage)】:
   消息达到高水位线（>6条）自动触发压缩节点，更新累积摘要并使用 RemoveMessage 物理修剪 Redis 历史。
5. 【双通道记忆沉淀机制】:
   - 主动通道: Agent 自主识别意图并调用 save_user_memory 工具显式写入 Postgres。
   - 被动通道: summarize_node 在压缩历史时自动进行事实反思与画像提炼，隐式回写 Postgres。
6. 【交互式控制台】: 内置丰富调试指令 (/new, /ttl, /expire_test, /profile, /state, /history, /switch_user)。
================================================================================
"""

import json
import os
import random
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from typing_extensions import TypedDict

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

# 规避系统代理阻断
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import redis
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langgraph.config import get_store
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.postgres import PostgresStore

# =====================================================================
# 1. 环境变量与客户端校验
# =====================================================================
bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError("未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件！")

db_uri = os.getenv("SUPABASE_DB_URL")
if not db_uri:
    raise ValueError("未检测到 SUPABASE_DB_URL，请检查 code/chapter16/.env 文件！")

redis_url = os.getenv("UPSTASH_REDIS_URL")
if not redis_url:
    raise ValueError("未检测到 UPSTASH_REDIS_URL，请检查 code/chapter16/.env 文件！")

# 初始化统一的大模型客户端
llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

# 初始化 Redis 客户端
redis_client = redis.from_url(redis_url, decode_responses=False)


# =====================================================================
# 2. 短期记忆实现：支持 TTL 自动过期的 RedisCheckpointSaver
# =====================================================================
class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    基于 Redis 实现的 LangGraph 短期记忆检查点存储器。
    具备 TTL 自动老化过期机制，支持会话超时销毁与冷启动测试。
    """

    def __init__(self, client: redis.Redis, default_ttl: int = 3600):
        super().__init__()
        self.client = client
        self.default_ttl = default_ttl
        # 记录特定 thread 的自定义 ttl（如被 /ttl 指令动态修改）
        self.thread_ttls: dict[str, int] = {}

    def get_thread_ttl(self, thread_id: str) -> int:
        return self.thread_ttls.get(thread_id, self.default_ttl)

    def set_thread_ttl(self, thread_id: str, ttl: int):
        self.thread_ttls[thread_id] = ttl
        # 同时对已有键应用新的过期时间
        keys = self._get_all_keys_for_thread(thread_id)
        if keys:
            pipe = self.client.pipeline()
            for k in keys:
                pipe.expire(k, ttl)
            pipe.execute()

    def _get_all_keys_for_thread(self, thread_id: str) -> list[bytes]:
        pattern = f"lg:*:{thread_id}:*"
        return self.client.keys(pattern)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _format_key(self, *parts) -> str:
        return ":".join(str(p) for p in parts)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not checkpoint_id:
            latest_key = self._format_key(
                "lg", "threads", thread_id, checkpoint_ns, "latest"
            )
            checkpoint_id_bytes = self.client.get(latest_key)
            if not checkpoint_id_bytes:
                return None
            checkpoint_id = checkpoint_id_bytes.decode("utf-8")

        cp_key = self._format_key("lg", "cp", thread_id, checkpoint_ns, checkpoint_id)
        cp_data = self.client.hgetall(cp_key)
        if not cp_data:
            return None

        checkpoint = self.serde.loads_typed(
            (cp_data[b"checkpoint_type"].decode("utf-8"), cp_data[b"checkpoint"])
        )
        metadata = self.serde.loads_typed(
            (cp_data[b"metadata_type"].decode("utf-8"), cp_data[b"metadata"])
        )
        parent_id = cp_data.get(b"parent_id")
        parent_checkpoint_id = parent_id.decode("utf-8") if parent_id else None

        # 加载通道值
        channel_values = {}
        for channel, version in checkpoint.get("channel_versions", {}).items():
            blob_key = self._format_key(
                "lg", "blob", thread_id, checkpoint_ns, channel, version
            )
            blob_data = self.client.hgetall(blob_key)
            if blob_data:
                blob_type = blob_data[b"type"].decode("utf-8")
                blob_val = blob_data[b"data"]
                if blob_type != "empty":
                    channel_values[channel] = self.serde.loads_typed(
                        (blob_type, blob_val)
                    )

        checkpoint["channel_values"] = channel_values

        # 加载待写入状态
        writes = []
        writes_set_key = self._format_key(
            "lg", "writes_keys", thread_id, checkpoint_ns, checkpoint_id
        )
        w_keys = self.client.smembers(writes_set_key)
        for wk in w_keys:
            w_data = self.client.hgetall(wk)
            if w_data:
                task_id = w_data[b"task_id"].decode("utf-8")
                channel = w_data[b"channel"].decode("utf-8")
                w_type = w_data[b"type"].decode("utf-8")
                w_val = w_data[b"data"]
                writes.append(
                    (task_id, channel, self.serde.loads_typed((w_type, w_val)))
                )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=writes,
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        values: dict[str, Any] = c.pop("channel_values")
        ttl = self.get_thread_ttl(thread_id)

        pipe = self.client.pipeline()

        # 1. 存储 channel blobs
        for k, v in new_versions.items():
            blob_key = self._format_key("lg", "blob", thread_id, checkpoint_ns, k, v)
            if k in values:
                t, data = self.serde.dumps_typed(values[k])
                pipe.hset(blob_key, mapping={"type": t, "data": data})
            else:
                pipe.hset(blob_key, mapping={"type": "empty", "data": b""})
            pipe.expire(blob_key, ttl)

        # 2. 存储 checkpoint 元信息
        cp_key = self._format_key(
            "lg", "cp", thread_id, checkpoint_ns, checkpoint["id"]
        )
        t_cp, data_cp = self.serde.dumps_typed(c)
        meta_dict = get_checkpoint_metadata(config, metadata)
        t_meta, data_meta = self.serde.dumps_typed(meta_dict)
        parent_id = config["configurable"].get("checkpoint_id", "")

        pipe.hset(
            cp_key,
            mapping={
                "checkpoint_type": t_cp,
                "checkpoint": data_cp,
                "metadata_type": t_meta,
                "metadata": data_meta,
                "parent_id": parent_id or "",
            },
        )
        pipe.expire(cp_key, ttl)

        # 3. 更新最新 checkpoint 指针
        latest_key = self._format_key(
            "lg", "threads", thread_id, checkpoint_ns, "latest"
        )
        pipe.set(latest_key, checkpoint["id"], ex=ttl)

        pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        ttl = self.get_thread_ttl(thread_id)

        writes_set_key = self._format_key(
            "lg", "writes_keys", thread_id, checkpoint_ns, checkpoint_id
        )
        pipe = self.client.pipeline()

        for idx, (c, v) in enumerate(writes):
            idx_mapped = WRITES_IDX_MAP.get(c, idx)
            w_key = self._format_key(
                "lg", "w", thread_id, checkpoint_ns, checkpoint_id, task_id, idx_mapped
            )
            t, data = self.serde.dumps_typed(v)
            pipe.hset(
                w_key,
                mapping={
                    "task_id": task_id,
                    "channel": c,
                    "type": t,
                    "data": data,
                    "task_path": task_path,
                },
            )
            pipe.expire(w_key, ttl)
            pipe.sadd(writes_set_key, w_key)

        pipe.expire(writes_set_key, ttl)
        pipe.execute()

    def delete_thread(self, thread_id: str) -> None:
        keys = self._get_all_keys_for_thread(thread_id)
        if keys:
            self.client.delete(*keys)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if (
            config
            and "configurable" in config
            and "thread_id" in config["configurable"]
        ):
            t = self.get_tuple(config)
            if t:
                yield t


import threading

# 线程锁，保障并发工具调用时的 PostgreSQL 读-改-写原子性
store_lock = threading.Lock()


# =====================================================================
# 3. 长期记忆工具定义 (供 Agent 在对话中主动调用)
# =====================================================================
@tool
def save_user_memory(key: str, value: str, config: RunnableConfig) -> str:
    """
    当用户在对话中明确表述个人偏好、身份背景、习惯、忌口、常用技术栈或长期规则时调用此工具，持久化沉淀至云端 PostgreSQL。
    :param key: 偏好或属性类别，例如 'name', 'job', 'dietary_restrictions', 'tech_stack', 'city', 'hobby'
    :param value: 属性的具体取值，例如 '喜欢吃清淡粤菜，不吃香菜' 或 '全栈工程师，精通 Python'
    """
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    with store_lock:
        # 从 Postgres 获取现有画像并原子合并
        namespace = ("user_profiles",)
        item = store.get(namespace, user_id)
        profile_data = item.value if item else {}
        profile_data[key] = value
        profile_data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 写回 PostgreSQL
        store.put(namespace, user_id, profile_data)

    print(
        f"\n   💾 [主动长期记忆沉淀 -> PostgreSQL] 用户: {user_id} | 键: {key} -> 值: {value}"
    )
    return f"已成功将用户偏好【{key}: {value}】持久化保存至云端 PostgreSQL 数据库！"


tools = [save_user_memory]
model_with_tools = llm.bind_tools(tools)


# =====================================================================
# 4. 图状态 (State) 定义与核心节点实现
# =====================================================================
class MemoryAgentState(TypedDict):
    # 短期对话消息流水账（通过 add_messages 实现追加与 RemoveMessage 删除）
    messages: Annotated[list[BaseMessage], add_messages]
    # 累积历史滚动摘要
    summary: str
    # 召回的长期画像文本
    recalled_profile: str


# 节点 1: 长期记忆召回节点 (从 PostgreSQL 读取)
def recall_memory_node(state: MemoryAgentState, config: RunnableConfig):
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    item = store.get(("user_profiles",), user_id)
    if item and item.value:
        profile_dict = item.value
        # 格式化为便于 Prompt 理解的清单
        facts = [f"• {k}: {v}" for k, v in profile_dict.items() if k != "last_updated"]
        profile_text = "\n".join(facts)
    else:
        profile_text = "（暂无此用户的已知画像记录）"

    return {"recalled_profile": profile_text}


# 节点 2: 核心对话推理节点 (结合长期画像 + 滚动摘要 + trim_messages 安全拦截)
def chatbot_node(state: MemoryAgentState):
    recalled_profile = state.get("recalled_profile", "（暂无）")
    summary = state.get("summary", "")
    messages = state["messages"]

    system_prompt = (
        "你是一个具备自适应多层记忆架构的专业 AI 架构师助理。\n\n"
        "【长期用户档案 (来自 PostgreSQL 数据库)】:\n"
        f"{recalled_profile}\n\n"
        "【前情对话滚动摘要 (来自历史压缩)】:\n"
        f"{summary if summary else '（暂无历史摘要）'}\n\n"
        "【交互准则】:\n"
        "1. 回复时必须无缝结合长期档案偏好与短期会话上下文。\n"
        "2. 若用户透露出新的长期偏好、习惯或个人信息，请务必主动调用 save_user_memory 工具将其记录到云端数据库。\n"
        "3. 保持自然、专业、贴心的回答语气。"
    )

    # 安全防护：执行消息裁剪，保障上下文不超过窗口（保留 SystemMessage 与最新消息）
    trimmed_history = trim_messages(
        messages,
        max_tokens=10,  # 限制保留最近 10 条消息
        token_counter=len,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )

    prompt = [SystemMessage(content=system_prompt)] + trimmed_history
    response = model_with_tools.invoke(prompt)
    return {"messages": [response]}


# 节点 3: 滚动摘要与被动记忆沉淀节点
def summarize_node(state: MemoryAgentState, config: RunnableConfig):
    store = get_store()
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    existing_summary = state.get("summary", "")
    messages = state["messages"]

    # 保留最近的 2 条消息，其余全部压缩
    messages_to_compress = messages[:-2]

    # 1. 增量摘要生成
    history_text = "\n".join(
        f"[{m.__class__.__name__}]: {m.content}" for m in messages_to_compress
    )
    summary_prompt = (
        f"请根据已有摘要和新发生的一段对话，生成一段精炼、连贯的合并摘要。\n"
        f"【已有摘要】: {existing_summary if existing_summary else '无'}\n"
        f"【新增对话】:\n{history_text}\n\n"
        f"请直接输出合并后的最新核心摘要文本："
    )
    new_summary = llm.invoke(summary_prompt).content

    print(
        f"\n   📝 [滚动摘要已触发] 压缩了 {len(messages_to_compress)} 条消息 | 新摘要: {new_summary[:60]}..."
    )

    # 2. 被动画像反思与沉淀 (Passive Reflection)
    reflection_prompt = (
        f"请分析以下被压缩的对话内容，提取出用户明确透露的个人偏好、技术栈、背景或习惯（若无新增则返回空JSON大括号 {{}}）。\n"
        f"对话内容:\n{history_text}\n\n"
        f'请严格只返回合法 JSON 字典，例如: {{"hobby": "羽毛球", "preferred_lang": "Python"}}'
    )
    try:
        extract_res = llm.invoke(reflection_prompt).content
        cleaned = extract_res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json")
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```")
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```")
        cleaned = cleaned.strip()

        new_facts = json.loads(cleaned)
        if isinstance(new_facts, dict) and new_facts:
            with store_lock:
                namespace = ("user_profiles",)
                item = store.get(namespace, user_id)
                current_profile = item.value if item else {}
                for k, v in new_facts.items():
                    current_profile[k] = v
                current_profile["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                store.put(namespace, user_id, current_profile)
            print(f"   🔍 [被动长期画像提炼 -> PostgreSQL] 自动捕获事实: {new_facts}")
    except Exception as e:  # noqa: BLE001
        print(f"   ❌ [被动长期画像提炼 -> PostgreSQL] 自动捕获事实失败: {e}")

    # 3. 构造 RemoveMessage 物理清除已压缩的旧消息
    delete_ops = [RemoveMessage(id=m.id) for m in messages_to_compress if m.id]

    return {"summary": new_summary, "messages": delete_ops}


# 条件边逻辑
def should_continue(state: MemoryAgentState):
    last_message = state["messages"][-1]
    # 如果有工具调用，优先转到工具节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 消息条数超过阈值（如 > 6 条），流转到摘要修剪节点
    if len(state["messages"]) > 6:
        return "summarize_node"
    return END


# =====================================================================
# 5. 构建 LangGraph 状态图
# =====================================================================
workflow = StateGraph(MemoryAgentState)

# 添加节点
workflow.add_node("recall_memory", recall_memory_node)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summarize_node", summarize_node)

# 组织连线拓扑
workflow.add_edge(START, "recall_memory")
workflow.add_edge("recall_memory", "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        "summarize_node": "summarize_node",
        END: END,
    },
)
workflow.add_edge("tools", "chatbot")
workflow.add_edge("summarize_node", END)


# =====================================================================
# 6. 交互式控制台实现
# =====================================================================
def print_banner():
    banner = """
=================================================================================
🤖 LangGraph 多层记忆综合系统 (Postgres 长期记忆 + Redis 短期 TTL + 滚动摘要)
=================================================================================
【内置调试指令】:
  /help                  - 显示帮助指南
  /new                   - 开启新会话 (换 thread_id)，验证跨会话长期画像召回
  /ttl <秒数>            - 设置当前会话在 Redis 中的 TTL 过期时间
  /expire_test [秒数]    - 触发会话过期测试 (等待过期后验证短期清空、长期保留)
  /profile               - 查看当前用户在 PostgreSQL 云端库中的完整长期画像
  /state                 - 查看 LangGraph 当前 State 内部状态 (消息数/摘要/召回)
  /history               - 查看 Redis 短期记忆中当前活跃的消息流水账
  /switch_user <user_id> - 切换用户 ID (验证多租户数据隔离)
  /clear                 - 清屏
  /exit                  - 退出系统
=================================================================================
"""
    print(banner)


def run_interactive_system():
    print("正在连接 Supabase 云端 PostgreSQL (长期记忆 Store)...")
    with PostgresStore.from_conn_string(db_uri) as store:
        store.setup()
        print("✅ PostgreSQL 长期记忆库连接并就绪！")

        print("正在连接 Upstash 云端 Redis (短期记忆 Checkpointer)...")
        checkpointer = RedisCheckpointSaver(redis_client, default_ttl=3600)
        print("✅ Redis 短期记忆检查点存储器连接并就绪！\n")

        # 编译 LangGraph
        app = workflow.compile(checkpointer=checkpointer, store=store)

        # 启动时允许自定义用户账号，默认沿用示例账号
        print(
            "👤 请输入当前用户账号 ID (直接回车默认使用: user_wanglin_99): ",
            end="",
            flush=True,
        )
        try:
            custom_user = input().strip()
            current_user = custom_user if custom_user else "user_wanglin_99"
        except (KeyboardInterrupt, EOFError):
            current_user = "user_wanglin_99"

        current_thread = f"th_{uuid.uuid4().hex[:8]}"

        print_banner()

        # 启动时检查档案
        init_item = store.get(("user_profiles",), current_user)
        if init_item and init_item.value:
            print(
                f"👋 欢迎回来！检测到账号 \033[92m{current_user}\033[0m 已存在长期档案画像: {list(init_item.value.keys())}"
            )
        else:
            print(
                f"🌱 欢迎新用户！账号 \033[92m{current_user}\033[0m 目前是空白档案，聊天中透露的偏好会自动沉淀到云端 PostgreSQL。"
            )

        while True:
            ttl_now = checkpointer.get_thread_ttl(current_thread)
            # 实时检查该用户的档案状态
            user_prof = store.get(("user_profiles",), current_user)
            prof_status = (
                f"\033[92m已沉淀{len(user_prof.value)}项\033[0m"
                if (user_prof and user_prof.value)
                else "\033[90m空白档案\033[0m"
            )

            prompt_header = f"\n💡 [账号ID: \033[92m{current_user}\033[0m ({prof_status}) | 会话: \033[94m{current_thread}\033[0m | Redis-TTL: \033[93m{ttl_now}s\033[0m]"
            print(prompt_header)

            try:
                user_input = input("👉 请输入消息或指令: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 感谢使用，系统已安全退出。")
                break

            if not user_input:
                continue

            # ---------------- 指令解析 ----------------
            if user_input in ("/exit", "/quit"):
                print("\n👋 感谢使用，系统已安全退出。")
                break

            elif user_input == "/help":
                print_banner()
                continue

            elif user_input == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                continue

            elif user_input == "/new":
                current_thread = f"th_{uuid.uuid4().hex[:8]}"
                print(
                    f"\n✨ [已开启新会话窗口] 新 Thread ID: \033[94m{current_thread}\033[0m"
                )
                print(
                    "👉 短期消息流水账已归零，但云端 PostgreSQL 中的长期画像依然会被自动召回！"
                )
                continue

            elif user_input.startswith("/ttl"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    new_ttl = int(parts[1])
                    checkpointer.set_thread_ttl(current_thread, new_ttl)
                    print(
                        f"\n⏳ 当前会话 {current_thread} 的 Redis TTL 已动态调整为: {new_ttl} 秒！"
                    )
                else:
                    print("\n⚠️ 参数错误，使用格式: /ttl <秒数>，例如: /ttl 10")
                continue

            elif user_input.startswith("/expire_test"):
                parts = user_input.split()
                wait_sec = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 4
                print("\n🧪 [开始执行 Redis 短期记忆过期测试]")
                print(
                    f"   1. 将当前 Thread [{current_thread}] 的 TTL 调整为 {wait_sec} 秒..."
                )
                checkpointer.set_thread_ttl(current_thread, wait_sec)
                print(
                    f"   2. 正在休眠等待 Redis 键自然过期（等待 {wait_sec + 1} 秒）..."
                )
                time.sleep(wait_sec + 1)

                # 检查 Redis 键是否还存在
                tuple_check = checkpointer.get_tuple(
                    {"configurable": {"thread_id": current_thread}}
                )
                if tuple_check is None:
                    print("   ✅ [验证成功] Redis 中的短期记忆已被 TTL 机制彻底清空！")
                else:
                    print("   ⚠️ 键尚未完全过期，请重试。")
                print(
                    "   3. 现在你可以发送任何消息，测试系统如何从空白短期记忆中重新起步并完美继承长期画像！"
                )
                continue

            elif user_input == "/profile":
                item = store.get(("user_profiles",), current_user)
                print(f"\n📂 【PostgreSQL 长期画像档案 - {current_user}】")
                if item and item.value:
                    print(json.dumps(item.value, ensure_ascii=False, indent=2))
                else:
                    print("   （当前用户尚未在云端库中沉淀长期画像）")
                continue

            elif user_input == "/state":
                tuple_state = checkpointer.get_tuple(
                    {"configurable": {"thread_id": current_thread}}
                )
                print(f"\n🔍 【LangGraph 当前 State 快照 - {current_thread}】")
                if tuple_state and tuple_state.checkpoint:
                    cv = tuple_state.checkpoint.get("channel_values", {})
                    msgs = cv.get("messages", [])
                    summary_val = cv.get("summary", "")
                    recalled_val = cv.get("recalled_profile", "")
                    print(f"  • 当前活跃消息数: {len(msgs)} 条")
                    print(
                        f"  • 累积滚动摘要: {summary_val if summary_val else '（无）'}"
                    )
                    print(f"  • 注入的画像:\n{recalled_val}")
                else:
                    print("  • 当前会话尚无活跃 Checkpoint 快照（空会话）。")
                continue

            elif user_input == "/history":
                tuple_state = checkpointer.get_tuple(
                    {"configurable": {"thread_id": current_thread}}
                )
                print(f"\n📜 【Redis 短期记忆消息流水 - {current_thread}】")
                if tuple_state and tuple_state.checkpoint:
                    msgs = tuple_state.checkpoint.get("channel_values", {}).get(
                        "messages", []
                    )
                    if msgs:
                        for i, m in enumerate(msgs, 1):
                            print(f"   {i}. [{m.__class__.__name__}]: {m.content}")
                    else:
                        print("   （消息列表为空）")
                else:
                    print("   （当前会话暂无消息历史）")
                continue

            elif user_input.startswith("/switch_user"):
                parts = user_input.split()
                if len(parts) > 1:
                    current_user = parts[1]
                    current_thread = f"th_{uuid.uuid4().hex[:8]}"
                    print(
                        f"\n👤 [已切换用户] 当前用户: \033[92m{current_user}\033[0m | 自动分配新会话: \033[94m{current_thread}\033[0m"
                    )
                else:
                    print(
                        "\n⚠️ 参数错误，使用格式: /switch_user <user_id>，例如: /switch_user user_lisi_66"
                    )
                continue

            # ---------------- 业务对话处理 ----------------
            config = {
                "configurable": {
                    "thread_id": current_thread,
                    "user_id": current_user,
                }
            }

            print("\n🤖 AI 思考中...", end="", flush=True)
            inputs = {"messages": [HumanMessage(content=user_input)]}

            try:
                # 执行图工作流
                result = app.invoke(inputs, config=config)
                print("\r" + " " * 20 + "\r", end="")  # 清除思考提示

                last_ai_msg = [
                    m for m in result["messages"] if isinstance(m, AIMessage)
                ][-1]
                print(f"\n🤖 Agent 回复:\n{last_ai_msg.content}\n")

            except Exception as e:  # noqa
                print(f"\n❌ 模型调用或参数解析失败: {e}")


if __name__ == "__main__":
    run_interactive_system()
