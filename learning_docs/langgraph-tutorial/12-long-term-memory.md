# 第 12 章：长期记忆 Store

Checkpointer 的记忆以 thread 为界——换一个 thread_id 就"失忆"。**Store（长期记忆存储）跨 thread 存取数据**，让 Agent 记住用户偏好、历史事实、工作方法。

## 12.1 Checkpointer vs Store

| | Checkpointer | Store |
|---|---|---|
| 作用域 | 单个 thread 内 | **跨 thread、跨会话** |
| 数据形态 | 状态快照链（自动写） | 键值记录（**业务代码显式写**） |
| 检索 | 按 thread 取最新 | 按命名空间 / 语义向量取 |
| 典型内容 | 完整对话状态 | 用户画像、事实、偏好、经验 |

## 12.2 核心概念：namespace + 记忆条目

Store 的数据模型：**嵌套命名空间（tuple）下的键值记录**：

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

store = InMemoryStore()

# put(命名空间元组, 键, 值dict)
store.put(
    ("users", "alice", "preferences"),       # 命名空间：用户 → 类别
    "food",                                   # 键
    {"preference": "不吃辣，喜欢日料"},        # 值
)

# get
item = store.get(("users", "alice", "preferences"), "food")
print(item.value)          # {"preference": "不吃辣，喜欢日料"}
print(item.key, item.namespace, item.updated_at)

# search：列出某命名空间下的记录
for item in store.search(("users", "alice", "preferences")):
    print(item.key, item.value)

# delete
store.delete(("users", "alice", "preferences"), "food")
```

namespace 设计就是**记忆的 schema 设计**。常见分层：

```
("users", {user_id}, "preferences")   # 用户偏好（事实型）
("users", {user_id}, "episodes")      # 对这次交互的情景记忆
("org", {org_id}, "procedures")       # 组织级 SOP/工作方法（所有用户共享）
```

## 12.3 在图中使用 Store

compile 时传入，节点内通过 `get_store()` 或 config 访问：

```python
from langgraph.config import get_store

def recall(state: AgentState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    store = get_store()

    # 取该用户所有偏好，注入 system prompt
    memories = store.search(("users", user_id, "preferences"))
    memory_text = "\n".join(f"- {m.value['preference']}" for m in memories)
    return {"system_context": memory_text}


def remember(state, config):
    user_id = config["configurable"]["user_id"]
    get_store().put(
        ("users", user_id, "preferences"),
        "location",
        {"preference": "常驻上海"},
    )
    return {}
```

## 12.4 语义记忆检索（向量搜索）

给 Store 配置嵌入索引后，`search` 支持 `query` 语义检索：

```python
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
    }
)

store.put(("users", "alice", "episodes"), "e1",
          {"text": "用户去了东京旅行，对筑地市场印象很深"})

# 语义检索：不依赖关键词命中
hits = store.search(
    ("users", "alice", "episodes"),
    query="用户去过哪里旅游",        # 会向量化后找最相关的记忆
    limit=5,
)
```

生产环境用 `PostgresStore`（pgvector）：

```python
# pip install langgraph-checkpoint-postgres
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string("postgresql://...")
store.setup()   # 建表 + 向量索引

graph = builder.compile(checkpointer=checkpointer, store=store)
```

## 12.5 记忆的三种类型与写入策略

| 类型 | 内容 | namespace 示例 | 写入时机 |
|---|---|---|---|
| 语义记忆 semantic | 事实："用户叫 Alice，在上海工作" | `preferences` / `facts` | 对话中发现新事实 |
| 情景记忆 episodic | 经历："上周帮他订过东京酒店" | `episodes` | 每次会话结束后批处理 |
| 过程记忆 procedural | 方法："该用户喜欢先看摘要再看细节" | `procedures` | 交互模式稳定后 |

### 写入策略一：热路径写入（in the loop）

给 Agent 一个 `memory` 工具，模型边聊边存：

```python
from langchain_core.tools import tool

@tool
def save_memory(content: str, category: str) -> str:
    """当对话中出现值得长期记住的用户信息（偏好/事实/背景）时调用。
    category: preferences | facts | episodes"""
    user_id = get_config()["configurable"]["user_id"]
    key = str(uuid4())[:8]
    get_store().put(("users", user_id, category), key, {"text": content})
    return "saved"

agent = create_react_agent(llm, tools=[save_memory],
                           prompt="……你可以用 save_memory 记住用户信息")
```

### 写入策略二：后台批处理（background）

会话结束时（或定时任务）跑一个"记忆提炼"图：读 thread 历史 → LLM 提取值得记的 → 写入 store。优点：主链路零开销、可人工审核；缺点：非实时。生产常用 **后台为主 + 热路径为辅**。

## 12.6 使用记忆的完整闭环

```
用户消息 ──┐
           ▼
    recall 节点：store.search(用户相关记忆) → 注入 system prompt
           ▼
      Agent 思考/工具调用（期间可能调用 save_memory）
           ▼
    会话结束 → 后台提炼图 → store.put(情景记忆/更新画像)
           ▼
    下次任意新 thread：recall 时这些记忆仍然可用（跨会话生效）
```

## 12.7 生产注意事项

1. **记忆也要有治理**：过期（updated_at 判断）、错误记忆的删除入口、用户可查看自己的记忆（合规要求）
2. ** namespace 里永远放 user_id/org_id**，多租户隔离的基本功
3. 嵌入模型变更 = 全量重建索引，提前规划（记录 embed 模型版本）
4. 记忆注入 prompt 时做**预算控制**（最多 top-k 条、每条截断），别把记忆库整个塞进上下文

## 本章小结

- Store 跨 thread 持久化；namespace（tuple）+ key + value 数据模型
- 配嵌入索引即得语义检索（生产用 PostgresStore + pgvector）
- 三类记忆：semantic / episodic / procedural，对应不同 namespace 与写入时机
- 写入两策略：Agent 热路径工具存 + 会话后后台提炼
- 多租户隔离、记忆治理、注入预算是生产必修课

> 下一章：时间旅行——检查点链的高级玩法。
