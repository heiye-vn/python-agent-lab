# 第 30 章：项目二 —— RAG 知识库问答 Agent

**技能点**：检索工具化（第 16 章）+ 自适应多轮检索（第 7 章循环）+ 引用溯源 + 并行检索（第 7 章 fan-out）+ 结构化输出（第 19 章）+ 缓存（第 4 章）。

与"传统 RAG 管线"的区别：**把检索做成 Agent 的工具**，模型自己决定"查什么、查几轮、什么时候够了"。

## 30.1 需求与架构

```
用户提问 → RAG Agent（create_react_agent）
             ├─ retrieve 工具：向量检索 + 关键词检索（并行）
             ├─ 多轮自适应：不够就换个 query 再查
             ├─ 回答必须带 [引用编号]
             └─ response_format：结构化答案（answer + sources + confidence）
```

```
rag_agent/
├── ingest.py     # 文档入库（切块+向量化）
├── tools.py      # 检索工具
├── agent.py      # Agent 组装
└── main.py
```

## 30.2 完整实现

### ingest.py —— 知识库入库

```python
# pip install chromadb langchain-chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DOCS = [
    {"id": "hr-1", "text": "年假政策：入职满1年5天，满3年10天，可跨年结转一次……"},
    {"id": "hr-2", "text": "报销制度：差旅报销需在行程结束后7日内提交，单笔超5000元需总监审批……"},
    {"id": "it-1", "text": "VPN 使用：请通过 ssoprovider 登录，禁止共享账号，异常登录会触发锁定……"},
]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

vectorstore = Chroma(
    collection_name="company_kb",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./kb_store",
)

for doc in DOCS:
    for i, chunk in enumerate(splitter.split_text(doc["text"])):
        vectorstore.add_texts(
            [chunk],
            ids=[f"{doc['id']}-c{i}"],
            metadatas=[{"source": doc["id"], "chunk": i}],
        )

print("入库完成")
```

### tools.py —— 检索工具（引用编号设计）

```python
import json
from typing import Annotated
from langchain_core.tools import tool

# 全局引用登记表：本 run 内编号 → 出处（回答前重置）
_citations: dict[int, dict] = {}
_next_id = 1


def reset_citations():
    global _next_id
    _citations.clear()
    _next_id = 1


@tool
def search_kb(query: str, top_k: int = 3) -> str:
    """在公司知识库中检索。返回带编号[1][2]的资料片段，
    回答时必须用这些编号标注引用来源。若无相关内容会明确说明。"""
    global _next_id
    hits = vectorstore.similarity_search_with_relevant_scores(query, k=top_k)
    if not hits or hits[0][1] < 0.3:          # 相关性阈值
        return json.dumps({"result": "未检索到相关内容"}, ensure_ascii=False)

    out = []
    for doc, score in hits:
        cid = _next_id
        _next_id += 1
        _citations[cid] = {"source": doc.metadata["source"], "score": round(score, 2)}
        out.append(f"[{cid}] {doc.page_content}")
    return "\n\n".join(out)


def get_citations() -> dict:
    """供 response 后处理使用：拿到 编号→真实出处 的映射"""
    return dict(_citations)
```

### agent.py —— 组装

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import CachePolicy

from tools import search_kb, reset_citations

llm = init_chat_model("openai:gpt-4o-mini")


class Answer(BaseModel):
    answer: str = Field(description="给用户的回答，引用处标注[编号]")
    sources: list[str] = Field(description="使用了的引用编号，如 ['1','2']")
    confidence: float = Field(ge=0, le=1, description="答案把握度；低于0.6应提示人工")
    need_human: bool = Field(description="是否建议转人工")


PROMPT = """你是公司知识库助手。严格规则：
1. 回答前先调用 search_kb 检索；检索不到就说不知道，禁止编造
2. 第一次检索不理想时，改写关键词再检索（最多 2 轮重试）
3. 回答中事实必须带 [编号] 引用
4. 没有引用支撑的内容不得出现在回答中
5. 最终用结构化格式输出（answer/sources/confidence/need_human）"""

rag_agent = create_react_agent(
    llm,
    tools=[search_kb],
    prompt=PROMPT,
    response_format=Answer,
    checkpointer=InMemorySaver(),
)
```

### main.py —— 运行与引用校验

```python
from agent import rag_agent
from tools import get_citations, reset_citations

config = {"configurable": {"thread_id": "rag-1"}}
reset_citations()

result = rag_agent.invoke(
    {"messages": [("user", "出差回来多久内要报销？超过5000怎么审批？")]},
    config,
)

ans: Answer = result["structured_response"]
print(ans.answer)                    # 含 [1][2] 引用
print("来源映射:", get_citations())   # {1: {"source":"hr-2",...}}
print("置信度:", ans.confidence)

# ── 质量门禁：引用完整性校验（可进 CI，第 23 章）──
import re
used = set(re.findall(r"\[(\d+)\]", ans.answer))
assert used <= {str(k) for k in get_citations()}, "回答引用了不存在的编号！"
```

## 30.3 进阶改造

### 多路并行检索（第 7 章静态 fan-out）

向量+关键词两路并发，`operator.add` 汇聚后统一编号：

```python
class SearchState(TypedDict):
    query: str
    hits: Annotated[list, operator.add]

# vector_search 与 keyword_search 两节点同轮并行 → rerank 节点重排取 top-k
```

（生产中这层通常做成**一个工具内部**的并发，Agent 无感知——除非你需要"多 query 并发改写"（第 31 章用 Send 实现）。）

### 缓存高频问答（第 4 章 CachePolicy）

```python
# 把"标准问题→标准答案"做成独立图节点开缓存：
builder.add_node("faq_lookup", faq_fn, cache_policy=CachePolicy(ttl=86400))
# 命中 FAQ 直接返回，不进 Agent 循环——高频问题成本趋近于零
```

### 对话式追问（多轮 RAG）

checkpointer 已解决：追问"那超过 1 万呢？"时，Agent 在同 thread 里看到上一轮的引用上下文，可判断是否需要新检索。

### 检索质量评估（第 27 章）

数据集字段：`question → 期望命中的 source`。评估器：`used_source ⊆ 期望`。每次换 embedding 模型/切块参数跑回归。

## 30.4 与传统 RAG 管线对比

| | 固定管线（retrieve→generate） | Agent 化 RAG（本项目） |
|---|---|---|
| 检索次数 | 固定 1 次 | 模型自适应多轮 |
| query 质量 | 原样透传 | 模型可改写关键词 |
| 无答案处理 | 硬生成 | 明确"不知道"+ need_human |
| 引用 | 靠后处理拼接 | 模型在生成时标注 |
| 成本/延迟 | 低而稳定 | 稍高（多一轮思考） |

**建议**：高并发低延迟入口（搜索框补全）用固定管线 + 缓存；对话式知识助手用 Agent 化——或用路由区分（第 7 章）。

## 本章小结

- RAG Agent 化核心：检索是工具，检索轮数是模型决策
- 引用编号 = 全局登记表 + prompt 规则 + 输出校验三件套
- response_format 让答案可直接进业务系统（含 need_human 路由）
- FAQ 缓存层让高频问题成本归零；检索质量用数据集回归保障

> 下一项目：多 Agent 研究助手（Supervisor + 并行研究 + Deep Agents 思路）。
