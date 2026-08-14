# LlamaIndex 框架学习指南（学习路线图版）

> 定位：按学习顺序组织的完整学习指南，从零基础到生产级应用。每站含"核心概念 → 代码实践 → 练习建议"三段式。
>
> 环境约定：代码示例采用通用 OpenAI 写法（`OpenAI(model="gpt-4o-mini")`）。任何 OpenAI 兼容端点（DeepSeek、阿里百炼 DashScope、Moonshot 等）都可以通过设置 `base_url` + `api_key` 替换，代码结构不变。
>
> 文档时效性：基于 LlamaIndex 当前版本生态编写（模块化包结构 + llama-index-core 核心 + 独立集成包；Agent API 已统一到 `llama_index.core.agent.workflow`；Workflows 为独立事件驱动框架；LlamaParse 已并入 LlamaCloud 平台）。官方文档主站在 developers.llamaindex.ai（旧域名 docs.llamaindex.ai 会 301 跳转）。

---

## 学习路线总览

| 阶段 | 主题 | 建议投入 | 学习目标 |
|------|------|---------|---------|
| 第 0 站 | 认知定位与生态地图 | 0.5 天 | 搞清 LlamaIndex 是什么、和 LangChain/LangGraph 怎么分工 |
| 第 1 站 | 环境搭建与五分钟 RAG | 0.5-1 天 | 跑通第一个 RAG 问答，建立全链路直觉；含本地开源方案（Ollama） |
| 第 2 站 | 数据模型与摄入链路 | 1 天 | 掌握 Document/Node、Reader 生态、切分策略、IngestionPipeline、元数据提取器、中文场景优化 |
| 第 3 站 | 索引与向量数据库生态 | 1 天 | 掌握四种索引类型、向量库选型、持久化 |
| 第 4 站 | 查询引擎与检索链路 | 2-3 天 | QueryEngine/ChatEngine、后处理器、元数据过滤、Router 路由、结构化输出、Text-to-SQL/Pandas、Prompts 定制、流式输出 |
| 第 5 站 | 高级检索策略 | 2-3 天 | 句子窗口、自动合并、递归检索、查询变换、语义切分、属性图索引 |
| 第 6 站 | Agent 与 Workflows | 2-3 天 | FunctionAgent、AgentWorkflow 多智能体、自建 Workflow、MCP 工具接入、与 LangGraph 配合 |
| 第 7 站 | LlamaParse 与多模态 | 1 天 | LlamaCloud 平台、复杂文档解析、图片/音频处理 |
| 第 8 站 | 评估与可观测性 | 1 天 | 检索评估（hit_rate/MRR）、响应评估（忠实度/相关性）、追踪调试 |
| 第 9 站 | 生产化要点 | 按需 | 增量摄入、缓存、并发、部署 |

总计约 2.5-3.5 周（每天 2-3 小时的业余节奏）。已有 LangChain/LangGraph 基础（你目前的状态）可以跳过第 0 站大部分内容，第 6 站学起来会特别快——概念和 LangGraph 高度相通。

---

# 第 0 站：认知定位与生态地图

## 0.1 LlamaIndex 是什么

LlamaIndex 是一个专为"让 LLM 使用你的私有数据"设计的数据框架。它解决的核心问题是：LLM 不知道你的公司文档、产品手册、数据库内容，怎么把这些数据高效、结构化地喂给 LLM？

答案的技术路线是 RAG（Retrieval-Augmented Generation，检索增强生成）：把文档切分、向量化、建索引，用户提问时先检索最相关的片段，再交给 LLM 生成回答。LlamaIndex 把这条链路的每一环都做成了标准化的抽象：Reader（读取）→ Node（切分）→ Index（索引）→ Retriever（检索）→ NodePostprocessor（后处理）→ QueryEngine（合成回答）。

一句话记忆：**LangChain/LangGraph 管"怎么编排 LLM 的推理流程"，LlamaIndex 管"怎么把数据高质量地交给 LLM"**。前者是编排框架，后者是数据框架。

## 0.2 生态地图（四个组成部分）

学习 LlamaIndex 最大的认知障碍是"生态里名字太多"。先建立这张地图，后面每一站都会回到它：

**① LlamaIndex Framework（核心框架）**：`llama-index-core` 加上一系列独立集成包。核心包提供 Document/Node/Index/QueryEngine/Agent/Workflow 等抽象；集成包按需安装（如 `llama-index-llms-openai`、`llama-index-vector-stores-chroma`）。这是学习的主战场。

**② LlamaHub（集成市场）**：官方维护的数据连接器（Reader）、LLM、Embedding、向量库、工具的集合。上百个 Reader 覆盖 PDF、Word、Notion、Google Drive、Slack、GitHub、SQL 数据库等数据源。每个集成都是一个 pip 包，用哪个装哪个。地址：llamahub.ai。

**③ LlamaCloud（云端平台）**：LlamaIndex 公司的商业化托管服务，一个 API Key 覆盖六个产品——Parse（智能文档解析）、Extract（结构化抽取）、Classify（文档分类）、Split（文档拆分）、Sheets（表格提取）、Index（托管向量检索）。个人学习主要用它的免费额度试 Parse。

**④ LlamaAgents（Agent 运行时）**：Agent 与 Workflows 相关的能力统称，文档中单列一节。Agent 在当前版本中本质上是"预构建好的 Workflow"——这句话记住，第 6 站会展开。

## 0.3 与 LangChain / LangGraph 的关系

| 维度 | LlamaIndex | LangChain / LangGraph |
|------|-----------|----------------------|
| 核心抽象 | 数据管道（Reader→Index→Query） | 调用链 / 状态图 |
| 强项 | 检索质量、索引策略、文档解析 | Agent 循环、多步编排、状态管理 |
| Agent 能力 | 有（FunctionAgent/AgentWorkflow/Workflow），偏轻量 | 更成熟（LangGraph 图、HITL、持久化） |
| 典型配合 | 用 LlamaIndex 建 RAG 检索器，包装成工具交给 LangGraph Agent | 用 LangGraph 编排，节点内调 LlamaIndex 检索 |

实践建议：RAG 为主的应用选 LlamaIndex 为主框架；Agent 为主、RAG 为辅的应用选 LangGraph 为主框架，把 LlamaIndex 检索器当工具嵌入（第 6.5 节给方案）。

## 0.4 包结构认知（避免装错包）

早期 LlamaIndex 是一个大包（`pip install llama-index` 装全量），现在已模块化：

- `llama-index`：新手捆绑包，装完自带 openai LLM/embedding 等常用集成，适合入门
- `llama-index-core`：纯核心抽象，不带任何集成
- `llama-index-llms-openai`、`llama-index-embeddings-openai`、`llama-index-vector-stores-chroma`……：按需安装的集成包
- `llama-index-workflows`：Workflows 独立包（core 里也内置了同一套 API，通过 `llama_index.core.workflow` 访问）
- `llama-parse` / `llama-cloud`：LlamaCloud 客户端

判断技巧：遇到 `ModuleNotFoundError: No module named 'llama_index.llms.openai'`，说明只装了 core，缺对应集成包。

**第 0 站练习**：不写代码。画一张自己的生态地图（框架/LlamaHub/LlamaCloud/Agent 四块），标出自己想做的应用会用到哪几块。

---

# 第 1 站：环境搭建与五分钟 RAG

## 1.1 安装与配置

```bash
# 方式一：新手推荐，捆绑常用集成（含 openai llm/embedding、文件 reader）
pip install llama-index

# 方式二：精确控制（后续生产推荐）
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai llama-index-readers-file
```

环境变量（`OPENAI_API_KEY` 必须设置，否则默认配置会报错）：

```bash
export OPENAI_API_KEY=sk-xxxx
# Windows cmd: set OPENAI_API_KEY=sk-xxxx
```

> 用国产模型的同学：LlamaIndex 的 OpenAI 类支持 `OpenAI(model="...", api_key="...", base_url="...")`，任何 OpenAI 兼容端点都能接入（DeepSeek、百炼 DashScope compatible-mode、硅基流动等），只需在初始化时传参或设 `OPENAI_API_BASE` 环境变量。

## 1.2 五分钟最小 RAG

准备一个 `data` 目录放几个 txt/md/pdf 文件，然后：

```python
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 全局配置（也可以不配，用默认值）
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 1. 读取：把 data 目录下所有支持的文件加载为 Document 列表
documents = SimpleDirectoryReader("./data").load_data()

# 2. 索引：切分 + 向量化 + 建索引（内部自动走 Settings 里的切分器和 embedding）
index = VectorStoreIndex.from_documents(documents)

# 3. 查询：检索 top-k 相关块 + 交给 LLM 合成回答
query_engine = index.as_query_engine()
response = query_engine.query("这份文档的主要内容是什么？")
print(response)
```

这 10 行代码背后发生了什么（重要，值得逐行理解）：

1. `SimpleDirectoryReader` 递归扫描目录，按文件后缀选择解析器，产出 `Document` 对象（含文本 + 元数据）
2. `from_documents` 内部先用 `SentenceSplitter` 把 Document 切成 `Node`（默认 chunk_size=1024 token，overlap=20）
3. 每个 Node 被 embedding 模型向量化，存入内存向量索引
4. `query()` 时：问题被向量化 → 余弦相似度检索 top-2（默认）→ 命中节点拼进 prompt → LLM 生成带引用的回答

## 1.3 Settings：全局配置中心

`Settings` 是单例配置对象，控制所有"没显式传参"环节的默认行为：

```python
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.context_window = 128000   # 供自动 prompt 裁剪参考
Settings.num_output = 1024
```

优先级规则：显式传参 > Settings > 内置默认值。学习中建议把 llm/embed_model 显式配进 Settings，避免"默认模型不是我想要的"这类困惑。

## 1.4 没有 OpenAI Key？本地开源方案（Ollama）

内网环境、不想花钱、数据不出本机——用 Ollama 跑本地 LLM + HuggingFace 本地 embedding，整套 RAG 完全离线：

```bash
# 1. 安装 Ollama（ollama.com），拉一个模型
ollama pull qwen3:8b          # 中文场景推荐；轻量可用 llama3.1:8b

# 2. 装集成包
pip install llama-index-llms-ollama llama-index-embeddings-huggingface
```

```python
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")  # 首次运行自动下载模型
```

之后的 `VectorStoreIndex.from_documents` 等代码与云端版完全一致——这就是 Settings 抽象的价值。注意三点：首次下载 embedding 模型需要几分钟和几个 GB 磁盘；本地小模型的指令遵循和 function calling 能力弱于云端旗舰（做第 6 站 Agent 实验时建议回云端）；bge-m3 是中英双语模型，中文检索效果优于多数默认选项。

**第 1 站练习**：① 跑通五分钟 demo，换不同问题观察回答质量；② 把 chunk_size 从默认 1024 改成 256 和 2048，各问同一个细节问题，对比回答差异，直观感受切分粒度对 RAG 的影响；③（可选）按 1.4 配置本地 Ollama 方案，验证同一份文档在纯本地环境也能问答。

---

# 第 2 站：数据模型与摄入链路

RAG 的效果上限由数据摄入质量决定——"垃圾进，垃圾出"。这一站解决"怎么把各种数据干净地装进索引"。

## 2.1 Document 与 Node：两级数据模型

**Document**：一个完整数据源的载体（一个 PDF、一个网页、一条数据库记录），核心是 `text` 字段 + `metadata` 字典（来源、页码、作者等）。

**Node**：切分后的最小检索单元（一个文本块），是索引和检索的真正操作对象。Node 继承自 Document 的元数据，并额外携带节点级信息（node_id、与父节点关系、前后块关系）。

```python
from llama_index.core import Document, SimpleDirectoryReader

# 手动构造 Document（数据库/API 数据走这条路）
doc = Document(
    text="这是一段来自工单系统的记录……",
    metadata={"source": "ticket_system", "department": "客服", "priority": "高"},
)

# Reader 产出的就是 Document 列表
documents = SimpleDirectoryReader("./data").load_data()
print(documents[0].metadata)   # {'file_name': 'xxx.pdf', ...}
print(len(documents))          # 文档数
```

关键认知：**metadata 会在检索时透传到 Node**，是后续做元数据过滤（第 4.4 节）和引用溯源的基础。摄入时多存一分元数据，检索时就多一分过滤能力。

## 2.2 Reader 生态：LlamaHub 的正确打开方式

`SimpleDirectoryReader` 覆盖本地常见格式（txt/md/pdf/docx/csv/pptx/图片等）。特殊数据源用 LlamaHub 专用 Reader，模式统一为 `pip install 对应包` → `from 包 import XxxReader` → `load_data()`：

```python
# 例：网页读取
# pip install llama-index-readers-web
from llama_index.readers.web import SimpleWebPageReader
docs = SimpleWebPageReader(html_to_text=True).load_data(["https://example.com/about"])

# 例：Notion（数据库/页面）
# pip install llama-index-readers-notion
from llama_index.readers.notion import NotionPageReader
docs = NotionPageReader(integration_token="...").load_data(page_ids=["..."])
```

LlamaHub 上按数据源检索（llamahub.ai → Readers），每个 Reader 页面都有安装命令和示例。选型原则：官方维护的优先；标 experimental 的预期 API 会变；找不到满意 Reader 时，手写一个函数返回 `Document` 列表就是自定义 Reader（不需要继承任何类）。

## 2.3 切分策略：SentenceSplitter 与参数权衡

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,     # 每块最大 token 数（按 embedding 模型 token 计）
    chunk_overlap=100,   # 相邻块重叠 token 数
)
nodes = splitter.get_nodes_from_documents(documents)
print(nodes[0].get_content()[:100])
print(nodes[0].metadata)      # 继承自 Document
print(nodes[0].node_id)
```

`SentenceSplitter` 是默认切分器，特点：优先在句子/段落边界切，而不是生硬地按 token 硬切。参数权衡：

- **chunk_size 太小（128-256）**：检索命中精度高，但上下文太碎，LLM 拿到的信息不完整，适合事实型 QA
- **chunk_size 太大（2048+）**：上下文完整，但一个块里混多个主题，向量表示被"稀释"，检索精度下降，还费 token
- **经验起点**：通用文档 512-1024；表格/代码慎切（考虑整表一个块）；法律/合同等长条文可配合第 5 站的高级策略

其他内置切分器：`TokenTextSplitter`（纯按 token）、`MarkdownNodeParser`（按标题层级切，md 文档强烈推荐）、`HTMLNodeParser`（按 HTML 标签切）、`JSONNodeParser`（按 JSON 结构切，默认路径递归展开）。

## 2.4 IngestionPipeline：标准化摄入管道

生产中不要直接 `from_documents` 一把梭，用 `IngestionPipeline` 把切分、变换、向量化串成显式管道，并获得缓存与去重能力：

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        embed_model,                    # 向量化也作为一步变换
    ],
    # cache=IngestionCache(...),        # 接入缓存后，重复内容不重复调 embedding（省钱）
    # docstore=docstore,                # 接入 docstore 后，按内容哈希去重、支持增量更新
)
nodes = pipeline.run(documents=documents)
index = VectorStoreIndex(nodes)
```

缓存与去重的价值：文档集合更新时，未变化的块直接命中缓存（不重新花钱算 embedding），新增块才真正处理。第 9 站生产化会再回到这里。

## 2.5 元数据提取器：用 LLM 增强每个块

第 2.1 节说过"摄入时多存一分元数据，检索时就多一分过滤能力"。MetadataExtractor 就是让 LLM 帮你批量生成增强元数据的组件，作为 IngestionPipeline 的变换步骤插入：

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor,               # 给每个块生成小标题
    QuestionsAnsweredExtractor,   # 生成"这个块能回答哪些问题"
    SummaryExtractor,             # 生成块摘要（可含前一块的衔接摘要）
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=2, llm=llm),
        SummaryExtractor(summaries=["prev", "self"], llm=llm),
        embed_model,
    ],
)
nodes = pipeline.run(documents=documents)
print(nodes[0].metadata)   # 多出 node_title / questions_this_doc_can_answer / section_summary
```

原理：这些元数据以文本形式挂在 Node 上，部分向量库（或自定义检索）会把它们纳入匹配范围——"questions_this_doc_can_answer" 与用户真实提问的相似度，往往比原文正文更高（问句对问句）。代价是每个块都要过 LLM，大语料成本可观，适合中小规模高价值语料。

## 2.6 中文场景专项优化

中文不是英文的简单换肤，三个环节要单独留意：

**切分**：`SentenceSplitter` 按标点分句对中文基本可用（句号问号感叹号都能正确断开），但 `chunk_size` 单位是 token——中文一个汉字在多数 tokenizer 下约 1-2 个 token，所以中文文档的"1024 token 块"实际汉字数比英文单词数少得多，建议中文从 256-512 起调。结构化中文文档（产品文档、技术手册）优先用 `MarkdownNodeParser` 按标题层级切。

**Embedding 选型**：默认的 OpenAI text-embedding-3-small 中文效果可用但非最优。中文场景首选 BGE 系列——`BAAI/bge-m3`（中英双语、支持 8192 长文本、稠密+稀疏多模式，本地免费）或 `bge-large-zh-v1.5`（纯中文经典款）。接入方式就是 1.4 节的 `HuggingFaceEmbedding(model_name="BAAI/bge-m3")`，或任何提供中文 embedding 的云端端点。

**Rerank 配套**：第 4.3 节的 `BAAI/bge-reranker-v2-m3` 本身就是中文友好的重排模型，中文场景的"召回-重排"组合拳比英文场景收益更大（中文语义检索的初排噪音偏高）。

验证方式：别凭感觉，用第 8 站的 hit_rate/MRR 评估来对比不同 embedding 与切分参数在**你的语料**上的真实表现——中文语料的最佳参数没有通用答案。

**第 2 站练习**：① 用 MarkdownNodeParser 重新切分一份 md 文档，打印每个 Node，观察标题层级如何进入 metadata；② 给 Document 手动加 `category` 元数据字段，为第 4 站的元数据过滤做数据准备；③ 用 IngestionPipeline 跑一遍与 from_documents 等价的流程；④ 给小语料接 QuestionsAnsweredExtractor，打印生成的元数据，再用手动提问题验证这些元数据是否真的提升了命中率。

---

# 第 3 站：索引与向量数据库生态

## 3.1 四种核心索引类型

```python
from llama_index.core import (
    VectorStoreIndex,        # 向量索引（默认，语义检索）
    SummaryIndex,            # 摘要索引（遍历全部节点，适合"总结全文"）
    KeywordTableIndex,       # 关键词索引（BM25 风格，关键词精确匹配）
    KnowledgeGraphIndex,     # 知识图谱索引（抽取三元组建图，适合多跳推理）
)
```

- **VectorStoreIndex**：90% 场景的选择。语义相似度检索，能理解"同义不同词"
- **SummaryIndex**：不检索，把所有节点按顺序喂给 LLM（配合 refine 模式逐块总结）。适合"这篇文档讲了什么"这类全局问题——向量检索在这类问题上反而容易漏
- **KeywordTableIndex**：从节点提取关键词建倒排表。适合精确术语查询（型号、人名、错误码），常与向量索引组成混合检索
- **KnowledgeGraphIndex**：用 LLM 抽取实体-关系三元组。构建成本高（每块都要过 LLM），适合关系密集领域（医学、金融风控）

同一个 Node 列表可以同时建多个索引，按问题类型路由到不同索引（第 6 站的 Agent 路由就是干这个的）。

## 3.2 向量数据库生态与选型

`VectorStoreIndex` 默认把向量存在内存里，进程退出即丢。生产需要接入外部向量库。所有集成的接入模式高度统一：**建 vector_store 对象 → 包进 StorageContext → from_documents / from_vector_store**。

| 向量库 | 定位 | 安装 | 适用 |
|--------|------|------|------|
| 内存（默认） | 零依赖 | 无 | demo、小数据 |
| FAISS | 本地库，Meta 出品 | `faiss-cpu` + `llama-index-vector-stores-faiss` | 单机、数据量中等、不想起服务 |
| Chroma | 本地嵌入式，可持久化 | `chromadb` + `llama-index-vector-stores-chroma` | 单机开发到小规模生产，最顺手 |
| Milvus | 分布式专业向量库 | `llama-index-vector-stores-milvus` | 大规模、高并发生产 |
| Qdrant | Rust 实现，性能好 | `llama-index-vector-stores-qdrant` | 生产，支持本地/云 |
| pgvector | Postgres 插件 | `llama-index-vector-stores-postgres` | 已有 PG 基础设施，数据统一管理 |
| Pinecone | 全托管云服务 | `llama-index-vector-stores-pinecone` | 免运维，按量付费 |

以 Chroma 为例的完整流程（建库 → 查询 → 复用）：

```python
# pip install chromadb llama-index-vector-stores-chroma
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

# ===== 建库（一次性）=====
documents = SimpleDirectoryReader("./data").load_data()
db = chromadb.PersistentClient(path="./chroma_db")          # 数据落盘到本地目录
collection = db.get_or_create_collection("my_docs")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# ===== 复用（后续每次）=====
db2 = chromadb.PersistentClient(path="./chroma_db")
vector_store2 = ChromaVectorStore(chroma_collection=db2.get_or_create_collection("my_docs"))
index2 = VectorStoreIndex.from_vector_store(vector_store2)   # 不重新 embedding
print(index2.as_query_engine().query("……"))
```

FAISS 版本的差异只在 vector_store 构造（`faiss.IndexFlatL2(dim)` 包装一层），其余代码完全不变——这就是统一抽象的价值。

## 3.3 持久化：StorageContext

不接外部向量库时，也能用内置存储持久化（存的是本地 JSON/pickle 文件）：

```python
from llama_index.core import StorageContext, load_index_from_storage

# 保存
index.storage_context.persist(persist_dir="./storage")

# 加载
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

注意：内置持久化保存的是"已算好的 embedding"，加载时不重新调 embedding 接口；但如果换了 embedding 模型，旧索引必须重建（向量空间不同，不能混用）。

**第 3 站练习**：① 同一份文档分别建 VectorStoreIndex 和 SummaryIndex，问"文档主旨"和"某个具体细节"两类问题，观察哪个索引更擅长哪类；② 用 Chroma 完整走一遍"建库→重启进程→加载→查询"，确认持久化生效；③ FAISS 替换 Chroma 再走一遍，体会接口统一性。

---

# 第 4 站：查询引擎与检索链路

这一站把 `as_query_engine()` 这一个入口拆开，看清内部的五个环节：Retriever（检索）→ NodePostprocessor（后处理）→ Response Synthesizer（合成）→ 输出。理解了内部结构，才知道怎么调优。

## 4.1 QueryEngine 的可控参数

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,            # 检索召回块数（默认 2，常调到 3-10）
    response_mode="compact",       # 合成策略（见 4.2）
    node_postprocessors=[...],     # 后处理器（见 4.3）
    streaming=True,                # 流式输出
)
response = query_engine.query("……")
print(response)                    # Response 对象，支持 print 直接输出
print(response.source_nodes)       # 引用来源！溯源就靠它
for node in response.source_nodes:
    print(node.score, node.metadata.get("file_name"))
```

`response.source_nodes` 是被检索使用且相似度非零的节点列表，带 `score`、`metadata`——做"回答引用了哪些原文"的溯源 UI 就靠它。

## 4.2 response_mode：四种合成策略

合成器（Response Synthesizer）决定"检索到的块怎么喂给 LLM"：

- **compact**（默认）：把检索块拼接/压缩进单个 prompt，一次调用。快，最常用
- **refine**：逐块顺序喂给 LLM，每次"带着之前的答案精炼"。质量稳但 N 块 = N 次调用，慢且贵
- **tree_summarize**：两两合并总结再向上归并成树，适合长文档全局总结
- **no_text**：不调 LLM，直接返回检索到的块（做纯检索调试时用）

实践：默认 compact；用户反馈"答案丢细节"时试 refine；做全文总结时用 tree_summarize（配 SummaryIndex）。

## 4.3 NodePostprocessor：检索后处理

检索完成后、合成之前，可以插拔后处理器过滤/重排节点：

```python
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
# 本地 rerank 模型版（需 pip install sentence-transformers）
from llama_index.core.postprocessor import SentenceTransformerRerank

query_engine = index.as_query_engine(
    similarity_top_k=10,                       # 粗召回 10 块
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7),        # ① 分数过滤
        KeywordNodePostprocessor(required_keywords=["2026"]),  # ② 关键词过滤
        SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=4),  # ③ 精排取 4
    ],
)
```

经典组合拳"召回-重排"（retrieve-then-rerank）：向量检索粗召回 top 10 → 用专门的 rerank 模型精排出 top 3-5。这是提升 RAG 命中率最立竿见影的手段之一（另一个是第 5 站的检索策略）。云服务版可用 Cohere Rerank（`llama-index-postprocessor-cohere-rerank`）。

## 4.4 元数据过滤

摄入时存的 metadata，在这里变现：

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

filters = MetadataFilters(filters=[
    MetadataFilter(key="department", value="客服", operator=FilterOperator.EQ),
    MetadataFilter(key="year", value=2025, operator=FilterOperator.GTE),
])
query_engine = index.as_query_engine(filters=filters)
```

适合"只在某类文档里搜"的场景（按部门、按时间、按文件类型）。注意：不同向量库对 FilterOperator 的支持程度不同（EQ 基本全支持，复杂操作看具体库）。

## 4.5 ChatEngine：多轮对话

QueryEngine 是一问一答、无记忆；ChatEngine 在其上加了"对话历史压缩"层：

```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",   # 推荐模式：先压缩历史+改写问题，再走 RAG
    memory=memory,
)
response = chat_engine.chat("公司的年假政策是什么？")
response = chat_engine.chat("那病假呢？")   # "那病假呢"会被结合上下文改写后再检索
print(chat_engine.chat_history)           # 查看对话历史
```

`condense_plus_context` 的工作原理：每轮先把历史对话 + 新问题交给 LLM 改写成独立完整的问题（"那病假呢？" → "公司的病假政策是什么？"），再检索。这解决了 RAG 多轮对话最核心的指代消解问题。

## 4.6 流式输出

```python
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("……")
for token in response.response_gen:    # 生成器，逐 token 消费
    print(token, end="", flush=True)
```

ChatEngine 同理：`response = chat_engine.stream_chat("……")` 后迭代 `response.response_gen`。

## 4.7 Router：多索引间的动态路由

第 3.1 节埋的伏笔在这里兑现：多个索引各自擅长不同问题（向量索引擅长细节、SummaryIndex 擅长全局），Router 负责"让 LLM 根据问题选路"：

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata

vector_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(name="facts", description="查询文档中的具体细节、条款、数据"),
)
summary_tool = QueryEngineTool(
    query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(name="summary", description="总结文档主旨、整体内容"),
)

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=llm),   # 单选；LLMMultiSelector 可同时选多个
    query_engine_tools=[vector_tool, summary_tool],
)
print(router_engine.query("这份文档的核心观点是什么？"))   # 自动路由到 summary
print(router_engine.query("年假有多少天？"))               # 自动路由到 facts
```

selector 依据各工具的 `description` 让 LLM 做选择题（输出路由名，不执行）。Router 与 Agent 的分界：Router 只做**一次性选路**（无循环、无多步），Agent 可以**多步循环**地调工具和反思——轻量分流用 Router，复杂决策用第 6 站的 Agent。

## 4.8 结构化输出：从文本抽取 Pydantic 对象

让 LLM 按 schema 吐 JSON 而不是自由文本，两条路：

```python
from pydantic import BaseModel, Field

class PolicyItem(BaseModel):
    title: str = Field(description="制度条目名称，如：年假")
    days: int = Field(description="对应天数")

class HRPolicy(BaseModel):
    items: list[PolicyItem]
    summary: str = Field(description="整体摘要，一句话")

# 路线一：LLM 直出（不需要检索，直接从给定文本抽取）
sllm = llm.as_structured_llm(HRPolicy)
output = sllm.complete(f"从以下制度文本中抽取条目：\n{text}")
print(output.raw)            # HRPolicy 实例，字段可直接访问
print(output.raw.items[0].title)

# 路线二：RAG + 结构化（先检索，再把命中内容按 schema 抽取）
qa_engine = index.as_query_engine(output_cls=HRPolicy, response_mode="compact")
result = qa_engine.query("公司有哪些假期制度？")
print(result.response)       # HRPolicy 对象
```

底层靠 function calling / JSON Schema 约束输出，字段描述写得好坏直接影响抽取质量。典型用途：制度条款抽取、工单字段归一化、简历解析——以及给第 6 章 Agent 提供"工具返回结构化数据"的能力。

## 4.9 结构化数据查询：Text-to-SQL 与 Pandas

非文档数据（数据库表、DataFrame）不用做 RAG，LlamaIndex 有专门的查询引擎：

```python
# ===== 自然语言查数据库 =====
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine

engine = create_engine("sqlite:///company.db")
sql_engine = NLSQLTableQueryEngine(
    engine, tables=["employees"], llm=llm,   # 只暴露白名单表
)
print(sql_engine.query("入职超过 5 年的员工有多少人？"))
# 内部流程：取表结构 → 生成 SQL → 执行 → LLM 组织成自然语言答案

# ===== 自然语言查 DataFrame =====
from llama_index.core.query_engine import PandasQueryEngine

df = ...  # 任意 pandas DataFrame
pandas_engine = PandasQueryEngine(df=df, verbose=True)   # verbose 可看到生成的代码
print(pandas_engine.query("各部门平均薪资是多少？"))
# 内部流程：把 df 的列结构告诉 LLM → 生成 pandas 表达式 → 执行 → 返回结果
```

安全要点：生产中数据库账号给**只读权限**、表走白名单；PandasQueryEngine 会执行 LLM 生成的 Python 代码，处理敏感数据时注意沙箱（与你 chapter08 数据分析 Agent 里 PythonAstREPLTool 的安全注意事项是同一件事）。

## 4.10 Prompts 定制：改写合成模板

不满意默认回答风格（英文腔、爱编造、太啰嗦）时，替换查询引擎的 prompt 模板：

```python
from llama_index.core import PromptTemplate

QA_PROMPT = PromptTemplate(
    "以下是背景信息：\n"
    "---------------------\n{context_str}\n---------------------\n"
    "请仅依据背景信息用中文回答问题。如果背景信息不足以回答，"
    "直接说\"资料中未提及\"，不要编造。\n"
    "问题：{query_str}\n回答："
)
query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)
```

可替换的模板位有两个：`text_qa_template`（compact 默认合成模板）和 `refine_template`（refine 模式的精炼模板）。调试技巧：`print(query_engine.get_prompts())` 能看到引擎当前实际使用的全部模板——先打印默认模板再改，比盲改高效得多。中文场景把"找不到就说未提及"写进模板，是压制幻觉的低成本手段（配合第 8 章忠实度评估验证效果）。

**第 4 站练习**：① 打印 response.source_nodes 做 5 次提问的溯源分析，检查有没有"检索跑偏"（命中了无关文档）；② 实践召回-重排：先 no_text 模式看 top10 原始召回，加上 reranker 后对比 top4 是否更相关；③ 用 ChatEngine 连续问 3 轮带指代的问题（"介绍下 A"→"它多少钱"→"比 B 呢"），验证改写机制；④ 建"向量 + Summary"双索引，用 RouterQueryEngine 各问两类问题，打印路由选择是否符合预期；⑤ 用 output_cls 从制度文档抽取结构化假期条目；⑥（有数据库的话）给一张小表跑通 NLSQLTableQueryEngine。

---

# 第 5 站：高级检索策略（进阶 RAG）

第 4 站的手段都是在"检索到的块"上做文章。这一站解决更根本的问题：**块本身切得对不对**。三个核心矛盾：小块检索准但上下文碎；大块上下文全但检索散；一问一检索单路召回覆盖不了复杂问题。

## 5.1 句子窗口检索（Sentence Window）

思路：**按句子切做检索单位（小而准），命中后把前后各 N 句拼成窗口送 LLM（大而全）**。

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

# ① 按句切分，每个节点带 window 元数据（前后各 3 句）
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
nodes = node_parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

# ② 查询时用 MetadataReplacementPostProcessor 把"句子"替换成"窗口"
query_engine = index.as_query_engine(
    similarity_top_k=6,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window"),
        SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=3),
    ],
)
```

适合：细节事实型问答（条款、参数、定义），文档语料偏叙事性文本。

## 5.2 自动合并检索（Auto-Merging，层级分块）

思路：**同时按大中小三档切块（如 2048/512/128）建层级，检索只搜最小块；若同一父块下的小块被大量命中，自动"合并"成父块送 LLM**——碎块拼回完整章节。

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex, StorageContext

# ① 三档切分
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
all_nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(all_nodes)

# ② 全部节点（含父块）存入 docstore，供回溯合并
docstore = SimpleDocumentStore()
docstore.add_documents(all_nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)

# ③ 只用叶子节点建索引；检索时包一层 AutoMergingRetriever
automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
base_retriever = automerging_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
query_engine = RetrieverQueryEngine.from_args(retriever)
```

适合：结构化文档（手册、报告、合同），问题有时落在段落级、有时落在章节级。

## 5.3 递归检索（Recursive Retrieval）

思路：**建两级索引——摘要/标题级索引先粗定位，命中后跟随引用跳到对应的细粒度块**。用 `IndexNode`（带引用的节点）实现"小块指向大块"：

```python
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core import VectorStoreIndex

# 概念示例：每个小结的"摘要句子"作为索引节点，指向该小结的完整文本块
chunks = {
    "chunk_1": TextNode(text="第一章完整内容……很长的正文"),
    "chunk_2": TextNode(text="第二章完整内容……很长的正文"),
}
index_nodes = [
    IndexNode(text="第一章介绍系统架构设计", index_id="chunk_1"),
    IndexNode(text="第二章讲解部署与运维", index_id="chunk_2"),
]
summary_index = VectorStoreIndex(index_nodes)

retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": summary_index.as_retriever(similarity_top_k=1)},
    node_dict=chunks,   # 命中 IndexNode 后，按 index_id 取出真正的完整块
    verbose=True,
)
nodes = retriever.retrieve("系统架构是怎么设计的？")
```

适合：多文档集合（每篇文档的摘要 → 定位 → 拉全文）、表格/图片与文字混排（文字描述指向表格节点）。这是 LlamaIndex 官方高级教程的招牌模式，思想比代码重要。

## 5.4 查询变换：改造问题再检索

检索质量差有时不是数据问题，是**问题本身不好检索**（太笼统、复合问题、口语化）。三种武器：

**子问题分解（SubQuestionQueryEngine）**——把复合问题拆成多个子问题，分别检索各自的数据源再汇总：

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

qa_tool_2025 = QueryEngineTool(
    query_engine=index_2025.as_query_engine(),
    metadata=ToolMetadata(name="report_2025", description="2025 年年度报告"),
)
qa_tool_2026 = QueryEngineTool(
    query_engine=index_2026.as_query_engine(),
    metadata=ToolMetadata(name="report_2026", description="2026 年年度报告"),
)
engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=[qa_tool_2025, qa_tool_2026])
response = engine.query("对比 2025 和 2026 年的营收变化")   # 自动拆成两个子问题分别检索
```

**HyDE（假设性文档嵌入）**——先让 LLM 生成一段"假想的答案"，用假答案的向量去检索（答案和答案的相似度 > 问题和答案的相似度）：

```python
from llama_index.core.indices.query.transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

base_engine = index.as_query_engine()
hyde = HyDEQueryTransform(include_original=True)
hyde_engine = TransformQueryEngine(base_engine, hyde)
print(hyde_engine.query("服务器扩容的操作步骤"))
```

**多查询（Multi-Query）**——把一个问题改写成多个变体分别检索，结果合并去重，提高召回覆盖面。

## 5.5 知识图谱索引：KnowledgeGraphIndex 与 PropertyGraphIndex

图谱索引用 LLM 从文本抽取（主体, 关系, 客体）三元组建图，查询时沿关系多跳遍历，适合"多跳问题"（A 的合作方的竞争对手是谁）。LlamaIndex 有两代实现：

```python
# 旧版：KnowledgeGraphIndex（内置三元组抽取，小语料快速起步）
from llama_index.core import KnowledgeGraphIndex
kg_index = KnowledgeGraphIndex.from_documents(documents, max_triplets_per_chunk=5)

# 新版（当前推荐）：PropertyGraphIndex——属性图模型，实体和关系都可带属性
from llama_index.core import PropertyGraphIndex
pg_index = PropertyGraphIndex.from_documents(
    documents, llm=llm, embed_model=embed_model, show_progress=True,
)
query_engine = pg_index.as_query_engine()   # 支持向量+图混合检索
```

`PropertyGraphIndex` 相比旧版的进化：抽取策略可插拔（schema 引导的 `SchemaLLMPathExtractor`、动态的 `DynamicLLMPathExtractor`、零 LLM 成本的 `ImplicitPathExtractor`）；图存储可外接 Neo4j（`llama-index-graph-stores-neo4j`）或嵌入式 Kuzu，大规模图不受内存限制；查询支持 text-to-Cypher 与向量检索结合。代价依旧：构建期大量 LLM 调用（贵且慢），大语料慎用；知识密集型小语料（医学、金融风控、企业股权关系）收益明显。初学阶段理解思想即可，优先吃透 5.1-5.4。

## 5.6 语义切分：SemanticSplitterNodeParser

第 2 站的切分器都按"长度/结构"切，语义切分按"主题边界"切——用 embedding 计算相邻句的相似度，在相似度骤降（话题切换）处下刀：

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser

splitter = SemanticSplitterNodeParser(
    buffer_size=1,                        # 相似度窗口
    breakpoint_percentile_threshold=95,   # 相似度差异超过该分位 → 切分
    embed_model=embed_model,
)
nodes = splitter.get_nodes_from_documents(documents)
# 块大小不再固定：同一主题再长也尽量不切，主题一换立即开新块
```

适合：主题边界模糊、没有清晰结构（标题/段落）的长文（访谈记录、会议纪要、小说）。代价：每个短句都要过一次 embedding（摄入成本上升数倍）。`breakpoint_percentile_threshold` 是核心旋钮——调低切得更碎，调高块更大更完整。与句子窗口（5.1）可组合：语义切分定边界 + 窗口保上下文。

**第 5 站练习**：① 选一份长 PDF，分别用"朴素 512 切分"和"句子窗口"跑同一组 10 个细节问题，人工对比命中质量；② 实现 Auto-Merging（verbose=True 观察合并日志，看哪些查询触发了父块合并）；③ 用 SubQuestionQueryEngine 问一个必须对比两个数据源的问题，打印它自动生成的子问题；④（可选）对一份会议纪要试 SemanticSplitter，打印各块首句，观察切分点是否落在话题切换处。

---

# 第 6 站：Agent 与 Workflows

有 LangGraph 基础，这一站可以快进：概念几乎一一对应，对照表先给出来。

## 6.1 概念对照表（LangGraph → LlamaIndex）

| LangGraph | LlamaIndex | 备注 |
|-----------|-----------|------|
| StateGraph + State | Workflow + Context | Workflow 用事件驱动替代显式 State |
| 节点函数 | @step 装饰的步骤函数 | 步骤按"事件类型注解"自动连接 |
| 条件边路由 | 返回不同 Event 类型 / ctx.send_event | 分支就是 if 返回不同事件 |
| Checkpointer | ctx.store / Memory | 运行内状态用 store，跨会话记忆用 memory 对象 |
| interrupt / HITL | workflow 内事件挂起 + human-in-the-loop 模式 | 文档有专门示例 |
| create_react_agent | FunctionAgent / ReActAgent | 预构建 Agent，开箱即用 |
| 子图 | Workflow 组合 / AgentWorkflow | Agent 本质是预构建 Workflow |
| astream_events | handler.stream_events() | 细粒度流式事件 |

## 6.2 工具：从 FunctionTool 到 QueryEngineTool

```python
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata

# ① 普通函数工具（与 LangChain @tool 等价，靠类型注解+docstring 生成 schema）
def multiply(a: float, b: float) -> float:
    """两个数相乘，返回乘积"""
    return a * b
tool = FunctionTool.from_defaults(fn=multiply)

# ② RAG 工具（LlamaIndex 特色）：把整个查询引擎包装成工具给 Agent 用
rag_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(
        name="company_docs",
        description="回答公司规章制度、年假、报销等问题时使用",
    ),
)
```

`QueryEngineTool` 是两个框架的天然桥点：任何 LlamaIndex 索引都能一行代码变成 LangGraph/LlamaIndex Agent 的工具。

## 6.3 FunctionAgent：当前标准 Agent API

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini")

agent = FunctionAgent(
    tools=[multiply, rag_tool],
    llm=llm,
    system_prompt="你是公司助手，规章制度问题用 company_docs 工具查询，计算用乘法工具。",
)

async def main():
    response = await agent.run(user_msg="年假是多少天？顺便算一下 12 乘 3.5")
    print(response)

asyncio.run(main())
```

要点：① 导入路径统一为 `llama_index.core.agent.workflow`（老的 `OpenAIAgent`/`agent_runner` 是历史 API，新代码别用）；② `run()` 是 async，默认流式；③ `FunctionAgent` 要求 LLM 支持 function calling（gpt-4o 系列、qwen、deepseek 都支持），不支持的工具调用模型用 `ReActAgent`（提示词驱动）或 `CodeActAgent`（生成代码执行）。

## 6.4 AgentWorkflow：多智能体协作

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

researcher = FunctionAgent(
    name="researcher",
    tools=[rag_tool],
    llm=llm,
    system_prompt="你是资料研究员，负责从公司文档中检索信息，完成后把结果交给 writer。",
)
writer = FunctionAgent(
    name="writer",
    tools=[],
    llm=llm,
    system_prompt="你是撰稿人，根据研究员提供的资料输出结构化回答。",
)
workflow = AgentWorkflow(agents=[researcher, writer], root_agent=researcher)

async def main():
    result = await workflow.run(user_msg="总结一下新版报销制度的变化")
    print(result)

asyncio.run(main())
```

Agent 间流转通过 `handoff_to("writer")`（在 system_prompt 里指示，或作为工具调用）完成；`root_agent` 指定入口。对照 LangGraph 的 Supervisor 模式理解：AgentWorkflow 更轻，LangGraph 的控制力更强（自定义路由、HITL 暂停点、状态持久化更细）。

## 6.5 Workflows：从零自建（对应手搭 LangGraph 图）

当预构建 Agent 不够用时，用 Workflow 自己写事件驱动的多步骤流程：

```python
import asyncio
from typing import Any
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, Context, step
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

class RetrievedEvent(Event):
    query: str          # 自定义事件：Pydantic 风格，带数据的类型化消息

class RAGWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, **kwargs):
        super().__init__(**kwargs)
        self._index = index

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievedEvent:
        query = ev.get("query", "")                      # StartEvent 放 run() 的 kwargs
        await ctx.store.set("query", query)              # store：跨步骤共享状态
        return RetrievedEvent(query=query)               # 返回什么事件，就触发消费该事件的步骤

    @step
    async def synthesize(self, ctx: Context, ev: RetrievedEvent) -> StopEvent:
        query = await ctx.store.get("query")
        response = self._index.as_query_engine(similarity_top_k=3).query(query)
        return StopEvent(result=str(response))           # StopEvent 结束整个流程

async def main():
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    wf = RAGWorkflow(index=index, timeout=60, verbose=True)
    result = await wf.run(query="文档里提到的主要风险是什么？")
    print(result)

asyncio.run(main())
```

核心心智模型：**每个 @step 声明"我消费什么事件、产出什么事件"，框架根据类型注解自动连线**（相当于隐式的条件边）；分支 = 返回不同事件类型；循环 = 后面的步骤产出前面步骤消费的事件；并发 = 返回 `list[Event]` 或用 `ctx.send_event` / `ctx.collect_events`。`run()` 返回 `WorkflowHandler`，`await` 拿结果，或 `handler.stream_events()` 拿过程事件流。

进阶控制流四件套（对应 LangGraph 的高级特性）：`ctx.collect_events(ev, [EventA, EventB])`——等齐指定的一组事件再触发下一步，实现并行结果的**汇聚**（对应 LangGraph 多边汇合）；`ctx.send_event`——运行时动态**扇出**不定数量的并行任务（对应 `Send`）；Human-in-the-loop——用一个"等待人工输入"的事件把流程挂起，人工回复后以新事件续跑（官方 Workflows 文档有 story-crafting 的 HITL 完整示例）；`workflow.validate()`——测试中校验事件图连通性（有没有产出无人消费的事件、有没有到不了 StopEvent 的死路）。

> 生态说明：Workflows 也有独立发行版 `pip install llama-index-workflows`（`from workflows import Workflow, step`），`llama-index-core` 内置同款 API（`llama_index.core.workflow`）。跟着框架装就不用额外操心。

## 6.6 与 LangGraph 配合（推荐架构之一）

LlamaIndex 负责检索质量，LangGraph 负责编排控制，桥点是 QueryEngineTool + LangChain Tool 适配：

```python
# 思路示例：LlamaIndex 检索器包装成 LangChain Tool，挂到 LangGraph 的 create_react_agent
from langchain_core.tools import tool
from llama_index.core import VectorStoreIndex

_index = VectorStoreIndex.from_documents(documents)   # LlamaIndex 侧

@tool
def search_docs(query: str) -> str:
    """在公司文档库中检索相关信息。参数 query: 检索问题"""
    nodes = _index.as_retriever(similarity_top_k=3).retrieve(query)
    return "\n\n".join(n.get_content() for n in nodes)

# 之后：langgraph 的 create_react_agent(model=llm, tools=[search_docs])
```

反向组合也成立：LlamaIndex FunctionAgent 的工具列表里放一个调用外部系统的函数。选型建议：RAG 流程复杂（多级检索、多数据源）→ LlamaIndex 为主；Agent 流程复杂（HITL、多轮状态、复杂路由）→ LangGraph 为主。

## 6.7 MCP：接入开放工具协议

MCP（Model Context Protocol）是 Anthropic 发起的工具接入开放协议，正在成为 Agent 生态的"USB 口"——任何 MCP server 暴露的能力（文件系统、GitHub、数据库、浏览器……）都能被任何支持 MCP 的 Agent 消费。LlamaIndex 双向支持：作为 **client** 把外部 MCP server 的工具挂给自己的 Agent，也可以把自身能力发布为 MCP server。client 侧用法：

```python
# pip install llama-index-tools-mcp
from llama_index.tools.mcp import McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent

# 连接任意 MCP server（本地起的 filesystem server、第三方服务均可）
tool_spec = McpToolSpec.from_server_url("http://localhost:8000/sse")
mcp_tools = await tool_spec.to_tool_list_async()   # server 暴露的工具自动转为 LlamaIndex Tool

agent = FunctionAgent(
    tools=mcp_tools,
    llm=llm,
    system_prompt="你可以使用提供的 MCP 工具完成任务。",
)
```

对照理解：MCP 之于 Agent 工具，约等于 LlamaHub 之于数据连接器——一个是标准化的外部能力接口，一个是标准化的数据源接口。官方还提供了一个文档检索 MCP server（developers.llamaindex.ai/mcp），可以让任何支持 MCP 的编程助手直接查 LlamaIndex 文档。对学习者的意义：掌握 McpToolSpec 后，你的 Agent 工具箱从"自己写的函数"扩展到"整个 MCP 生态"。

**第 6 站练习**：① 用 FunctionAgent 挂一个 QueryEngineTool 跑通"Agent 自主决定何时查文档"；② 把 6.5 的 RAGWorkflow 扩展成三步（检索 → 相关性判断，不相关返回重新检索事件形成循环 → 合成），体会事件驱动循环；③ 用 LangChain @tool 包装 LlamaIndex 检索器挂到 LangGraph create_react_agent，跑通双框架协作；④（可选）本地起一个 MCP server（如官方 filesystem server），用 McpToolSpec 接入 Agent 完成一次文件操作。

---

# 第 7 站：LlamaParse 与多模态

## 7.1 LlamaCloud 平台：六大产品

LlamaParse 已升级为 LlamaCloud 平台产品族，一个 API Key 六个能力（在 cloud.llamaindex.ai 注册获取 `LLAMA_CLOUD_API_KEY`）：

| 产品 | 能力 | 典型场景 |
|------|------|---------|
| Parse | 智能文档解析（agentic OCR），PDF/扫描件/130+ 格式转 Markdown | 扫描件、复杂排版 PDF 入库 |
| Extract | 按自定义 schema（Pydantic）抽取结构化 JSON | 发票、简历、合同字段提取 |
| Classify | 自然语言规则做文档分类路由 | 工单/文档自动分派 |
| Split | 把拼接合并过的 PDF 按逻辑拆分 | 扫描批次文件还原 |
| Sheets | 从混乱表格提取结构化数据（Parquet） | 财务报表、Excel 治理 |
| Index | 托管向量检索（dense/sparse 混合、rerank） | 免自建向量库的 RAG |

新版客户端用法（`pip install llama-cloud>=2.8`）：

```python
from llama_cloud import LlamaCloud

client = LlamaCloud()   # 自动读 LLAMA_CLOUD_API_KEY 环境变量

file = client.files.create(file="report.pdf", purpose="parse")
result = client.parsing.parse(
    file_id=file.id,
    tier="agentic",          # 智能解析档位（另有 FAST 等模式）
    version="latest",
    expand=["markdown"],
)
markdown_text = result.markdown.pages[0].markdown
print(markdown_text)
```

传统用法（`pip install llama-parse`，老教程常见，仍然可用）：

```python
from llama_parse import LlamaParse
documents = LlamaParse(result_type="markdown", api_key="llx-...").load_data("./report.pdf")
```

另外生态中出现了轻量本地解析器 **LiteParse**（对应 LlamaHub 的 `llama-index-readers-litellm` 一类的轻量集成思路），适合不想上云的简单解析场景。个人学习路径：本地 PyPDF/pdfplumber 处理简单 PDF → 免费额度试 LlamaParse 感受复杂文档的解析质量差距 → 生产按预算决策。

## 7.2 多模态：图片理解

```python
# pip install llama-index-multi-modal-llms-openai
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader

# 读取本地图片
image_documents = SimpleDirectoryReader(input_files=["chart.png"]).load_data()

mm_llm = OpenAIMultiModal(model="gpt-4o-mini", max_new_tokens=512)
response = mm_llm.complete(
    prompt="描述这张图表的内容，并提取其中的关键数字。",
    image_documents=image_documents,
)
print(response.text)
```

这是"多模态 RAG"的地基：用多模态 LLM 给图片生成文字描述 → 描述进向量索引 → 检索命中后把原图+描述一起交给多模态 LLM 回答。任何 OpenAI 兼容的多模态端点（qwen-vl 系列）都可同样接入。

## 7.3 音频与视频

音频走"转写 → 文本索引"：LlamaHub 有音频转写 Reader（底层 Whisper 类服务），产出 Document 后走第 2 站标准链路：

```python
# pip install llama-index-readers-audio（以 LlamaHub 页面为准，API 可能微调）
from llama_index.readers.audio import AudioTranscriptReader
docs = AudioTranscriptReader(file=["meeting.mp3"])   # 转写为文本 Document
```

视频没有一步到位的官方 Reader，通用做法：`ffmpeg` 抽音轨 → 音频转写索引；需要画面理解时按关键帧截图 → 多模态 LLM 生成帧描述 → 与字幕文本一起索引。记住原则：**多模态 RAG 的本质是"把非文本模态翻译成高质量的文本描述，再进文本检索链路"**，各种花式玩法都是这个原则的变体。

**第 7 站练习**：① 找一份带表格的 PDF，分别用本地 PDF Reader 和 LlamaParse 解析，对比表格还原质量；② 用多模态 LLM 给 3 张业务截图生成描述并入库，测试"用文字搜出图"的体验；③ 录一段 1 分钟音频，走完"转写→索引→问答"全链路。

---

# 第 8 站：评估与可观测性

没有评估的 RAG 调优是"盲调"。LlamaIndex 内置两层评估：检索层（ retriever 好不好）和响应层（LLM 答得好不好）。

## 8.1 检索评估：hit_rate 与 MRR

```python
import asyncio
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    RetrieverEvaluator,
)

# ① 用 LLM 自动生成"问题-标准出处"数据集（也可以人工标注，质量更高）
qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2,
)

# ② 评估器：hit_rate（正确块是否进了 top-k）+ MRR（正确块排名的倒数）
retriever = index.as_retriever(similarity_top_k=3)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["hit_rate", "mrr"], retriever=retriever,
)

async def eval_all():
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    hit = sum(r.metric_vals_dict["hit_rate"] for r in eval_results) / len(eval_results)
    mrr = sum(r.metric_vals_dict["mrr"] for r in eval_results) / len(eval_results)
    print(f"hit_rate: {hit:.3f}  mrr: {mrr:.3f}")

asyncio.run(eval_all())
```

用法：把它当成"检索配置的单元测试"——改 chunk_size、换 embedding、加 reranker、换检索策略，每次跑一遍看分数变化。hit_rate 是第一优先指标（找不到就是全错），MRR 反映排序质量。

## 8.2 响应评估：忠实度与相关性

```python
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

query = "年假有多少天？"
response = query_engine.query(query)

faithfulness = FaithfulnessEvaluator(llm=llm)   # 回答是否忠实于检索到的原文（防幻觉）
relevancy = RelevancyEvaluator(llm=llm)         # 回答是否切题

f_result = faithfulness.evaluate_response(query=query, response=response)
r_result = relevancy.evaluate_response(query=query, response=response)
print(f"忠实度: {'通过' if f_result.passing else '失败'}  {f_result.feedback}")
print(f"相关性: {'通过' if r_result.passing else '失败'}  {r_result.feedback}")
```

三大指标分工：**Faithfulness**（答案有没有编造——检索质量差或 LLM 瞎编时亮红灯）、**Relevancy**（答案跑没跑题）、**Correctness**（与标准答案比对，需要标注数据）。诊断口诀：忠实度低 → 先修检索（第 4/5 站）；忠实度高但相关性低 → 修 prompt 和 response_mode；都对但用户不满意 → 数据本身不覆盖问题，补文档。

## 8.3 可观测性：Instrumentation 与第三方追踪

LlamaIndex 的追踪基于 instrumentation（dispatcher）体系，第三方通过回调包接入：

```bash
pip install llama-index-callbacks-arize-phoenix   # 或 llama-index-callbacks-langfuse
```

以 Arize Phoenix（开源，可本地起 UI）为例，安装配置后每次查询的完整链路（检索了哪些块、每步 LLM 的输入输出、耗时）都会可视化呈现，配合第 8.1/8.2 的量化指标构成完整的调优闭环。已有 LangSmith 的同学也可以直接复用（LlamaIndex 兼容 OpenTelemetry 生态）。

**第 8 站练习**：① 给自己的语料生成 QA 数据集，跑通 hit_rate/MRR 基线；② 依次做三个实验并记录分数：chunk 256 vs 1024、加 reranker vs 不加、朴素检索 vs 句子窗口，形成自己的调优结论表；③ 挑 10 个问题的回答做忠实度评估，找出失败样本并归因。

---

# 第 9 站：生产化要点

**增量摄入与去重**：文档会更新，别每次全量重建。方案：IngestionPipeline 接 docstore + cache（第 2.4 节）——按内容哈希判断新增/变更/删除，只处理增量；定时任务（cron）触发 pipeline，索引常驻。

**Embedding 成本控制**：缓存命中是第一手段；其次选对模型（text-embedding-3-small 这类性价比款够用）；最后才是换向量库。换 embedding 模型 = 全量重建索引（向量空间不兼容）。

**并发与异步**：QueryEngine 有同步和异步两套 API（`query` / `aquery`，ChatEngine 同理），Web 服务用异步版；Workflows 和 Agent 本身 async-first。高并发下注意 LLM 供应商的 RPM 限额。

**部署形态**：最简——FastAPI 包一层 query_engine（与包 LangGraph 的方式完全一致，参考 LangGraph 指南 4.7 节）；进阶——索引构建与查询服务分离（离线 pipeline 写入向量库，在线服务只读）；托管——直接用 LlamaCloud Index 免去自建。

**安全**：文档级权限要在元数据过滤层实现（第 4.4 节），不要指望 prompt 约束；外部数据源 Reader 的凭证走环境变量。

---

# 附录 A：学习资源

- 官方文档主站（含 Python 框架、LlamaCloud、LlamaAgents）：developers.llamaindex.ai（可给任意页面 URL 加 `index.md` 后缀拿原始 Markdown，还有一个官方文档 MCP 服务器）
- LlamaHub 集成市场：llamahub.ai
- LlamaCloud 控制台：cloud.llamaindex.ai
- 官方示例仓库：github.com/run-llama/llama_index 下的 examples 目录（按场景分类，比文档更贴近实战）
- Llama Packs：LlamaHub 上的预制应用模板（一个 zip 就是一套完整可跑的参考实现，如高级 RAG 套件），适合学完本指南后"抄成熟作业"。注意 2024 年后官方重心转向 Workflows/Agents，部分老 Pack 的 API 已过时，参考其思路为主、代码需对照新 API 核对

# 附录 B：新手常见坑速查

1. **`ModuleNotFoundError: llama_index.llms.openai`** → 只装了 core，补装 `llama-index-llms-openai`（其他集成同理，缺啥装 `llama-index-<类别>-<名字>`）
2. **`OpenAIAPIKeyNotSetError` 或 401** → 没设 `OPENAI_API_KEY` 环境变量；用兼容端点时还要设 base_url
3. **回答和文档无关** → 先查 source_nodes 看检索命中（no_text 模式裸看召回），九成是检索问题不是 LLM 问题
4. **换了 embedding 模型后检索全乱** → 向量空间不兼容，必须全量重建索引
5. **Jupyter 里跑 async 报 `RuntimeError: notebook cannot asyncio.run`** → Notebook 环境直接 `await agent.run(...)`，或 `import nest_asyncio; nest_asyncio.apply()`
6. **chunk_overlap 设太大导致块几乎全重复** → overlap 一般为 chunk_size 的 5%-15%
7. **老教程代码跑不通**（`from llama_index import GPTSimpleVectorIndex` 之类）→ 那是 0.9 及更早的 API；现在统一是 `from llama_index.core import VectorStoreIndex`
8. **Agent 老写法报错**（`from llama_index.agent.openai import OpenAIAgent`）→ 已被 `llama_index.core.agent.workflow` 的 FunctionAgent/AgentWorkflow 取代
