# Chapter 02 学习笔记：LCEL、链、结构化输出与面试高频解析

## 一、 链（Chain）与 LCEL

**链（Chain）**：把 Prompt、Model、OutputParser 等组件按顺序连接起来，前一个组件的输出自动作为后一个组件的输入，对外统一暴露 `invoke / stream / batch / ainvoke` 等标准调用接口。

**LCEL（LangChain Expression Language）**：用管道符 `|` 组合各个 Runnable 组件来声明式地构建链，是当前**唯一推荐**的组链方式。

> [!IMPORTANT]
> 旧版的 `LLMChain`、`SequentialChain` 等 Chain 类**已全部废弃**，现代 LangChain 开发一律使用 LCEL 管道写法。

### 简单链示例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("user", "{question}"),
])

# 最基本的链：提示词模板 | 模型 | 字符串输出解析器
chain = prompt | model | StrOutputParser()

result = chain.invoke({"question": "什么是装饰器？"})
```

### LCEL 核心优势

| 优势           | 说明                                                                          |
| :------------- | :---------------------------------------------------------------------------- |
| **统一接口**   | 所有组件都实现 `Runnable` 协议，`.invoke()`、`.stream()`、`.batch()` 开箱即用 |
| **自动流式**   | 链条自动支持流式输出，中间组件透明传递 Token 流                               |
| **异步/并发**  | 原生支持 `ainvoke`、`abatch`，无需额外改造                                    |
| **自由组合**   | 链本身也是 Runnable，可嵌套进其他链（链中套链）                               |
| **可观测性**   | 全链路自动接入 LangSmith 追踪                                                 |

---

## 二、 复合链（链的组合）

复合链 = 把多条子链/节点组合成一条更大的链，常见四种组合方式：

### 1. 串行复合（链中套链）

上一条链的输出直接作为下一条链的输入：

```python
# 摘要链的输出（字符串）直接喂给标题链
title_chain = summary_chain | title_prompt | model | StrOutputParser()
```

### 2. 并行复合 —— RunnableParallel

同一份输入同时分流到多条子链并发执行，结果汇总为一个 dict：

```python
from langchain_core.runnables import RunnableParallel

analysis_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain,
)
# 简写：直接写 dict 字面量，LCEL 会自动转成 RunnableParallel
# analysis_chain = {"summary": summary_chain, "keywords": keywords_chain}
```

> [!NOTE]
> 并发机制：同步 `invoke` 底层用**线程池**（`ThreadPoolExecutor`）并发各分支；异步 `ainvoke` 用 `asyncio.gather` 真正协程并发。

### 3. 透传与追加 —— RunnablePassthrough.assign

保留上游的原始输入，同时追加新计算出的字段（常用于保留中间结果）：

```python
from langchain_core.runnables import RunnablePassthrough

full_chain = {
    "raw_news": news_chain,  # 第一步：生成的新闻正文保留下来
} | RunnablePassthrough.assign(
    summary=lambda x: summary_chain.invoke({"news": x["raw_news"]})  # 第二步：追加提取结果
)
# 最终 result 同时包含 raw_news 和 summary 两个字段
```

> [!WARNING]
> `assign` 中用 lambda 直接调用 `.invoke()` 是**同步阻塞**写法，会绕过 LCEL 的流式/异步传播机制。推荐改为 Runnable 组合写法：
> ```python
> summary=RunnableLambda(lambda x: {"news": x["raw_news"]}) | summary_chain
> ```

### 4. 条件路由 —— RunnableBranch

根据输入内容动态选择执行不同的子链（类似 if-elif-else）：

```python
from langchain_core.runnables import RunnableBranch

chain = RunnableBranch(
    (lambda x: "代码" in x["question"], code_chain),      # 条件1 → 走代码链
    (lambda x: "翻译" in x["question"], translate_chain),  # 条件2 → 走翻译链
    default_chain,  # 兜底 → 通用链
)
```

> [!NOTE]
> 判断条件从上到下依次匹配，命中第一个为 `True` 的分支后立即执行，不再往下判断。最后一个参数是**必填的兜底链**。

---

## 三、 自定义节点

把任意普通 Python 逻辑接入 LCEL 链，有两种方式：

| 方式                          | 适用场景                                     | 写法                            |
| :---------------------------- | :------------------------------------------- | :------------------------------ |
| **`RunnableLambda`**（常用）  | 简单的一次性函数（一个输入 → 一个输出即可）  | `RunnableLambda(my_func)`       |
| **继承 `Runnable`**（进阶）   | 需要复用、带参数、逻辑复杂的组件             | 实现 `invoke(self, input, ...)` |

```python
from langchain_core.runnables import Runnable, RunnableLambda

# 方式一：RunnableLambda 包装普通函数
def clean_text(text: str) -> str:
    return " ".join(text.split())

chain = RunnableLambda(clean_text) | prompt | model

# 方式一的语法糖：@chain 装饰器（等价于 RunnableLambda 包装）
from langchain_core.runnables import chain as chain_decorator

@chain_decorator
def clean_and_count(text: str) -> dict:
    cleaned = " ".join(text.split())
    return {"text": cleaned, "length": len(cleaned)}
# clean_and_count 现在直接就是 Runnable，无需手动包装

# 方式二：继承 Runnable 写组件类
class TextStatsNode(Runnable):
    def invoke(self, input, config=None, **kwargs):
        return {"char_count": len(input)}
```

> [!TIP]
> 调试技巧：使用 `chain.get_graph().print_ascii()` 可以可视化整条链的结构（需 `pip install grandalf`）。

---

## 四、 输出处理（OutputParser 与结构化输出）

处理大模型输出不只有 `with_structured_output` 一种方式，完整的输出处理体系分为三个层次：

### 第一层：OutputParser —— 基础文本解析

`langchain_core.output_parsers` 中内置了多种解析器，把模型返回的 `AIMessage` 转成目标格式，**不依赖 Tool Calling，任何模型都能用**：

| 解析器                             | 作用                            | 典型场景                        |
| :--------------------------------- | :------------------------------ | :------------------------------ |
| **`StrOutputParser`**              | 提取 `.content` 纯文本          | 最常用，几乎所有链都会挂        |
| `JsonOutputParser`                 | 从文本中提取 JSON 并解析        | 模型输出包含 JSON 片段时        |
| `CommaSeparatedListOutputParser`   | 解析逗号分隔文本为 `list[str]`  | "给我 5 个关键词"类场景         |
| `MarkdownListOutputParser`         | 解析 Markdown 列表为 `list[str]`| 模型输出列表内容时              |
| `XMLOutputParser`                  | 解析 XML 格式输出               | Claude 等偏好 XML 的模型        |

### 第二层：`with_structured_output` —— 结构化输出（推荐）

当需要模型返回**严格符合 schema 的结构化对象**时，使用 `with_structured_output`：

| 方式                                                       | 状态            | 原理与评价                                                                         |
| :--------------------------------------------------------- | :-------------- | :--------------------------------------------------------------------------------- |
| **`model.with_structured_output(Pydantic类)`**             | ✅ **当前推荐** | 利用模型原生的 Tool Calling / JSON Schema 能力约束输出，带类型校验，稳定可靠       |
| `PydanticOutputParser`                                     | ⚠️ 旧版兜底     | 靠提示词注入 `format_instructions` "求"模型遵守格式，再用 `json.loads` + Pydantic 解析。模型一旦输出多余文字就会解析报错，仅在模型不支持 Tool Calling 时兜底 |
| `StructuredOutputParser` + `ResponseSchema`                | ⚠️ 旧版兜底     | 同为提示词约定式，但连 Pydantic 校验都没有，只返回 `dict`，已不推荐                |

> [!IMPORTANT]
> **核心区别**：`with_structured_output` 在 **API 层面**强约束输出结构（模型只能返回符合 schema 的 JSON）；而 `PydanticOutputParser` 只是在提示词里"拜托"模型遵守格式，约束力弱，生产环境不可靠。

`with_structured_output` 底层支持两种策略，可通过 `method` 参数切换：

| method 值                    | 底层机制                   | 适用场景                             |
| :--------------------------- | :------------------------- | :----------------------------------- |
| `"function_calling"`（默认） | 利用模型 Tool Calling 能力 | 主流大模型（GPT-4、Claude、Qwen 等） |
| `"json_mode"`                | 利用模型 JSON Mode         | Tool Calling 不可用时的备选          |

### 第三层：自定义后处理 —— 最灵活

用 `RunnableLambda` 或 `@chain` 编写任意 Python 后处理逻辑，完全自由：

```python
import re

# 例：从模型输出中提取代码块
def extract_code(output: str) -> str:
    match = re.search(r"```python\n(.*?)```", output, re.DOTALL)
    return match.group(1) if match else output

chain = prompt | model | StrOutputParser() | RunnableLambda(extract_code)
```

### 推荐写法示例（`with_structured_output`）

```python
from pydantic import BaseModel, Field

# 1. 用 Pydantic 定义想要的结构化数据类型
class NewsSummary(BaseModel):
    time: str = Field(description="事件发生的时间")
    location: str = Field(description="事件发生的地点")
    event: str = Field(description="发生的具体事件")

# 2. 绑定到模型，正常情况下返回 NewsSummary 对象（生产环境建议 try/except 捕获 ValidationError）
structured_model = model.with_structured_output(NewsSummary)

# 3. 像普通链一样使用，输出直接是 Pydantic 实例
extract_chain = extract_prompt | structured_model
result = extract_chain.invoke({"news": "..."})  # -> NewsSummary(time=..., location=..., event=...)
```

### 如何选择？

| 需求                             | 推荐方式                          |
| :------------------------------- | :-------------------------------- |
| 只要纯文本                       | `StrOutputParser`                 |
| 需要 JSON / 列表                 | `JsonOutputParser` / `ListOutputParser` |
| 需要 Pydantic 强类型对象         | `with_structured_output` ✅       |
| 模型不支持 Tool Calling 的兜底   | `PydanticOutputParser`            |
| 自定义提取逻辑（正则、拆分等）   | `RunnableLambda` 自己写           |

---

## 五、 面试高频问题与满分解答（链与结构化输出篇）

### Q1: 什么是 LCEL？相比旧版的 `LLMChain` 有什么优势？

**答题切入点：Runnable 协议、组合性、统一调用范式**

1. **LCEL 是一套声明式的链式组合语法**：所有组件（Prompt、Model、Parser、自定义函数）都实现统一的 `Runnable` 协议，通过管道符 `|` 组合成链。
2. **统一的调用接口**：任何 LCEL 链都原生支持 `invoke / stream / batch / ainvoke`，流式输出和异步并发开箱即用；旧版 `LLMChain` 各接口割裂，流式支持差。
3. **强大的组合性**：链本身也是 Runnable，可以自由嵌套（链中套链）、并行（`RunnableParallel`）、透传（`RunnablePassthrough`），旧版 Chain 类的组合能力非常有限。
4. **工程化配套**：LCEL 链自动接入 LangSmith 全链路追踪，且社区生态（LangGraph 等）全部建立在 Runnable 协议之上。旧版 Chain 类已被官方废弃。

---

### Q2: 如何在一条链中同时保留中间结果和最终结果？

**答题切入点：RunnablePassthrough.assign、RunnableParallel**

- 核心思路：LCEL 管道默认每个节点只把**自己的输出**传给下一个节点，中间结果会丢失。解决办法有两类：
  1. **`RunnablePassthrough.assign(新字段=子链)`**：原样透传上游输入的所有字段，同时追加新字段。多次串联 `assign` 即可逐步累积中间结果。
  2. **`RunnableParallel`（或 dict 字面量简写）**：让同一份输入分流到多条子链，每个分支的结果以 key 的形式保留在输出的 dict 中。
- 实际项目中两者常配合使用：先 dict 并行保留原始产物，再 `assign` 追加下游加工结果。

---

### Q3: LangChain 结构化输出有哪几种方式？为什么现在推荐 `with_structured_output`？

**答题切入点：原生 Tool Calling vs 提示词约定、解析稳定性、类型校验**

1. **两种方式的本质区别**：
   - 旧版 `StructuredOutputParser` / `PydanticOutputParser`：把格式说明（`format_instructions`）写进提示词，**靠"求"模型遵守格式**，再用正则/JSON 解析。模型一旦输出多余文字或格式漂移就会解析报错。
   - 新版 `with_structured_output(Pydantic类)`：利用模型厂商**原生的 Tool Calling / JSON Schema 能力**，在 API 层面强约束输出结构。
2. **推荐 `with_structured_output` 的理由**：
   - 输出直接是 **Pydantic 实例**，自带类型校验和 IDE 类型提示；
   - 不依赖提示词约定，**解析稳定性高**，生产环境可靠；
   - 代码更简洁，无需手动注入 `format_instructions`。
3. 补充：旧版解析器方式只需了解原理，在模型不支持 Tool Calling 的极端场景下才作为兜底。
4. **加分项**：`with_structured_output` 支持 `include_raw=True` 参数，可同时返回原始模型输出和解析后的结构化对象（`{"raw": AIMessage(...), "parsed": Pydantic实例, "parsing_error": None}`），方便调试排查。

---

### Q4: 如何把一个普通的 Python 函数接入 LCEL 链？两种方式如何取舍？

**答题切入点：RunnableLambda、继承 Runnable**

1. **`RunnableLambda`（首选）**：只要函数满足"一个输入 → 一个输出"，用 `RunnableLambda(my_func)` 包装即可成为链上的一环，适合数据清洗、格式转换、结果后处理等简单逻辑。
2. **继承 `Runnable` 类**：实现 `invoke(self, input, config=None, **kwargs)` 方法，适合需要**复用、携带构造参数、或内部状态/逻辑较复杂**的自定义组件。
3. **取舍原则**：一次性、无状态的简单函数用 `RunnableLambda`；需要面向对象封装、多处复用时才继承 `Runnable`。两者在链中地位完全等价，可自由混用。

---

### Q5: LCEL 链的流式输出是如何工作的？哪些节点会“阻断”流式传播？

**答题切入点：stream 方法、流式传播链路、阻断场景**

1. **流式传播原理**：调用 `chain.stream(input)` 时，LCEL 从链尾回溯找到第一个能产出 Token 流的节点（通常是 LLM），逐个 Token 向下游传递。
2. **透传与阻断**：`StrOutputParser` 能逐 Token 透传流式输出；但 `JsonOutputParser` 等需要等完整 JSON 才能解析，会**阻断流式**传播。
3. **`RunnableLambda` 默认不支持流式**（整进整出），要支持流式需额外实现 `transform` 方法或使用生成器函数。
4. **并行分支的流式**：`RunnableParallel` 中各分支独立流式，`stream()` 输出的是合并后的增量 dict。
5. **实战建议**：如果对流式体验有要求，链中尽量避免需要“等完整输出”的节点；必须使用时可把它放在最后一环，前面的节点保持流式透传。
