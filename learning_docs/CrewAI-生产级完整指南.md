# CrewAI 生产级完整指南

> 基于掘金文章深度解析 + 官方文档 + 生产实践经验整理
> 适用版本：CrewAI >= 0.80（2025-2026）

---

## 一、CrewAI 是什么

CrewAI 是一个基于 Python 的开源多智能体协作框架。它的核心设计哲学是**"像管理人类团队一样管理 AI Agent"**——通过定义角色（Role）、目标（Goal）和背景故事（Backstory）来赋予每个 Agent 明确的身份，再让它们围绕具体任务协同工作。

### 1.1 与其他框架的定位对比

| 维度 | CrewAI | LangGraph | AutoGen | MetaGPT |
|------|--------|-----------|---------|---------|
| **核心范式** | 任务驱动 + 角色化分工 | 图论 + 状态机 | 对话驱动 | 软件开发模拟 |
| **学习曲线** | 低（声明式 API） | 中高（需理解图概念） | 中 | 中 |
| **灵活度** | 中等（牺牲部分灵活度换开发体验） | 极高 | 高 | 较低（场景专用） |
| **适用场景** | 流水线作业、内容创作、数据分析 | 复杂状态流转、需要精确控制 | 多轮对话、研究探索 | 软件工程自动化 |
| **生产就绪度** | 高（内置缓存/记忆/护栏） | 高 | 中 | 中 |
| **调试体验** | 好（日志清晰、角色可追溯） | 中（图调试较复杂） | 中 | 中 |

**核心结论**：如果你的场景是"多个 Agent 各司其职、按流程完成一项工作"，CrewAI 是开发效率最高的选择。如果需要精确控制每一跳的条件分支，LangGraph 更合适。

### 1.2 典型应用场景

- **内容创作流水线**：研究员 → 撰稿人 → 编辑 → SEO 专家
- **智能客服系统**：意图识别 → 路由分发 → 专属 Crew 处理 → 满意度评估
- **数据分析团队**：数据采集 → 深度分析 → 商业洞察 → 报告生成
- **代码审查**：安全审查 → 性能审查 → 风格审查 → 汇总报告
- **市场调研**：竞品分析 → 用户画像 → 趋势预测 → 策略建议

---

## 二、环境搭建与项目初始化

### 2.1 系统要求

- Python 3.10 ~ 3.13
- 推荐使用 `uv` 包管理器（CrewAI 官方推荐）

### 2.2 安装

```bash
# 1. 安装 uv（macOS/Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 安装 CrewAI CLI
uv tool install crewai

# 3. 创建项目
crewai create crew my_project
# 或传统 YAML 模式
crewai create crew my_project --classic

# 4. 安装依赖
cd my_project
crewai install
```

### 2.3 项目结构（JSON 模式）

```
my_project/
├── crew.jsonc              # 主配置：团队、流程、默认参数
├── agents/
│   ├── researcher.jsonc    # 研究员 Agent 配置
│   ├── writer.jsonc        # 撰稿人 Agent 配置
│   └── editor.jsonc        # 编辑 Agent 配置
├── tools/
│   └── custom_tool.py      # 自定义工具
├── knowledge/              # 知识库文件（可选）
├── .env                    # API 密钥等环境变量
└── pyproject.toml          # Python 项目依赖
```

### 2.4 环境变量配置（.env）

```bash
# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL_NAME=gpt-4o

# Azure OpenAI
AZURE_API_KEY=xxx
AZURE_API_BASE=https://xxx.openai.azure.com
AZURE_API_VERSION=2024-02-15-preview

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# 本地模型（Ollama）
# 无需 API Key，只需 Ollama 服务运行中

# 国内模型（如阿里百炼）
ALI_BAILIAN_API_KEY=sk-xxx
ALI_BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 2.5 混合模型策略（生产环境成本控制）

```python
# 核心思路：复杂任务用强模型，简单任务用轻量模型
researcher = Agent(
    role="高级研究员",
    llm="gpt-4o",          # 深度分析用强模型
    ...
)

formatter = Agent(
    role="格式整理员",
    llm="gpt-4o-mini",     # 简单格式化用轻量模型
    ...
)
```

---

## 三、核心组件详解

### 3.1 Agent（智能体）

Agent 是团队中的"员工"，每个 Agent 有明确的身份定义。

#### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `role` | str | 角色名称，如"资深研究员" |
| `goal` | str | 目标，定义 Agent 追求什么 |
| `backstory` | str | 背景故事，提供上下文和专业知识 |
| `llm` | str | 使用的语言模型 |
| `tools` | List[Tool] | 可用工具列表 |
| `allow_delegation` | bool | 是否允许将任务委托给其他 Agent |
| `max_iter` | int | 最大迭代次数（防止死循环） |
| `max_rpm` | int | 每分钟最大请求数（防止 API 限流） |
| `verbose` | bool | 是否输出详细日志 |
| `memory` | bool | 是否启用记忆 |
| `function_calling_llm` | str | 专门用于函数调用的模型（可选） |

#### 代码示例

```python
from crewai import Agent

researcher = Agent(
    role="AI 行业资深研究员",
    goal="深入分析 AI Agent 领域的最新趋势和技术突破",
    backstory="""你是一位在 AI 领域深耕 15 年的研究员，
    曾在顶级实验室工作，发表过多篇关于多智能体系统的论文。
    你擅长从海量信息中提炼核心洞察，并用通俗语言解释复杂概念。""",
    llm="gpt-4o",
    tools=[search_tool, scrape_tool],
    allow_delegation=False,
    max_iter=5,
    max_rpm=30,
    verbose=True,
    memory=True,
)
```

#### YAML 配置方式（crew.jsonc / agents/*.jsonc）

```jsonc
{
  "role": "AI 行业资深研究员",
  "goal": "深入分析 AI Agent 领域的最新趋势",
  "backstory": "你是一位在 AI 领域深耕 15 年的研究员...",
  "llm": "gpt-4o",
  "tools": ["search_tool", "scrape_tool"],
  "allow_delegation": false,
  "max_iter": 5
}
```

#### 最佳实践：角色设计原则

1. **具体且具区分度**：避免"全能助手"，每个 Agent 应有明确边界
2. **背景故事要丰富**：backstory 越详细，LLM 的行为越稳定
3. **避免功能重叠**：两个 Agent 的职责不应有歧义
4. **渐进式开发**：从 2 个 Agent 开始，验证通过后再增加

### 3.2 Task（任务）

Task 是分配给 Agent 的具体工作单元。

#### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `description` | str | 任务描述，必须清晰可衡量 |
| `expected_output` | str | 预期输出格式 |
| `agent` | Agent | 执行该任务的 Agent |
| `context` | List[Task] | 上下文依赖的其他任务 |
| `output_json` | Pydantic Model | 输出为 JSON 格式（可选） |
| `output_pydantic` | Pydantic Model | 输出为 Pydantic 模型（可选） |
| `async_execution` | bool | 是否异步执行 |
| `callback` | Callable | 完成后的回调函数 |
| `guardrail` | Callable | 输出质量验证函数 |

#### 代码示例

```python
from crewai import Task
from pydantic import BaseModel

class ArticleOutput(BaseModel):
    title: str
    content: str
    keywords: list[str]
    word_count: int

research_task = Task(
    description="""
    研究 2025-2026 年 AI Agent 框架的发展趋势。
    
    重点关注：
    1. 主流框架的市场份额变化
    2. 企业级部署的关键挑战
    3. 多智能体协作的最新突破
    
    输出要求：结构化报告，包含数据支撑和来源引用。
    """,
    expected_output="一份 1500 字以上的结构化研究报告，包含至少 3 个数据图表描述",
    agent=researcher,
    output_pydantic=ArticleOutput,  # 强制输出格式
)
```

#### Guardrail 护栏机制（生产环境必备）

```python
def validate_article(output):
    """验证文章质量"""
    text = output.raw
    issues = []
    
    if len(text) < 1000:
        issues.append("文章字数不足 1000 字")
    if "参考文献" not in text and "来源" not in text:
        issues.append("缺少来源引用")
    if text.count("。") < 10:
        issues.append("内容过于简略")
    
    if issues:
        return (False, "请修改以下问题：" + "；".join(issues))
    return (True, output)

quality_task = Task(
    description="审核文章质量",
    expected_output="审核通过的文章",
    agent=editor,
    guardrail=validate_article,  # 不通过会自动打回重做
)
```

### 3.3 Crew（团队）

Crew 是容器，负责组装 Agent 和 Task，并控制执行策略。

```python
from crewai import Crew

content_crew = Crew(
    agents=[researcher, writer, editor, seo_specialist],
    tasks=[research_task, write_task, edit_task, seo_task],
    process=Process.sequential,      # 顺序执行
    memory=True,                      # 启用记忆
    cache=True,                       # 启用缓存
    max_rpm=30,                       # 全局限流
    verbose=True,                     # 详细日志
    # planning=True                   # 启用规划模式（实验性）
)

# 执行
result = content_crew.kickoff(inputs={"topic": "AI Agent 2026 趋势"})
```

#### 批量执行

```python
# 对多个输入分别执行同一个 Crew 流程
topics = ["AI Agent 趋势", "RAG 技术演进", "MCP 协议解析"]
results = content_crew.kickoff_for_each(inputs_list=topics)
```

#### 训练模式（迭代提升质量）

```python
# 通过反馈迭代训练 Crew
content_crew.train(
    n_iterations=3,
    inputs={"topic": "AI Agent 趋势"},
    feedback="请增加更多实际案例，减少理论描述"
)
```

### 3.4 Process（执行流程）

#### Sequential（顺序执行）

```python
# 默认模式，像流水线：A 的输出自动成为 B 的输入
crew = Crew(process=Process.sequential)
```

适用场景：内容创作、数据处理、报告生成等有明确先后顺序的任务。

#### Hierarchical（层级执行）

```python
# 由管理者 Agent 动态分配任务
crew = Crew(
    process=Process.hierarchical,
    manager_agent=manager,  # 管理者 Agent
)
```

适用场景：任务复杂度高、需要动态决策的场景。管理者会根据任务内容智能分配给最合适的 Agent。

#### Consensual（协商执行）

```python
# 多专家协商达成共识
crew = Crew(
    process=Process.consensual,
)
```

适用场景：需要多视角讨论的复杂决策，如架构设计评审、风险评估。

### 3.5 Tools（工具）

工具赋予 Agent 与外部世界交互的能力。

#### 内置工具

```python
from crewai.tools import (
    SerperDevTool,        # 网络搜索
    ScrapeWebsiteTool,    # 网页抓取
    DirectoryReadTool,    # 目录读取
    FileReadTool,         # 文件读取
    WebsiteSearchTool,    # 网站搜索
)
```

#### 自定义工具（装饰器方式）

```python
from crewai.tools import tool

@tool("计算器")
def calculator(expression: str) -> str:
    """计算数学表达式并返回结果"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"
```

#### 自定义工具（类方式，适合复杂工具）

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="SQL 查询语句")
    database: str = Field(description="目标数据库名称")

class DatabaseTool(BaseTool):
    name: str = "数据库查询工具"
    description: str = "执行 SQL 查询并返回结果"
    args_schema: type[BaseModel] = DatabaseQueryInput
    
    def _run(self, query: str, database: str) -> str:
        # 实际数据库连接逻辑
        return f"查询 {database} 的结果: ..."
```

#### 工具缓存机制

```python
# CrewAI 内置工具缓存，避免重复调用
@tool("API 调用工具", cache=True)  # 默认开启
def api_call(endpoint: str) -> str:
    ...
```

### 3.6 Memory（记忆系统）

CrewAI 的记忆系统是其核心竞争力之一，让 Agent 在多轮交互和跨任务场景中保持上下文。

#### 记忆类型

| 类型 | 作用域 | 持久化 | 说明 |
|------|--------|--------|------|
| **短期记忆** (Short-term) | 单次 Crew 执行 | 否 | 当前任务的上下文，执行结束即清除 |
| **长期记忆** (Long-term) | 跨 Crew 执行 | 是 | 累积的经验和知识，跨会话保留 |
| **实体记忆** (Entity) | 跨 Crew 执行 | 是 | 关于特定实体（人、公司、项目）的记忆 |
| **用户记忆** (User) | 跨会话 | 是 | 用户偏好、习惯等个性化信息 |

#### 启用记忆

```python
from crewai import Crew
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,  # 启用全部记忆类型
    # 或精细控制：
    # short_term_memory=ShortTermMemory(),
    # long_term_memory=LongTermMemory(),
    # entity_memory=EntityMemory(),
)
```

#### 记忆的作用域隔离

```python
# 不同 Crew 之间的记忆默认隔离
crew_a = Crew(memory=True, ...)  # Crew A 的记忆
crew_b = Crew(memory=True, ...)  # Crew B 的记忆

# 如需共享，使用 Embedding 配置统一向量存储
```

#### 生产环境记忆配置建议

- **内容创作**：启用短期记忆即可（单次会话内上下文连贯）
- **客服系统**：启用长期记忆 + 实体记忆（记住客户历史交互）
- **数据分析**：启用长期记忆（积累领域知识）
- **个人助手**：启用用户记忆（记住用户偏好）

### 3.7 Flow（工作流编排）

Flow 是 CrewAI 的高级编排层，用于构建复杂的、事件驱动的工作流。

#### 核心概念

```python
from crewai.flow.flow import Flow, start, listen, router

class ContentWorkflow(Flow):
    
    @start()
    def begin(self):
        """工作流入口"""
        self.state["topic"] = "AI Agent 趋势"
        return self.state
    
    @listen(begin)
    def research(self, state):
        """研究阶段"""
        crew = ResearchCrew().crew()
        result = crew.kickoff(inputs={"topic": state["topic"]})
        state["research"] = result.raw
        return state
    
    @listen(research)
    def write(self, state):
        """写作阶段"""
        crew = WritingCrew().crew()
        result = crew.kickoff(inputs={"research": state["research"]})
        state["article"] = result.raw
        return state
    
    @router(write)
    def quality_check(self, state):
        """质量路由"""
        if len(state["article"]) > 2000:
            return "publish"
        else:
            return "rewrite"
    
    @listen("publish")
    def publish(self, state):
        """发布"""
        print(f"发布文章: {state['article'][:100]}...")
    
    @listen("rewrite")
    def rewrite(self, state):
        """重写"""
        state["needs_improvement"] = True
        return self.write(state)  # 回到写作阶段
```

#### 状态持久化

```python
from crewai.flow.flow import Flow, start, listen

class PersistentWorkflow(Flow):
    # 状态自动持久化，支持断点续跑
    
    @start()
    def init(self):
        self.state["step"] = 1
        return self.state
    
    @listen(init)
    def process(self, state):
        state["step"] = 2
        # 即使中途崩溃，重启后从 step=2 继续
        return state
```

#### 条件路由

```python
class CustomerServiceWorkflow(Flow):
    
    @start()
    def identify_intent(self):
        """识别用户意图"""
        # 调用意图识别 Crew
        intent = classify_intent(self.state["user_message"])
        self.state["intent"] = intent
        return intent
    
    @router(identify_intent)
    def route_by_intent(self, intent):
        """根据意图路由"""
        if "退款" in intent or "投诉" in intent:
            return "complaint"
        elif "故障" in intent or "报错" in intent:
            return "tech_support"
        else:
            return "faq"
    
    @listen("faq")
    def handle_faq(self, state):
        """FAQ 处理"""
        ...
    
    @listen("tech_support")
    def handle_tech(self, state):
        """技术支持"""
        ...
    
    @listen("complaint")
    def handle_complaint(self, state):
        """投诉处理"""
        ...
```

---

## 四、实战案例

### 4.1 内容创作流水线（Sequential 模式）

```python
from crewai import Agent, Task, Crew, Process

# === Agent 定义 ===
researcher = Agent(
    role="资深行业研究员",
    goal="从海量信息中提炼核心洞察",
    backstory="你是一位拥有 15 年经验的行业分析师，擅长数据挖掘和趋势预判。",
    llm="gpt-4o",
    tools=[search_tool, scrape_tool],
    memory=True,
)

writer = Agent(
    role="专业撰稿人",
    goal="将复杂概念转化为通俗易懂的文章",
    backstory="你是一位获奖科技作者，文风生动，善于用类比解释技术概念。",
    llm="gpt-4o",
    memory=True,
)

editor = Agent(
    role="资深编辑",
    goal="确保文章事实准确、逻辑清晰、无语法错误",
    backstory="你是一位严谨的编辑，有 10 年科技媒体经验。",
    llm="gpt-4o-mini",  # 编辑用轻量模型即可
)

seo_specialist = Agent(
    role="SEO 优化专家",
    goal="优化文章标题和关键词，提升搜索排名",
    backstory="你是一位 SEO 专家，熟悉搜索引擎算法和用户搜索习惯。",
    llm="gpt-4o-mini",
)

# === Task 定义 ===
research_task = Task(
    description="研究 {topic} 的最新发展，整理关键数据和趋势。",
    expected_output="结构化研究报告，包含 3-5 个核心发现和数据来源。",
    agent=researcher,
)

write_task = Task(
    description="基于研究报告撰写一篇 2000 字的深度文章。",
    expected_output="完整的文章初稿，包含引言、正文、结论。",
    agent=writer,
    context=[research_task],  # 依赖研究结果
)

edit_task = Task(
    description="审核文章的事实准确性、逻辑连贯性和语言质量。",
    expected_output="修改后的终稿，附带修改说明。",
    agent=editor,
    context=[write_task],
    guardrail=lambda output: (True, output) if len(output.raw) > 1500 else (False, "文章太短"),
)

seo_task = Task(
    description="优化标题、提取关键词、生成摘要。输出 JSON 格式。",
    expected_output='{"title": "...", "keywords": [...], "summary": "..."}',
    agent=seo_specialist,
    context=[edit_task],
)

# === Crew 组装 ===
content_crew = Crew(
    agents=[researcher, writer, editor, seo_specialist],
    tasks=[research_task, write_task, edit_task, seo_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    verbose=True,
)

# === 执行 ===
result = content_crew.kickoff(inputs={"topic": "2026 年 AI Agent 市场格局"})
print(result)
```

### 4.2 智能客服系统（Flow 模式）

```python
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen, router

# === 意图识别 Crew ===
intent_agent = Agent(
    role="意图识别专家",
    goal="准确判断用户咨询的类别",
    backstory="你是客服系统的核心，负责理解用户需求并分类。",
    llm="gpt-4o-mini",
)

intent_task = Task(
    description="分析用户消息：{user_message}，判断意图类别。",
    expected_output="意图类别：faq / tech_support / complaint",
    agent=intent_agent,
)

intent_crew = Crew(
    agents=[intent_agent],
    tasks=[intent_task],
    process=Process.sequential,
)

# === 各分支 Crew ===
def make_faq_crew():
    agent = Agent(role="FAQ 专家", goal="快速准确回答常见问题", llm="gpt-4o-mini")
    task = Task(description="回答用户问题：{user_message}", expected_output="清晰的回答", agent=agent)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential)

def make_tech_crew():
    agent = Agent(role="技术支持工程师", goal="诊断并解决技术问题", llm="gpt-4o")
    task = Task(description="诊断问题：{user_message}", expected_output="解决方案", agent=agent)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential)

def make_complaint_crew():
    agent = Agent(role="客户关系经理", goal="妥善处理投诉，维护客户关系", llm="gpt-4o")
    task = Task(description="处理投诉：{user_message}", expected_output="安抚方案", agent=agent)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential)

# === Flow 编排 ===
class CustomerServiceFlow(Flow):
    
    @start()
    def classify(self):
        result = intent_crew.kickoff(inputs={"user_message": self.state["user_message"]})
        self.state["intent"] = result.raw.strip()
        return self.state["intent"]
    
    @router(classify)
    def route(self, intent):
        if "complaint" in intent:
            return "complaint"
        elif "tech" in intent:
            return "tech"
        return "faq"
    
    @listen("faq")
    def handle_faq(self, state):
        return make_faq_crew().kickoff(inputs={"user_message": state["user_message"]})
    
    @listen("tech")
    def handle_tech(self, state):
        return make_tech_crew().kickoff(inputs={"user_message": state["user_message"]})
    
    @listen("complaint")
    def handle_complaint(self, state):
        return make_complaint_crew().kickoff(inputs={"user_message": state["user_message"]})

# === 使用 ===
flow = CustomerServiceFlow()
response = flow.kickoff(inputs={"user_message": "我的订单三天了还没发货"})
```

### 4.3 数据分析团队（Hierarchical 模式）

```python
from crewai import Agent, Task, Crew, Process

# === Agent 定义 ===
data_collector = Agent(
    role="数据采集专家",
    goal="从多个数据源高效采集和清洗数据",
    backstory="你是数据工程师，精通 ETL 流程和数据质量管理。",
    llm="gpt-4o-mini",
    tools=[scrape_tool, file_read_tool],
)

analyst = Agent(
    role="数据分析师",
    goal="运用统计方法发现数据中的模式和异常",
    backstory="你是资深数据分析师，擅长用数据讲故事。",
    llm="gpt-4o",
)

insight_translator = Agent(
    role="商业洞察翻译官",
    goal="将技术发现转化为管理层可理解的业务建议",
    backstory="你是技术与商业之间的桥梁，擅长将复杂数据转化为 actionable insights。",
    llm="gpt-4o",
)

report_writer = Agent(
    role="报告撰写专家",
    goal="生成专业、美观的数据分析报告",
    backstory="你是报告设计专家，注重信息层次和可视化呈现。",
    llm="gpt-4o-mini",
)

manager = Agent(
    role="项目管理者",
    goal="协调团队高效完成数据分析项目",
    backstory="你是经验丰富的项目经理，擅长资源分配和质量把控。",
    llm="gpt-4o",
    allow_delegation=True,
)

# === Task 定义 ===
collect_task = Task(
    description="采集并清洗 {dataset_name} 相关数据。",
    expected_output="清洗后的数据集和采集报告。",
    agent=data_collector,
)

analyze_task = Task(
    description="对采集的数据进行深度分析，发现关键模式。",
    expected_output="分析报告，包含统计结果和可视化描述。",
    agent=analyst,
    context=[collect_task],
)

insight_task = Task(
    description="将分析结果转化为 3-5 条可执行的业务建议。",
    expected_output="商业洞察报告，每条建议附带数据支撑。",
    agent=insight_translator,
    context=[analyze_task],
)

report_task = Task(
    description="整合所有成果，生成最终报告。",
    expected_output="完整的分析报告，包含摘要、发现、建议和附录。",
    agent=report_writer,
    context=[insight_task],
)

# === Crew（层级模式）===
data_crew = Crew(
    agents=[data_collector, analyst, insight_translator, report_writer],
    tasks=[collect_task, analyze_task, insight_task, report_task],
    process=Process.hierarchical,
    manager_agent=manager,
    memory=True,
    verbose=True,
)

result = data_crew.kickoff(inputs={"dataset_name": "2026 Q1 电商销售数据"})
```

---

## 五、生产环境最佳实践

### 5.1 成本控制

| 策略 | 说明 | 预估节省 |
|------|------|----------|
| 混合模型 | 复杂任务用 gpt-4o，简单任务用 gpt-4o-mini | 40-60% |
| 启用缓存 | `cache=True`，避免重复 API 调用 | 20-30% |
| 精简 backstory | 过长的 backstory 消耗更多 token | 10-15% |
| 限制 max_iter | 防止 Agent 陷入无限循环 | 避免意外高消费 |
| 设置 max_rpm | 防止 API 限流导致重试 | 稳定性提升 |

### 5.2 质量保障

```python
# 1. Guardrail 护栏（必须）
def validate_output(output):
    """输出质量底线检查"""
    text = output.raw
    checks = [
        (len(text) > 500, "内容过短"),
        ("参考文献" in text or "来源" in text, "缺少引用"),
        (text.count("\n") > 5, "格式不规范"),
    ]
    failures = [msg for passed, msg in checks if not passed]
    if failures:
        return (False, "请修正：" + "；".join(failures))
    return (True, output)

# 2. 结构化输出（推荐）
from pydantic import BaseModel

class StructuredResult(BaseModel):
    summary: str
    key_points: list[str]
    confidence: float
    sources: list[str]

task = Task(
    description="...",
    output_pydantic=StructuredResult,  # 强制 JSON Schema
)

# 3. 多 Agent 交叉验证
review_task = Task(
    description="审核前一个任务的输出，检查事实错误和逻辑漏洞。",
    agent=reviewer,
    context=[main_task],
)
```

### 5.3 错误处理与容错

```python
# 1. 设置合理的 max_iter
agent = Agent(
    max_iter=5,  # 最多重试 5 次
    max_rpm=30,  # 每分钟最多 30 次请求
)

# 2. 异常捕获
try:
    result = crew.kickoff(inputs={...})
except Exception as e:
    logger.error(f"Crew 执行失败: {e}")
    # 降级策略：使用缓存结果或返回默认值

# 3. 超时控制
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Crew 执行超时")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 分钟超时

try:
    result = crew.kickoff(inputs={...})
finally:
    signal.alarm(0)
```

### 5.4 日志与监控

```python
import logging

# 配置 CrewAI 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crewai.log'),
        logging.StreamHandler()
    ]
)

# verbose=True 会输出详细的 Agent 思考过程
crew = Crew(verbose=True, ...)
```

### 5.5 部署架构建议

```
┌─────────────────────────────────────────┐
│              API Gateway                │
├─────────────────────────────────────────┤
│           FastAPI / Flask               │
├─────────────────────────────────────────┤
│         CrewAI Service Layer            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Crew A  │ │ Crew B  │ │ Flow C  │  │
│  └─────────┘ └─────────┘ └─────────┘  │
├─────────────────────────────────────────┤
│    Redis (缓存)  │  PostgreSQL (记忆)   │
├─────────────────────────────────────────┤
│    LLM API (OpenAI / Azure / 本地)      │
└─────────────────────────────────────────┘
```

---

## 六、常见问题排错

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Agent 陷入循环/超时 | 任务描述模糊或 max_iter 过高 | 优化 description，降低 max_iter |
| Token 消耗过大 | backstory 过长、未启用缓存 | 精简 backstory，开启 cache=True |
| 上下文传递失败 | context 参数未正确配置 | 检查 Task 的 context 列表 |
| API 限流 (429) | 请求频率过高 | 设置 max_rpm，增加请求间隔 |
| 输出格式不一致 | 未使用结构化输出 | 使用 output_pydantic 或 output_json |
| Agent 之间委托失败 | allow_delegation 未开启 | 设置 allow_delegation=True |
| 记忆未生效 | memory 参数未启用 | Crew 和 Agent 都设置 memory=True |
| Flow 路由不触发 | router 返回值不匹配 | 检查 router 函数返回的字符串 |

---

## 七、学习路径建议

### 阶段一：入门（1-2 天）

1. 安装环境，跑通官方 quickstart
2. 理解 Agent / Task / Crew 三个核心概念
3. 构建一个 2-Agent 顺序流程（如：研究员 + 撰稿人）

### 阶段二：进阶（3-5 天）

1. 学习自定义 Tool 的两种方式
2. 掌握 Memory 系统的四种类型
3. 实现 Guardrail 护栏机制
4. 尝试 Hierarchical 和 Consensual 流程

### 阶段三：生产（1-2 周）

1. 用 Flow 构建事件驱动的复杂工作流
2. 实现混合模型策略控制成本
3. 搭建完整的错误处理和监控体系
4. 部署为 API 服务（FastAPI + CrewAI）

### 阶段四：精通（持续）

1. 自定义 Embedding 和向量存储
2. 多 Crew 协作和跨 Flow 编排
3. 与 LangGraph 混合使用（各取所长）
4. 参与社区贡献，跟踪版本更新

---

## 八、参考资源

- [CrewAI 官方文档](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [掘金原文：CrewAI 全面指南](https://juejin.cn/post/7649256802330443827)
- [DeepLearning.AI 短课程：Multi AI Agent Systems](https://www.deeplearning.ai/short-courses/)
- [CrewAI 最佳实践：Agent 团队配置与任务流转](https://gitcode.csdn.net/6a0d022610ee7a33f273ce09.html)
- [从零开始用 CrewAI：完整实战指南](https://www.tinyash.com/blog/crewai-multi-agent-workflow-guide/)
- [CrewAI Flow 开发问题积累](https://m.blog.csdn.net/weixin_44399264/article/details/162871936)
