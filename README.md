# Python Agent Lab 🤖

一个基于 Python 的 AI Agent 探索与实践实验室，专注于深入学习与实战 **LangChain**、**LangGraph**、**Multi-Agent 架构** 以及 **自主智能体 (Autonomous Agents)** 的核心技术。

---

## 📌 项目定位

本仓库用于记录与沉淀 AI Agent 开发相关的实验、示例代码、设计模式以及最佳实践：

- **LangChain 生态**：Prompt 模板、Chain 构造、Tools 定义与 LLM 整合。
- **LangGraph 工作流**：基于状态图 (Stateful Graph) 的复杂 Agent 循环与条件分支控制流程。
- **Multi-Agent 协作**：多智能体分工合作、角色扮演与任务编排模式。
- **Agent 工具与生态扩展**：自定义 Tool / MCP 工具接入、向量检索与 RAG 结合。

---

## 📁 目录规划

```text
python-agent-lab/
├── docs/                   # 相关架构设计文档与学习笔记
├── src/                    # 核心代码与实验模块
│   ├── basic_chains/       # 基础 Chain 与 Prompt 实验室
│   ├── langgraph_agents/   # LangGraph 状态图与智能体工作流
│   ├── multi_agent/        # 多智能体协作系统
│   └── tools/              # 自定义工具与 API 扩展
├── tests/                  # 单元测试与集成测试
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/heiye-vn/python-agent-lab.git
cd python-agent-lab
```

### 2. 创建并激活虚拟环境

```bash
# 使用 venv 创建虚拟环境
python -m venv .venv

# Windows PowerShell 激活
.\.venv\Scripts\Activate.ps1

# Linux / macOS 激活
source .venv/bin/activate
```

### 3. 配置环境变量

复制示例配置文件并填入 API Key：

```bash
cp .env.example .env
```

---

## 🛠️ 技术栈

- **语言**: Python 3.10+
- **核心框架**: LangChain, LangGraph
- **模型集成**: OpenAI / Anthropic / Google Gemini / Ollama
- **工具支持**: Pydantic, FAISS / Chroma

---

## 📄 开源协议

[MIT License](LICENSE)
