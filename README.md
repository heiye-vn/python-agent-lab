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

## 📁 目录结构

```text
python-agent-lab/
├── code/                   # 各章节学习代码（含示例与笔记）
│   ├── chapter01/          # LLM 客户端创建方式
│   ├── chapter02/          # 链与 LCEL
│   ├── chapter03/          # 对话系统（单轮/多轮/流式/Gradio）
│   ├── chapter04/          # 工具调用 Tool
│   ├── chapter05/          # 浏览器自动化（Playwright）
│   ├── chapter06/          # MCP 协议
│   ├── chapter07/          # PDF RAG 系统（FAISS）
│   ├── chapter08/          # 数据分析 Agent（Streamlit）
│   ├── chapter09/          # LangGraph 状态图
│   └── chapter10/          # LangGraph 多工具调用
├── learning_docs/          # 框架学习指南（LlamaIndex 等）
├── interview/              # 面试知识点速查（Agent 评估体系等）
├── CHAPTERS.md             # 章节主题与关键内容索引
├── .gitignore
├── README.md
└── requirements.txt
```

> 各章节的学习主题、关键知识点与代表文件说明，详见 [CHAPTERS.md](./CHAPTERS.md)。

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
.venv\Scripts\Activate.ps1

# Linux / macOS 激活
source .venv/bin/activate
```

### 3. 配置环境变量

进入要运行的章节目录（如 `code/chapter01/`），复制示例配置文件并填入 API Key：

```bash
cd code/chapter01
cp .env.example .env
```

---

## 🛠️ 技术栈

- **语言**: Python 3.10+
- **核心框架**: LangChain, LangGraph
- **模型集成**: OpenAI / Anthropic / Google Gemini / Ollama
- **工具支持**: Pydantic, FAISS / Chroma
