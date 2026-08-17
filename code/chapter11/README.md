# Chapter 11: LangGraph 智能体服务化与 LangSmith 全链路监控

本章节基于 **LangGraph** 与 **LangSmith**，演示如何构建一个具备工具调用能力的对话智能体（天气助手），并通过 **LangGraph CLI / LangGraph Studio** 进行本地服务化部署、可视化调试与全链路追踪监控。

---

## 目录

- [一、 项目文件结构](#一-项目文件结构)
- [二、 环境准备与依赖安装](#二-环境准备与依赖安装)
- [三、 环境变量配置（.env）](#三-环境变量配置env)
- [四、 项目启动与运行](#四-项目启动与运行)
- [五、 LangSmith 监控与使用指南](#五-langsmith-监控与使用指南)
- [六、 常见问题与排错（FAQ）](#六-常见问题与排错faq)

---

## 一、 项目文件结构

```text
code/chapter11/
├── langgraph.json       # LangGraph CLI / Studio 核心配置文件
├── graph.py             # 智能体图构建逻辑（模型、工具与 Agent 编译）
├── requirements.txt     # 项目 Python 依赖包清单
├── .env.example         # 环境变量配置模板
└── README.md            # 项目说明文档
```

### 核心文件说明

| 文件 | 作用说明 |
| :--- | :--- |
| **`langgraph.json`** | LangGraph 服务的配置文件。声明对外暴露的图入口（`"chatbot": "./graph.py:graph"`）、环境依赖与 `.env` 路径。 |
| **`graph.py`** | 核心业务代码。包含心知天气查询工具 `@tool get_weather`、阿里百炼大模型初始化以及 ReAct 结构 Agent 的编译。 |
| **`requirements.txt`** | 声明项目运行所需的核心依赖包。 |
| **`.env`** | 本地私有环境变量配置文件（**已加入 `.gitignore`，切勿提交至远程仓库**）。 |

---

## 二、 环境准备与依赖安装

确保在项目根目录激活了 Python 虚拟环境，然后安装依赖：

```powershell
# 1. 进入 chapter11 目录
cd code/chapter11

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装 LangGraph 开发服务 CLI（用于可视化 Studio 调试）
pip install "langgraph-cli[inmem]"
```

---

## 三、 环境变量配置（.env）

在 `code/chapter11/` 目录下复制 `.env.example` 并重命名为 `.env`：

```ini
# 阿里百炼（通义千问）模型配置
ALI_BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALI_BAILIAN_API_KEY=sk-your-aliyun-bailian-api-key

# 心知天气 API Key（用于即时天气查询）
XINZHI_WEATHER_API_KEY=your-seniverse-api-key

# LangSmith 链路追踪配置
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your_langsmith_api_key
LANGSMITH_PROJECT=langgraph_chatbot
```

### 配置项详解

- **`ALI_BAILIAN_API_KEY`**: 阿里云百炼平台 API 密钥。
- **`XINZHI_WEATHER_API_KEY`**: 心知天气 API 密钥。
- **`LANGSMITH_TRACING=true`**: 启用 LangSmith 自动追踪。开启后，每一次 Agent 执行链都会自动上报至 LangSmith 云端。
- **`LANGSMITH_PROJECT`**: **LangSmith 云端项目名称**。可自定义命名（如 `langgraph_chatbot`）。若云端不存在该项目，首次请求时 LangSmith 会自动创建。

---

## 四、 项目启动与运行

### 方式一：命令行单次脚本运行

直接运行 Python 脚本进行控制台交互测试：

```powershell
python graph.py
```

---

### 方式二：使用 LangGraph Studio 进行可视化调试（推荐）

通过 LangGraph CLI 启动本地开发服务，享受节点流转图与交互式 UI 调试：

> **Windows 系统注意事项**：
> 为防止控制台因 Emoji 或文件默认编码触发 `UnicodeDecodeError (GBK)`，启动前需开启 UTF-8 模式：

```powershell
# 1. 开启 Python 全局 UTF-8 模式
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

# 2. 启动 LangGraph 开发者服务
langgraph dev
```

启动成功后，控制台将输出类似信息：

```text
╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

- 浏览器将自动打开 **Studio UI**。
- 在左侧选择 `chatbot`，即可输入消息（如：`帮我查询一下北京的天气情况`）进行可视化图调试。

---

## 五、 LangSmith 监控与使用指南

配置并启动项目后，每次对话执行均可在 **[LangSmith 控制台](https://smith.langchain.com/)** 实时监控：

1. **查看项目追踪（Traces）**：
   - 登录 LangSmith，进入 Projects 页面，点击进入 `langgraph_chatbot`。
   - 可以看到每一轮对话的完整执行树（Execution Tree）。
2. **多维度调用分析**：
   - **节点流转**：清晰查看输入消息 $\rightarrow$ `model` 思考 $\rightarrow$ `get_weather` 工具调用 $\rightarrow$ 最终回复的全流程。
   - **Token 消耗**：准确统计 Prompt Tokens、Completion Tokens 及预计花费。
   - **耗时与延迟**：定位网络请求、模型推理及工具调用的耗时瓶颈。
   - **异常捕获**：若工具调用或模型调用失败，LangSmith 会标红并展示完整堆栈信息。

---

## 六、 常见问题与排错（FAQ）

### Q1: 启动 `langgraph dev` 报 `UnicodeDecodeError: 'gbk' codec can't decode...`？
- **原因**：Windows 简体中文系统默认采用 GBK 文件编码，读取 UTF-8 模板时报错。
- **解决**：在终端执行 `$env:PYTHONUTF8="1"` 和 `$env:PYTHONIOENCODING="utf-8"` 后再启动。

### Q2: 百炼模型报错 `Role must be in [...] and the role in last message must be in ["user", "function", "tool"]`？
- **原因**：阿里百炼 API 规定请求消息列表的**最后一条消息必须是用户（user）或工具返回（tool）**，不能以 `system` 或 `assistant` 结尾。
- **解决**：在 Studio 中测试时，直接在下方 Chat 输入框中发送用户问题，避免单独点击 `Continue -> model`。

### Q3: `langgraph.json` 可以配置多个 Agent 吗？
- **可以**。在 `graphs` 字段中添加多个路由即可：
  ```json
  "graphs": {
      "weather_bot": "./graph.py:graph",
      "another_agent": "./other_graph.py:graph"
  }
  ```
