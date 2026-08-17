# 第 22 章：Deep Agents

Deep Agents（`deepagents` 包）是 LangChain 官方推出的**高层 Agent 范式**：在 `create_react_agent` 之上内置了「规划工具 + 子 Agent + 虚拟文件系统 + 上下文管理」四大件，专攻**长时程、多阶段、大产出**的任务（深度研究、报告撰写、代码库分析）。它代表了"上下文工程最佳实践"的官方封装。

## 22.1 解决什么问题：长任务的四种死法

让一个普通 ReAct Agent 跑"研究并写一份 30 页报告"，通常会死于：

| 死法 | 原因 | Deep Agents 的解法 |
|---|---|---|
| 迷失方向 | 走到第 15 步忘了总目标 | **规划工具**：内置 todo/计划，执行中持续对照 |
| 上下文爆炸 | 阅读材料淹没了指令 | **文件系统**：中间产物写文件，上下文只留引用 |
| 无法并行/分工 | 单 Agent 串行苦干 | **子 Agent（task 工具）**：独立上下文分头干活 |
| 产出质量崩 | 一次性生成长文 | **分段写文件 + 迭代编辑** |

核心思想一句话：**给 Agent 一个"办公桌"（文件系统）和"工作计划"（todo），让它像人一样处理大任务**。

## 22.2 快速上手

```bash
pip install deepagents
```

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[web_search],                          # 你的业务工具
    instructions=(
        "你是一名行业分析师。接到任务后：\n"
        "1) 先用规划工具拆解任务并跟踪进度\n"
        "2) 搜索并阅读资料，把要点写入笔记文件\n"
        "3) 子任务可委派子 Agent\n"
        "4) 最终报告写入 report.md"
    ),
    model="openai:gpt-4.1",                      # 可选，默认用环境配置
)

result = agent.invoke(
    {"messages": [("user", "调研 2026 年 Agent 平台市场格局，写一份 2000 字报告")]},
    config={"configurable": {"thread_id": "deep-1"}},
)
```

Agent 运行时会自动获得一组内置工具（无需自己写）：

- **规划**：`write_todos` —— 创建/更新任务清单，计划存于状态并持续注入上下文
- **文件系统**：`ls` / `read_file` / `write_file` / `edit_file` / `glob` / `grep` —— 在虚拟工作区读写文件
- **子 Agent**：`task` —— 派生一个全新上下文的子 Agent 执行子任务，只回传结论

## 22.3 自定义子 Agent

```python
agent = create_deep_agent(
    tools=[web_search, sql_query],
    instructions="首席研究员工作流……",
    subagents={
        "web_researcher": {
            "description": "联网搜集资料并提炼带来源的要点",
            "prompt": "你是搜索研究员。只搜集和提炼，不做综合判断。",
            "tools": [web_search],
        },
        "data_analyst": {
            "description": "查询内部数据库并产出数据表",
            "prompt": "你是数据分析师，只负责取数和核对数字。",
            "tools": [sql_query],
        },
    },
)
```

主 Agent 会在说明书里"看到"这些子 Agent 的描述，需要时通过 `task` 工具委派。**子 Agent 上下文完全独立**——它读 50 个页面的污染不会进入主 Agent 窗口，只有结论回传（第 20 章隔离原则的落地）。

## 22.4 文件系统与后端（backend）

默认文件系统是**内存虚拟 FS**（会话内有效，可配合 checkpointer 持久化）。生产可接真实后端：

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[...],
    instructions="...",
    backend=my_storage_backend,    # 实现 FS 接口：接 S3、本地磁盘、公司文档系统
)
```

用途：
- **本地磁盘**：产出的 report.md 直接落盘
- **对象存储**：多服务器部署时共享工作区
- **挂载知识库**：把资料预置到 `/inputs/`，Agent 用 read/grep 工具按需取用（比一次性塞 prompt 高效得多——这本身就是一种 RAG）

## 22.5 上下文管理：内建摘要

长任务必然超窗。Deep Agents 内置滚动摘要机制（原理即第 11/18 章的 SummarizationMiddleware 思路）：历史消息过长时自动压缩为摘要 + 保留近期消息 + 计划/文件清单这类"高价值状态"始终在场。你也可以叠加自己的 middleware / pre_model_hook。

## 22.6 与其他方案的关系

| 方案 | 定位 | 何时选 |
|---|---|---|
| 手写 StateGraph | 完全掌控流程 | 流程确定、强业务约束 |
| create_react_agent | 单角色工具循环 | 聊天/客服/简单任务 |
| **deepagents** | 长任务作业系统 | 研究、报告、代码分析等**小时级任务** |
| langgraph-supervisor | 自定义团队拓扑 | 需要自己设计成员与交接协议 |

它们不互斥：deepagents 的子 Agent、supervisor 的成员，底层都是 create_react_agent；主流程想换 deepagents 就换。

## 22.7 生产化建议

1. **模型分级**：主 Agent 用强模型（规划质量决定一切），子 Agent 用性价比模型
2. **预算硬上限**：`recursion_limit` + 每任务 token 预算（deep 任务最容易失控烧钱）
3. **产出物校验**：报告文件生成后接独立"审阅 Agent"或结构化校验节点
4. **配合 LangSmith**：长任务必看 trace 的树状结构（子 Agent 嵌套 span），否则无法调试
5. **中断恢复**：任务跑 40 分钟断在第 30 步？checkpointer + `invoke(None)` 续跑——deep 任务强烈建议配持久化

## 本章小结

- Deep Agents = create_react_agent + 规划 + 虚拟文件系统 + 子 Agent + 上下文管理
- 内置工具：write_todos、文件读写、task 派子 Agent——不用自己写
- 文件系统是"上下文外挂"：中间产物出窗、按需回读；backend 可接真实存储
- 适合小时级长任务；生产必须配预算上限、产出校验、持久化

> 第七部分完成。接下来第八部分：从本地开发到企业级部署。
