# 大模型 / Agent 评估体系 — 面试知识点速查

## 一、核心认知转型（面试必答题）

> **面试官常问**："为什么传统评测（如 MMLU）在 Agent 上失效了？"

| 维度 | 传统模型评估 | Agent 评估 |
|---|---|---|
| 关注点 | 输入 → 输出的准确性 | **过程 + 结果 + 效率 + 安全**的综合表现 |
| 确定性 | 相同输入 → 相同输出 | 多步骤交互，**非确定性**路径 |
| 失效模式 | 单点错误 | **链路式失效**（蝴蝶效应） |
| 评估对象 | 单轮回答 | 完整的**执行轨迹（Trajectory）** |

---

## 二、四层评估框架（体系化回答）

面试中建议按这四个层次展开：

```
┌─────────────────────────────────────────────┐
│  第4层：风险层（Safety / Guardrails）         │
│  → 是否越权？是否误操作？是否触发合规问题？    │
├─────────────────────────────────────────────┤
│  第3层：效率层（Efficiency）                  │
│  → 响应延迟、Token 消耗、工具调用次数          │
├─────────────────────────────────────────────┤
│  第2层：过程层（Reasoning & Process）         │
│  → 推理路径合理性、工具选择精准度              │
├─────────────────────────────────────────────┤
│  第1层：结果层（Outcome）                     │
│  → 任务完成率、准确率、格式正确率              │
└─────────────────────────────────────────────┘
```

### 各层关键指标

| 层级 | 关键指标 | 说明 |
|---|---|---|
| **结果层** | Task Completion Rate | 任务是否最终达成目标 |
| | Answer Accuracy | 最终答案的正确性 |
| | Format Compliance | JSON 解析成功率、结构化输出合规率 |
| **过程层** | Trajectory Quality | 执行路径是否合理（是否绕弯、死循环） |
| | Tool Selection Accuracy | 是否选择了正确的工具 |
| | Tool Parameter Validity | 工具调用参数是否正确 |
| | Reasoning Coherence | 思考链是否逻辑连贯 |
| **效率层** | Latency (E2E) | 端到端响应时间 |
| | Token Usage | 总 Token 消耗（成本核算） |
| | Tool Call Count | 完成任务所需的工具调用次数 |
| **风险层** | Hallucination Rate | 幻觉/捏造信息的比例 |
| | Guardrail Violations | 安全护栏触发次数 |
| | Data Leakage | 是否泄露敏感信息 |

---

## 三、LLM-as-a-Judge（面试高频考点）

### 核心原理

用**强模型**（如 GPT-4o）作为评委，评估**弱模型/Agent**的输出质量。

```
输入 + Agent 输出 + 评分标准（Rubric）
        ↓
    Judge LLM
        ↓
  评分（1-5）+ 理由说明
```

### 常见技术

| 技术 | 说明 |
|---|---|
| **G-Eval** | 用 CoT 引导 Judge 按自定义标准评分（DeepEval 内置） |
| **Pairwise Comparison** | 让 Judge 对比两个输出选"赢家"，适合 A/B 测试 |
| **Reference-based** | 提供标准答案，让 Judge 对比评分 |

### 面试必答：局限性

> **Q: LLM-as-a-Judge 有什么问题？**

1. **位置偏差（Position Bias）**：倾向于选择排在前面的答案
2. **冗长偏差（Verbosity Bias）**：倾向于给更长的回答打高分
3. **自我偏好（Self-preference Bias）**：GPT-4 倾向于给 GPT-4 的输出打高分

**应对策略**：
- 用人类标注数据**定期校准** Judge 模型
- 交换输出顺序做**双向评估**
- 使用 CoT 引导 Judge **先推理再打分**

---

## 四、RAG 评估指标（Ragas 框架）

如果你的 Agent 涉及 RAG，以下指标必须掌握：

| 指标 | 评估什么 | 需要的数据 |
|---|---|---|
| **Faithfulness（忠实度）** | 回答是否基于检索到的上下文，有无幻觉 | 上下文 + 回答 |
| **Answer Relevancy（相关性）** | 回答是否切题 | 问题 + 回答 |
| **Context Precision（上下文精确率）** | 检索到的文档是否相关 | 问题 + 上下文 |
| **Context Recall（上下文召回率）** | 是否检索到了所有相关文档 | 问题 + 上下文 + 标准答案 |

```python
# Ragas 评估伪代码
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

results = evaluate(
    dataset=eval_dataset,       # 包含 question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy]
)
print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92}
```

---

## 五、主流评估工具对比

| 工具 | 定位 | 特点 |
|---|---|---|
| **DeepEval** | 通用 LLM/Agent 评估 | 类 pytest 风格，50+ 内置指标，CI/CD 集成 |
| **Ragas** | RAG 管线专项评估 | Faithfulness、Relevancy 等 RAG 专属指标 |
| **LangSmith** | LangChain 生态的可观测性平台 | Trace 追踪 + 在线评估 + 数据集管理 |
| **MLflow** | 企业级实验追踪 | 模型版本管理 + 评估对比 |
| **Promptfoo** | Prompt 工程评估 | 批量测试不同 Prompt 变体 |

---

## 六、主流 Benchmark（面试加分项）

| Benchmark | 评估能力 | 说明 |
|---|---|---|
| **AgentBench** | Agent 综合能力 | 涵盖代码、游戏、网页、数据库等 8 个环境 |
| **SWE-bench** | 软件工程能力 | 让 Agent 修复真实 GitHub Issue |
| **WebArena** | 网页导航能力 | 在真实网站环境中完成任务 |
| **τ-bench** | 工具-Agent-用户交互 | 评估工具调用和用户意图理解 |
| **HumanEval / MBPP** | 代码生成 | 函数级代码生成正确性 |
| **MMLU / GPQA** | 知识推理（模型级） | 传统多选题基准（Agent 评估中不够用） |

---

## 七、高频面试问答

### Q1: 如何评估一个 Agent 系统？

**满分回答框架**：

> 1. **离线评估**：构建开发集（Dev Set），用 LLM-as-a-Judge + 程序化规则做自动化评估
> 2. **链路追踪**：通过 Trace 定位是哪一环出问题（检索？推理？工具调用？）
> 3. **线上监控**：影子测试（Shadow Test）+ 生产指标看板（延迟、Token 成本、失败率）
> 4. **持续迭代**：将生产中的失败案例自动转化为回归测试集

### Q2: Agent 在生产环境出现失败，如何归因？

```
用户输入 → 意图识别 → RAG 检索 → 模型推理 → 工具调用 → 结果组装
              ↑           ↑          ↑           ↑          ↑
           意图错了？   没搜到？    推理跑偏？  参数错了？  格式坏了？
```

**关键**：通过 **Trace（链路追踪）** 逐步检查每个节点的输入输出。

### Q3: 怎么衡量评估结果是否可靠？

1. **Judge 校准**：用人类标注的"金标准"数据集，计算 Judge 和人类的**一致性（Cohen's Kappa）**
2. **多 Judge 投票**：用多个 Judge 模型打分，取共识
3. **置信区间**：多次评估，报告均值 ± 标准差

### Q4: 评估数据集怎么构建？

| 方法 | 适用场景 |
|---|---|
| **人工标注** | 金标准，但成本高 |
| **LLM 生成 + 人工审核** | 快速扩充，需抽样校验 |
| **生产日志挖掘** | 最真实的用户场景 |
| **对抗样本构造** | 测试边界情况和鲁棒性 |

---

## 八、面试实战建议

> [!TIP]
> **用 STAR 法则准备一个具体案例**
> 
> 准备一个你做过的 Agent 项目（如客服 Agent、数据分析 Agent），重点描述：
> - **S**: 业务场景和评估挑战
> - **T**: 你需要回答"这个 Agent 能不能上线"
> - **A**: 你设计了哪些评估指标、用了什么工具
> - **R**: 评估发现了什么问题、如何迭代优化

> [!IMPORTANT]
> **面试官最看重的不是你知道多少 Benchmark，而是你有没有「观测 → 评估 → 迭代」的工程闭环思维。**
