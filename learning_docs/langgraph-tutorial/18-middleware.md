# 第 18 章：Middleware 中间件

Middleware 是 LangGraph 1.0 引入的重磅机制：**在不改 Agent 代码的前提下，拦截和修改模型调用、工具执行、状态流转**。官方预构建 Agent 的所有官方扩展（human-in-the-loop、guardrails、模型路由…）正在全面迁移到这套机制上。

## 18.1 解决什么问题

Agent 的横切关注点（cross-cutting concerns）：日志、审计、敏感词、成本控制、动态提示、权限——过去要么改 Agent 源码，要么用一堆钩子拼接。Middleware 把它们做成**可插拔的洋葱层**：

```
请求 → [Middleware A → Middleware B → Agent 核心] → 响应
              ↑ 每个 middleware 可在"进入前/出去后"做手脚
```

价值：中间件一次编写、跨 Agent 复用（团队/公司级资产），Agent 代码保持纯粹业务。

## 18.2 核心 API：AgentMiddleware

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.wrappers import WrapToolCall
from langgraph.prebuilt import create_react_agent
```

一个 middleware 就是继承 `AgentMiddleware` 的类，按需覆写钩子：

```python
class AuditMiddleware(AgentMiddleware):
    def before_model(self, request: ModelRequest, runtime) -> dict | None:
        """模型调用前：可读取/修改将要发给模型的消息，返回状态更新"""
        log(request.state["messages"][-1])
        return None    # None = 不改任何东西

    def wrap_model_call(self, request: ModelRequest, handler, runtime):
        """包裹模型调用本身：可换模型、加超时、计费"""
        response = handler(request, runtime)   # 调用原模型
        log_cost(response.usage_metadata)
        return response

    def wrap_tool_call(self, request: ToolRequest, handler: Callable, runtime):
        """包裹工具执行：参数审查、权限校验、结果脱敏"""
        result = handler(request, runtime)
        return result

    def after_model(self, response, runtime) -> dict | None:
        """模型响应后：检查输出、注入额外消息"""
        return None
```

> 注意：中间件的钩子签名在 1.x 各小版本间有微调（如 runtime 参数、钩子名称），以你所用版本的 `langchain.agents.middleware` 文档/源码签名为准。本章展示的是设计模式，思路不变。

挂载：

```python
agent = create_react_agent(
    llm, tools,
    prompt="你是客服助手。",
    middleware=[AuditMiddleware(), GuardrailsMiddleware()],
    checkpointer=cp,
)
```

## 18.3 实战中间件示例

### 示例一：敏感词 Guardrail（进/出双向）

```python
class GuardrailsMiddleware(AgentMiddleware):
    BANNED_IN = ["竞品价格表", "内部考核"]

    def before_model(self, request, runtime):
        last = request.state["messages"][-1]
        if any(w in (last.content or "") for w in self.BANNED_IN):
            # 直接改写用户消息前的状态：插入拒答指令
            return {"messages": [{"role": "system", "content": "该话题禁止讨论，礼貌拒答。"}]}
        return None

    def after_model(self, response, runtime):
        if "公司机密" in (response.messages[-1].content or ""):
            return {"messages": [{"role": "assistant", "content": "（内容已按合规策略过滤）"}]}
        return None
```

### 示例二：工具权限控制（按租户）

```python
class ToolPermissionMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler, runtime):
        tool_name = request.name
        user = runtime.context.get("user")          # 运行时注入的上下文
        if tool_name in PRO_TOOLS and user.tier != "pro":
            return ToolResult({"content": "该功能需要升级到专业版"})
        return handler(request, runtime)
```

### 示例三：动态模型路由（简单问题用便宜模型）

```python
class ModelRouterMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler, runtime):
        msgs = request.state["messages"]
        if len(msgs) <= 2 and no_tool_so_far(msgs):   # 简单问答
            request.model = cheap_llm                   # 换模型
        return handler(request, runtime)
```

### 示例四：HITL 审批（中间件版，对比 17.5 的 hook 版）

```python
class ApprovalMiddleware(AgentMiddleware):
    DANGEROUS = {"delete_account", "transfer_money"}

    def after_model(self, response, runtime):
        last = response.messages[-1]
        dangerous = [tc for tc in (last.tool_calls or []) if tc["name"] in self.DANGEROUS]
        if dangerous:
            decision = interrupt({"type": "approval", "tool_calls": dangerous})
            if decision["action"] != "approve":
                return {"messages": [ToolMessage("用户拒绝", tool_call_id=tc["id"])
                                     for tc in dangerous], "goto_tool_node": False}
        return None
```

**中间件 vs hook 的关系**：`pre/post_model_hook` 一次只能挂一个且不可复用；middleware 是可组合、可参数化、跨项目携带的类。新代码优先 middleware。

## 18.4 @chain 装饰器与组合生态

Middleware 体系还提供 `@chain` 装饰器，用于定义可复用的编排片段（可被 middleware 拦截的自定义链）。1.x 的预构建架构（如 RAG、工具调用 Agent）都以 middleware 形式提供扩展点，例如官方 `langchain` 里的：

- HITL middleware（审批）
- SummarizationMiddleware（自动滚动摘要，解决长对话 token 爆炸）
- GuardrailsMiddleware（输入输出安全）

**拿来即用是最佳路径**——写自定义前先查官方中间件库（`from langchain.agents.middleware import ...`）。

例如直接启用自动摘要（免去第 11 章手写）：

```python
from langchain.agents.middleware.summarization import SummarizationMiddleware

agent = create_react_agent(
    llm, tools,
    middleware=[
        SummarizationMiddleware(
            model=llm,
            max_tokens_before_summary=6000,   # 超过预算触发摘要
            retain_last_n_messages=6,
        ),
    ],
    checkpointer=cp,
)
```

## 18.5 编写规范与调试

1. **单一职责**：一个 middleware 只做一件事（审计归审计、脱敏归脱敏），组合优于上帝类
2. **顺序即洋葱层次**：`middleware=[A, B]` 时 A 包裹 B——权限类放最外层，日志放更外层
3. **幂等与性能**：`wrap_model_call` 每轮循环都执行，别在里面做重查询
4. **调试**：LangSmith trace 中 middleware 的修改会体现在对应节点的输入/输出 diff 上；也可以在钩子里 `get_stream_writer()` 推自定义事件

## 本章小结

- Middleware = Agent 的可插拔横切层：before/after_model、wrap_model/tool_call
- 四大杀手锏场景：guardrails、工具权限、模型路由、HITL 审批
- 官方中间件库优先（如 SummarizationMiddleware），自定义保持单一职责
- 1.x 生态的扩展标准正在向 middleware 收敛，值得优先投入

> 下一章：上下文工程与 MCP——把工具生态接到标准协议上。
