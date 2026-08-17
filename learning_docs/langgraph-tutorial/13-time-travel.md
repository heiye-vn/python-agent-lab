# 第 13 章：时间旅行 Time Travel

检查点链本质是一条**可回溯的执行历史**。时间旅行 = 在这条历史上任选一点：**Replay**（从那里重跑）或 **Fork**（从那里分叉出新未来）。这是调试、纠错、A/B 对比的杀手锏，也是 LangGraph Studio 的核心交互。

## 13.1 浏览历史：get_state_history

```python
config = {"configurable": {"thread_id": "t-1"}}
graph.invoke(inputs, config)

for snapshot in graph.get_state_history(config):
    print(
        snapshot.config["configurable"]["checkpoint_id"],  # 快照 ID
        snapshot.metadata.get("step"),                     # 第几个 superstep
        snapshot.next,                                     # 当时下一批节点
        snapshot.created_at,
    )
```

注意迭代顺序是**从新到旧**。每个 checkpoint 都有唯一 `checkpoint_id`，它就是"时间坐标"。

## 13.2 Replay：从历史某点重放

把历史 checkpoint_id 放进 configurable，输入传 `None`：

```python
# 找到目标快照
history = list(graph.get_state_history(config))
target = [s for s in history if s.metadata.get("step") == 2][-1]

# 从该快照继续执行（之后的节点会重跑）
replay_config = {
    "configurable": {
        "thread_id": "t-1",
        "checkpoint_id": target.config["configurable"]["checkpoint_id"],
    }
}
graph.invoke(None, replay_config)
```

**两个用途**：
1. **修复后重跑**：发现第 3 步的逻辑有 bug → 改代码 → 从 step 3 的快照 replay，前 2 步不重复消耗 token
2. **换参数重跑**：先 `update_state` 改掉状态里的参数字段，再 replay

```python
# 先改状态再重放：把检索 top_k 从 3 改成 10
graph.update_state(replay_config, {"top_k": 10})
graph.invoke(None, replay_config)
```

## 13.3 Fork：从历史分叉新时间线

Replay 是"覆盖未来"，Fork 是"平行宇宙"——**换个新 thread_id + 指定历史 checkpoint_id**：

```python
fork_config = {
    "configurable": {
        "thread_id": "t-1-fork-a",                        # 新线程！
        "checkpoint_id": target.config["configurable"]["checkpoint_id"],
    }
}
graph.invoke(None, fork_config)   # 从旧快照出发，在新线程跑出不同未来
```

原 thread t-1 的历史完全不受影响。

**典型场景**：
- **A/B 测试**：同一状态起两个 fork，跑不同模型/参数，对比结果
- **"撤销"**：用户对结果不满意 → fork 回两步前，改输入重新生成
- **危险操作回滚**：审批发现错了，fork 回审批前重新来

## 13.4 Fork 时修改输入状态

Fork 的第一跳也可以带输入（而不是 None），相当于"改写历史后分叉"：

```python
# 回到 step 2，把用户问题改掉，在新线程重跑
graph.invoke(
    {"messages": [RemoveMessage(id=last_user_msg.id),
                  HumanMessage(content="换个问法：...")]},
    fork_config,
)
```

## 13.5 实战场景速查

| 场景 | 操作 |
|---|---|
| "第 3 步为什么走了这条路？" | Studio 点开 step 3 快照看 state + LangSmith 看 trace |
| "改了代码想重试后半段" | replay：同 thread + checkpoint_id + `invoke(None)` |
| "换个参数再跑一遍对比" | `update_state` 后 replay，或 fork 到新 thread |
| "回到用户提问前重问" | fork + 新输入 |
| "生产事故复盘" | dump 该 thread 全部快照，离线逐 step 重放 |

## 13.6 在 LangGraph Studio 中时间旅行（零代码）

启动本地 server 后打开 Studio（第 23 章详解）：

1. 左侧选择 thread → 看到检查点时间线
2. 点击任意历史节点 → 右侧显示当时完整状态
3. 修改任何字段 → 点 **Fork** / **Replay** 按钮 → 图从该点重新执行

调试 Agent 时"在这里改一下提示词会怎样"的实验，从"重跑整个对话"变成"点一下按钮"。

## 13.7 注意事项

- 时间旅行依赖 checkpointer（没配 = 无历史可用）
- Replay 重跑的节点**副作用会真实发生**（再发一次邮件）——对外副作用仍需幂等保护
- checkpoint_id 是内容的确定性哈希（可跨进程一致），thread 内唯一
- 历史 fork 不清理会持续占存储；生产注意归档策略

## 本章小结

- 检查点链 = 执行历史；`checkpoint_id` = 时间坐标
- Replay：同 thread 回退重跑（可先 update_state 改状态）
- Fork：新 thread 从历史点分叉，平行对比/撤销/回滚
- Studio 把这一切做成了可视化点击操作

> 第四部分完成。接下来：Human-in-the-Loop——企业落地的高频刚需。
