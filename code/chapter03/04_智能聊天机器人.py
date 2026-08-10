"""
智能聊天机器人 Pro - 基于 Gradio + LangChain 的多轮流式聊天机器人

功能亮点:
1. 现代高颜值 Web 界面: 采用双栏 Glassmorphism 现代质感 UI 设计
2. 多轮对话: 保留历史消息, 支持上下文连续对话
3. 流式打字机输出: 逐字平滑返回, 卓越用户体验
4. 可定制人设 & 预设角色一键切换: 支持运行时修改系统提示词
5. 快捷提示词 Pill: 点击即发送热门问题示例
6. 历史截断: 保留最近 50 条消息, 防止 token 溢出
"""

import os
from collections.abc import Generator
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── 环境配置 ──────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

DEFAULT_SYSTEM_PROMPT = (
    "你是一个智能聊天机器人, 擅长用友好、专业的语气回答各种问题。"
    "你可以进行多轮对话, 记住之前的聊天内容, 给出连贯的回复。"
)

PROMPT_PRESETS = {
    "🤖 通用": DEFAULT_SYSTEM_PROMPT,
    "💻 代码专家": "你是一名资深全栈架构师与 Python 代码专家，回答问题条理清晰，注重性能与最佳实践，并提供高质量代码范例。",
    "🌐 翻译大师": "你是一名精通中英等多国语言的同声传译大师，翻译地道流畅，优雅且符合上下文表达习惯。",
    "🎭 幽默导师": "你是一个幽默风趣、脑洞大开的智囊伙伴，喜欢用风趣幽默的打比方和段子解答深刻的技术与人生难题。",
}

MAX_HISTORY = 50  # 保留最近的消息数量

# ── 模型与链初始化 ────────────────────────────────────
api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not api_key:
    raise RuntimeError("请在 .env 文件中配置 ALI_BAILIAN_API_KEY")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    api_key=api_key,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model | StrOutputParser()


# ── 工具函数 ──────────────────────────────────────────
def gradio_history_to_langchain(history: list[dict]) -> list:
    """将 Gradio messages 格式历史转换为 LangChain 消息列表"""
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ── 交互事件处理 ──────────────────────────────────────
def user_submit(message: str, history: list[dict]):
    """处理用户发送消息，立即呈现在聊天框并清空输入框"""
    if not message.strip():
        return "", history
    new_history = history + [{"role": "user", "content": message}]
    return "", new_history


def bot_respond(history: list[dict], system_prompt: str):
    """流式输出 AI 响应"""
    if not history or history[-1]["role"] != "user":
        return

    user_message = history[-1]["content"]
    past_history = history[:-1]

    # 组装消息列表
    messages = gradio_history_to_langchain(past_history)
    messages.append(HumanMessage(content=user_message))
    messages = messages[-MAX_HISTORY:]

    # 追加 AI 回复占位
    history.append({"role": "assistant", "content": ""})

    partial = ""
    try:
        for chunk in chain.stream(
            {
                "system_prompt": system_prompt,
                "messages": messages,
            }
        ):
            partial += chunk
            history[-1]["content"] = partial
            yield history
    except Exception as exc:
        history[-1]["content"] = f"[错误] 调用模型失败: {exc}"
        yield history


# ── 自定义美化 CSS ──────────────────────────────────────
CUSTOM_CSS = """
/* ═══ 全局背景 ═══ */
body {
    background: linear-gradient(160deg, #0a0a18 0%, #101030 45%, #0c0c20 100%) !important;
    min-height: 100vh !important;
}

/* 隐藏 Gradio 默认页脚 */
footer, .footer, [data-testid="footer"] {
    display: none !important;
}

/* ═══ 主容器 ═══ */
.gradio-container {
    max-width: 96% !important;
    width: 96% !important;
    padding: 10px 16px !important;
    gap: 0 !important;
}

/* ═══ 顶部 Header ═══ */
.header-banner {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.45), rgba(15, 23, 42, 0.65)) !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: 14px !important;
    padding: 10px 24px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15) !important;
}

.header-banner h1 {
    font-size: 22px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 3px 0 !important;
    line-height: 1.3 !important;
    text-align: center !important;
}

.header-subtext {
    font-size: 12px !important;
    color: #94a3b8 !important;
    margin: 0 0 6px 0 !important;
    text-align: center !important;
}

.badge-row {
    text-align: center !important;
}

.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.2);
    margin: 0 3px;
}

.badge-tag.purple {
    background: rgba(129, 140, 248, 0.1);
    color: #818cf8;
    border-color: rgba(129, 140, 248, 0.2);
}

.badge-tag.green {
    background: rgba(52, 211, 153, 0.1);
    color: #34d399;
    border-color: rgba(52, 211, 153, 0.2);
}

/* ═══ 主体布局 ═══ */
.main-row {
    gap: 12px !important;
    align-items: flex-start !important;
}

/* ═══ 左侧侧边栏 ═══ */
.sidebar-panel {
    min-width: 270px !important;
    max-width: 270px !important;
    width: 270px !important;
    flex: 0 0 270px !important;
    max-height: 590px !important;
    background: rgba(18, 18, 38, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 14px !important;
    padding: 14px !important;
    box-sizing: border-box !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    overflow-y: auto !important;
}

/* 自定义滚动条 */
.sidebar-panel::-webkit-scrollbar {
    width: 4px;
}
.sidebar-panel::-webkit-scrollbar-track {
    background: transparent;
}
.sidebar-panel::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.25);
    border-radius: 4px;
}
.sidebar-panel::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.45);
}

/* 侧边栏标题 */
.sidebar-panel h3 {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #a5b4fc !important;
    margin: 0 0 8px 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.12) !important;
}

.sidebar-panel h4 {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #818cf8 !important;
    margin: 10px 0 6px 0 !important;
}

.sidebar-panel hr {
    border: none !important;
    border-top: 1px solid rgba(99, 102, 241, 0.08) !important;
    margin: 8px 0 !important;
}

/* ═══ 右侧聊天区 ═══ */
.main-chat-panel {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    gap: 10px !important;
}

/* 聊天框 */
.chatbot-box {
    border-radius: 14px !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1) !important;
}

/* 输入区域 */
.input-row {
    gap: 8px !important;
    align-items: center !important;
}

/* ═══ 按钮样式 ═══ */
/* 预设角色按钮 — 紫色系 */
.preset-btn {
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 7px 4px !important;
    background: rgba(99, 102, 241, 0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    color: #c7d2fe !important;
    transition: all 0.2s ease !important;
}
.preset-btn:hover {
    background: rgba(99, 102, 241, 0.22) !important;
    border-color: rgba(99, 102, 241, 0.45) !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
}

/* 快捷示例按钮 — 青色系 */
.sample-btn {
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 7px 4px !important;
    background: rgba(56, 189, 248, 0.06) !important;
    border: 1px solid rgba(56, 189, 248, 0.12) !important;
    color: #bae6fd !important;
    transition: all 0.2s ease !important;
}
.sample-btn:hover {
    background: rgba(56, 189, 248, 0.2) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
}

/* 清空按钮 — 红色系 */
.clear-btn {
    background: rgba(239, 68, 68, 0.08) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239, 68, 68, 0.18) !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    padding: 7px 4px !important;
    transition: all 0.2s ease !important;
}
.clear-btn:hover {
    background: rgba(239, 68, 68, 0.22) !important;
    color: #fff !important;
    border-color: rgba(239, 68, 68, 0.45) !important;
}

/* 发送按钮 */
.send-btn {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
}
"""

# ── 主题配置 ─══════════════════════════════════════════
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)


# ── Gradio 界面构建 ────────────────────────────────────
with gr.Blocks(title="智能聊天机器人") as demo:
    # 顶部 Header
    with gr.Column(elem_classes=["header-banner"]):
        gr.HTML(
            """
            <h1>🤖 智能聊天机器人</h1>
            <p class="header-subtext">基于 LangChain + 阿里云百炼通义千问大模型 | 多轮对话 & 流式输出</p>
            <div class="badge-row">
                <span class="badge-tag">⚡ qwen3.7-max</span>
                <span class="badge-tag purple">🔗 LangChain 1.3.14</span>
                <span class="badge-tag green">🌊 Stream Flow</span>
            </div>
            """
        )

    # 主体布局：左侧固定宽度控制栏 + 右侧全屏响应式主区
    with gr.Row(elem_classes=["main-row"]):
        # 左侧控制面板 (固定 270px)
        with gr.Column(elem_classes=["sidebar-panel"]):
            gr.Markdown("### ⚙️ 人设与配置")
            system_prompt_input = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                label="系统提示词",
                lines=3,
                placeholder="请输入自定义机器人人设...",
                info="支持随时修改提示词",
            )

            gr.Markdown("#### 🎭 预设角色")
            preset_btns = []
            preset_items = list(PROMPT_PRESETS.items())
            for i in range(0, len(preset_items), 2):
                with gr.Row():
                    for name, _ in preset_items[i : i + 2]:
                        btn = gr.Button(
                            name, size="sm", elem_classes=["preset-btn"], scale=1
                        )
                        preset_btns.append((btn, name))

            gr.Markdown("#### 💡 快捷提问")
            sample_questions = [
                ("📝 简单介绍", "你好, 请简单介绍一下你自己"),
                ("🐍 Python 快排", "用 Python 写一个快速排序算法"),
                ("❓ 什么是 RAG", "解释一下什么是 RAG (检索增强生成)"),
                ("🌐 翻译名言", "帮我翻译: Knowledge is power"),
                ("🧑‍💻 程序员笑话", "讲一个程序员的日常趣事笑话"),
            ]
            sample_btns = []
            for i in range(0, len(sample_questions), 2):
                with gr.Row():
                    for label, full_q in sample_questions[i : i + 2]:
                        btn = gr.Button(
                            label, size="sm", elem_classes=["sample-btn"], scale=1
                        )
                        sample_btns.append((btn, full_q))

            gr.Markdown("---")
            clear_btn = gr.Button(
                "🗑️ 清空历史对话", elem_classes=["clear-btn"], size="sm"
            )

        # 右侧主对话区
        with gr.Column(elem_classes=["main-chat-panel"]):
            chatbot = gr.Chatbot(
                height=520,
                show_label=False,
                placeholder="👋 你好！我是智能 AI 助手。\n\n请在下方输入框提问，或选择左侧的快捷示例开启对话！",
                avatar_images=(
                    "https://api.dicebear.com/9.x/avataaars/svg?seed=User",
                    "https://api.dicebear.com/9.x/bottts/svg?seed=ChatbotPro",
                ),
                elem_classes=["chatbot-box"],
            )

            with gr.Row(elem_classes=["input-row"]):
                msg_input = gr.Textbox(
                    placeholder="输入您的问题 (按 Enter 发送)...",
                    show_label=False,
                    scale=7,
                    container=False,
                )
                send_btn = gr.Button(
                    "🚀 发送", variant="primary", scale=1, elem_classes=["send-btn"]
                )

    # ── 事件绑定 ──────────────────────────────────────────
    # 消息发送逻辑
    msg_input.submit(
        user_submit,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False,
    ).then(
        bot_respond,
        inputs=[chatbot, system_prompt_input],
        outputs=[chatbot],
    )

    send_btn.click(
        user_submit,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
        queue=False,
    ).then(
        bot_respond,
        inputs=[chatbot, system_prompt_input],
        outputs=[chatbot],
    )

    # 角色预设一键切换绑定
    def make_preset_handler(p_text):
        return lambda: p_text

    for btn, name in preset_btns:
        btn.click(
            fn=make_preset_handler(PROMPT_PRESETS[name]),
            inputs=[],
            outputs=[system_prompt_input],
        )

    # 快捷提问示例绑定
    def make_sample_handler(q_text):
        def handler(history, sys_prompt):
            _, updated_hist = user_submit(q_text, history)
            yield from bot_respond(updated_hist, sys_prompt)

        return handler

    for btn, full_q in sample_btns:
        btn.click(
            fn=make_sample_handler(full_q),
            inputs=[chatbot, system_prompt_input],
            outputs=[chatbot],
        )

    # 清空对话
    clear_btn.click(lambda: [], outputs=[chatbot], queue=False)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS, share=False)
