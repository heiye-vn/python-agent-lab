import os
from pathlib import Path
from dotenv import load_dotenv

import gradio as gr
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not api_key:
    raise RuntimeError("请在 .env 文件中配置 ALI_BAILIAN_API_KEY")

# ──────────────────────────────────────────────
# 1. 模型、Prompt、Chain
# ──────────────────────────────────────────────

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    api_key=api_key,
)
parser = StrOutputParser()

chatbot_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你是一个智能AI助手，说话幽默风趣且专业。"),
        MessagesPlaceholder(variable_name="messages"),  # 手动传入历史
    ]
)

qa_chain = chatbot_prompt | model | parser  # LCEL 组合

# ──────────────────────────────────────────────
# 2. Gradio 组件
# ──────────────────────────────────────────────
CSS = """
.main-container {max-width: 1200px; margin: 0 auto; padding: 20px;}
.header-text {text-align: center; margin-bottom: 20px;}
"""


def create_chatbot():
    with gr.Blocks(title="聊天机器人") as demo:
        with gr.Column(elem_classes=["main-container"]):
            gr.Markdown(
                "# 🤖 LangChain智能对话机器人系统", elem_classes=["header-text"]
            )

            chatbot = gr.Chatbot(
                height=500,
                avatar_images=(
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f464.png",
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f916.png",
                ),
            )
            msg = gr.Textbox(placeholder="请输入您的问题...", container=False, scale=7)
            submit = gr.Button("发送", scale=1, variant="primary")
            clear = gr.Button("清空", scale=1)

        # ---------------  状态：保存 messages_list  ---------------
        state = gr.State([])  # 这里存放真正的 Message 对象列表

        # ---------------  主响应函数（流式） ----------------------
        async def respond(user_msg: str, chat_hist: list, messages_list: list):
            # 1) 输入为空直接返回
            if not user_msg.strip():
                yield "", chat_hist, messages_list
                return

            # 2) 追加用户消息与初始AI占位
            messages_list.append(HumanMessage(content=user_msg))
            chat_hist = chat_hist + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ""},
            ]
            yield "", chat_hist, messages_list  # 先显示用户消息

            # 3) 流式调用模型
            partial = ""
            try:
                async for chunk in qa_chain.astream({"messages": messages_list}):
                    partial += chunk
                    # 更新最后一条 AI 回复
                    chat_hist[-1] = {"role": "assistant", "content": partial}
                    yield "", chat_hist, messages_list
            except Exception as e:
                error_msg = f"⚠️ [调用失败] {e}"
                chat_hist[-1] = {"role": "assistant", "content": error_msg}
                yield "", chat_hist, messages_list
                return

            # 4) 完整回复加入历史，裁剪到最近 50 条
            messages_list.append(AIMessage(content=partial))
            messages_list = messages_list[-50:]

            # 5) 最终返回（Gradio 需要把新的 state 传回）
            yield "", chat_hist, messages_list

        # ---------------  清空函数 -------------------------------
        def clear_history():
            return [], "", []  # 清空 Chatbot、输入框、messages_list

        # ---------------  事件绑定 ------------------------------
        msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])
        submit.click(respond, [msg, chatbot, state], [msg, chatbot, state])
        clear.click(clear_history, outputs=[chatbot, msg, state])

    return demo


# ──────────────────────────────────────────────
# 3. 启动应用
# ──────────────────────────────────────────────
demo = create_chatbot()
demo.launch(server_name="0.0.0.0", share=False, debug=True, css=CSS)
