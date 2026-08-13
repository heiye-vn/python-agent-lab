import os
import sys
import uuid
import warnings
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

# 过滤不必要的警告信息
warnings.filterwarnings("ignore")

# 设置 Matplotlib 后端为 Agg（仅文件输出模式，适用于服务器环境）
matplotlib.use("Agg")

# 强制系统标准输出编码为 UTF-8，防止 Windows 终端环境打印中文乱码
sys.stdout.reconfigure(encoding="utf-8")

# ==========================================
# 1. 现代化 LangChain / LangGraph 依赖导入
# ==========================================
# 使用 LangChain 0.2.7+ / 0.3+ 推荐的统一模型工厂函数
from langchain.chat_models import init_chat_model

# 导入 LangChain Core 标准消息对象，用于构建多轮对话上下文
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 导入 LangGraph 预构建的 ReAct Agent（替代已过时的 AgentExecutor）
from langgraph.prebuilt import create_react_agent

# 内置代码解析执行工具（从 experimental 包中导入）
from langchain_experimental.tools import PythonAstREPLTool

# 加载同目录下的 .env 配置文件
load_dotenv(Path(__file__).parent / ".env")


# ==========================================
# 2. LLM 与环境配置
# ==========================================
@st.cache_resource
def init_llm():
    """
    初始化大语言模型 (LLM)
    使用 init_chat_model 工厂函数适配 OpenAI 兼容 API 接口（如阿里云百炼 DashScope）
    """
    api_key = os.getenv("ALI_BAILIAN_API_KEY")
    base_url = os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if not api_key:
        st.error("⚠️ 未检测到 ALI_BAILIAN_API_KEY，请检查 .env 配置文件！")

    return init_chat_model(
        model="qwen3.7-plus",
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,  # 数据分析需要高确定性，降低随机性
    )


# ==========================================
# 3. 会话状态初始化 (Streamlit Session State)
# ==========================================
def init_session_state():
    """初始化 Streamlit 会话持久化状态"""
    if "csv_messages" not in st.session_state:
        # csv_messages 存储聊天记录: [{"role": "user"/"assistant", "content": "...", "type": "text"/"image", "img_path": "..."}]
        st.session_state.csv_messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None


# ==========================================
# 4. 核心 Agent 分析响应逻辑
# ==========================================
# 确保图表输出目录 py_output 存在（没有则自动创建）
OUTPUT_DIR = Path(__file__).parent / "py_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_csv_response(query: str) -> dict:
    """
    处理用户查询的主逻辑函数
    使用 LangGraph 驱动的 ReAct Agent，支持多轮记忆透传与图表隔离保存
    """
    if st.session_state.df is None:
        return {"type": "text", "content": "请先上传 CSV 数据文件"}

    # 再次确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = init_llm()

    # 【重要安全注释】：
    # PythonAstREPLTool 会在宿主服务器直接执行 Python 代码。
    # 生产环境中需确保此工具运行在 Docker 或独立代码隔离沙箱（如 E2B）中。
    locals_dict = {"df": st.session_state.df, "plt": plt}
    tools = [PythonAstREPLTool(locals=locals_dict)]

    # 生成预览结构数据，便于 LLM 快速认知表格模式
    df_head_str = st.session_state.df.head().to_markdown()

    # 将图表保存至 py_output 文件夹下，使用动态 UUID 标识防止同名覆盖
    unique_plot_name = f"plot_{uuid.uuid4().hex[:8]}.png"
    relative_plot_path = f"py_output/{unique_plot_name}"

    # 系统级提示词定义
    system_prompt = f"""你是一名专业的数据分析师。给定一个 pandas DataFrame 变量 `df`。
以下是 `df.head()` 的预览数据：
```markdown
{df_head_str}
```

处理要求：
1. 你可以直接编写并执行 Python 代码对 `df` 进行分析计算或回答用户问题。
2. 如果用户要求生成或绘制图表：
   - 必须使用 matplotlib 保存图片至指定文件路径：`py_output/{unique_plot_name}` （无需自己 mkdir，该目录已存在）。
   - 绘图代码执行完毕后必须调用 `plt.close()` 清理全局画布，防止后续图表重叠！
   - 在你的最终回答结尾，必须附加标记：`GRAPH:{relative_plot_path}`。
3. 一旦获取到足够数据或分析结果，请立即给出清晰的文字回答或分析结论。
"""

    # 构建 LangChain 格式的历史消息列表（透传对话上下文，实现多轮记忆）
    history_messages = [SystemMessage(content=system_prompt)]

    # 将 Streamlit 历史记录转换为标准的 LangChain Message 格式
    for msg in st.session_state.csv_messages:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg["type"] == "text":
            history_messages.append(AIMessage(content=str(msg["content"])))

    # 写入当前最新的用户查询
    history_messages.append(HumanMessage(content=query))

    try:
        # 【现代化 API 替换】：
        # 使用 LangGraph 的 create_react_agent 替代已过时的 AgentExecutor
        agent_executor = create_react_agent(model=llm, tools=tools)

        # 执行 Agent 图计算
        response = agent_executor.invoke({"messages": history_messages})

        # 获取最后一轮 Agent 输出的最终文本
        final_message = response["messages"][-1].content

        # 判断是否包含绘图输出
        if "GRAPH:" in final_message:
            # 提取生成的具体图表文件名
            graph_filename = final_message.split("GRAPH:")[1].strip()
            clean_text = final_message.split("GRAPH:")[0].strip()

            return {
                "type": "image",
                "content": clean_text if clean_text else "图表生成完成：",
                "img_path": graph_filename,
            }
        else:
            return {"type": "text", "content": final_message}

    except Exception as e:
        return {"type": "text", "content": f"❌ 分析计算过程中发生错误: {str(e)}"}


# ==========================================
# 5. Streamlit 主界面渲染逻辑
# ==========================================
def main():
    # 页面宽屏配置与全局设置（必须在主界面最顶部调用）
    st.set_page_config(
        page_title="LangChain v1 智能数据分析系统",
        page_icon="🤖",
        layout="wide",  # 开启宽屏模式，充分利用屏幕左右两侧空间
    )

    # 注入 Custom CSS 样式：规范内边距，确保标题完整展示，隐藏无用页脚
    st.markdown(
        """
        <style>
        /* 规范顶部与两侧内边距，防止标题被顶部视口切掉 */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        /* 隐藏无用的默认页脚 */
        footer { visibility: hidden; }
        /* 规范标题边距 */
        h2 { margin-top: 0rem !important; margin-bottom: 0.4rem !important; }
        /* 优化展开面板的外边距 */
        div[data-testid="stExpander"] { margin-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # 应用标题与描述
    st.markdown("## 🤖 LangChain v1 智能数据分析系统")
    st.markdown("基于 **LangGraph ReAct Agent** + **Pandas** 构建的现代化数据分析助手")

    # 宽屏比例划分：左侧 7（对话区），右侧 3（数据面板区）
    col1, col2 = st.columns([7, 3])

    with col1:
        st.markdown("### 📈 数据分析对话")

        # 数据状态指示卡片（固定在对话容器上方）
        if st.session_state.df is not None:
            st.success("✅ 数据加载成功，随时可以发起提问。")
        else:
            st.warning("⚠️ 请在右侧面板上传 CSV 文件后开始分析。")

        # 核心滚动对话容器：精确设定高度为 420px，保证内部独立纵向滚动且整体不触发浏览器外层滚动条
        chat_container = st.container(height=420, border=True)
        with chat_container:
            # 渲染历史对话记录
            for message in st.session_state.csv_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # 只有当前消息有对应的独立图片文件且存在时才渲染
                    if message.get("type") == "image" and message.get("img_path"):
                        if os.path.exists(message["img_path"]):
                            st.image(message["img_path"])

        # 处理用户输入（固定在页面底部/容器下方）
        if csv_query := st.chat_input(
            "📊 输入对 CSV 数据的分析指令...", disabled=st.session_state.df is None
        ):
            # 追加用户提问到 session_state
            st.session_state.csv_messages.append(
                {"role": "user", "content": csv_query, "type": "text"}
            )

            # 在对话滚动容器内部呈现最新输入与回答
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(csv_query)

                with st.chat_message("assistant"):
                    with st.spinner("🔄 Agent 正在分析数据与编写代码..."):
                        res = get_csv_response(csv_query)

                    # 渲染助手回答
                    st.markdown(res["content"])
                    if res["type"] == "image" and os.path.exists(res.get("img_path", "")):
                        st.image(res["img_path"])

            # 保存助手回答到 session_state
            st.session_state.csv_messages.append(
                {
                    "role": "assistant",
                    "content": res["content"],
                    "type": res["type"],
                    "img_path": res.get("img_path", ""),
                }
            )

    with col2:
        st.markdown("### 📊 数据管理面板")

        # CSV 文件上传器
        csv_file = st.file_uploader("📈 上传 CSV 数据文件", type="csv")
        if csv_file:
            # 校验是否是新文件，或者当前的 df 是否尚未加载
            if (
                st.session_state.df is None
                or st.session_state.get("uploaded_filename") != csv_file.name
            ):
                try:
                    # 增加常用中文编码解析 fallback
                    try:
                        new_df = pd.read_csv(csv_file, encoding="utf-8")
                    except UnicodeDecodeError:
                        csv_file.seek(0)
                        new_df = pd.read_csv(csv_file, encoding="gbk")

                    st.session_state.df = new_df
                    st.session_state.uploaded_filename = csv_file.name
                    # 关键修复：解析新文件后立即重新渲染全页，使 col1 输入框能感知 df 已加载并恢复启用！
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 读取 CSV 文件失败: {str(e)}")

            # 显示数据解析成功与预览
            st.success("✅ 文件解析成功！")
            with st.expander("👀 数据预览与结构", expanded=True):
                st.dataframe(st.session_state.df.head())
                st.info(
                    f"维度: **{st.session_state.df.shape[0]}** 行 × **{st.session_state.df.shape[1]}** 列"
                )

        # 显示数据列详细信息
        if st.session_state.df is not None:
            if st.button("📋 显示数据列详情", use_container_width=True):
                with st.expander("📊 数据类型分布", expanded=True):
                    dtype_df = pd.DataFrame(
                        {
                            "列名": st.session_state.df.columns,
                            "数据类型": [
                                str(dtype) for dtype in st.session_state.df.dtypes
                            ],
                            "缺失值数量": [
                                int(val) for val in st.session_state.df.isnull().sum()
                            ],
                        }
                    )
                    st.dataframe(dtype_df, use_container_width=True)

        # 清除数据及临时生成的文件
        if st.button("🗑️ 清空数据与聊天历史", use_container_width=True):
            # 删除临时生成的图表图片
            for msg in st.session_state.csv_messages:
                if msg.get("type") == "image" and msg.get("img_path"):
                    if os.path.exists(msg["img_path"]):
                        try:
                            os.remove(msg["img_path"])
                        except Exception:
                            pass

            st.session_state.df = None
            st.session_state.uploaded_filename = None
            st.session_state.csv_messages = []
            st.success("数据与临时文件已清理！")
            st.rerun()


if __name__ == "__main__":
    main()
