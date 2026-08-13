import warnings

warnings.filterwarnings("ignore")

import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# 加载同目录下的 .env 配置文件
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
from pypdf import PdfReader

# LangChain 1.x 最新规范依赖导入
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent

sys.stdout.reconfigure(encoding="utf-8")

# 环境变量与模型初始化 (兼容通义千问 / DashScope OpenAI 模式)
api_key = os.getenv("ALI_BAILIAN_API_KEY")
base_url = os.getenv(
    "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

llm = ChatOpenAI(
    model="qwen3.7-max-2026-05-20",
    api_key=api_key,
    base_url=base_url,
    temperature=0.2,
)

embeddings = DashScopeEmbeddings(
    model="qwen3.7-text-embedding",
    dashscope_api_key=api_key,
)

FAISS_DB_DIR = "faiss_db"


# PDF 文本提取
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# 文本切分 (Chunking)
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


# 向量数据库构建 (FAISS)
def create_vector_store(text_chunks, batch_size=16):
    db = None
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i : i + batch_size]
        if db is None:
            db = FAISS.from_texts(batch, embedding=embeddings)
        else:
            db.add_texts(batch)
    if db:
        db.save_local(FAISS_DB_DIR)


# 检查数据库状态
def check_db_ready():
    return os.path.exists(FAISS_DB_DIR) and os.path.exists(
        os.path.join(FAISS_DB_DIR, "index.faiss")
    )


# 基于 LangGraph 1.x 的 RAG Agent 查询引擎
def query_rag_agent(user_question):
    db = FAISS.load_local(
        FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})

    retrieval_tool = create_retriever_tool(
        retriever,
        "pdf_extractor",
        "Search and extract knowledge from uploaded PDF documents to answer user queries.",
    )

    system_prompt = """你是智能AI助手。请优先使用 `pdf_extractor` 工具检索上传的文档内容来回答用户问题。
1. 如果检索到的文档内容包含相关答案，请结合文档详细回答。
2. 如果检索到的文档内容不包含答案或问题与文档无关，请使用你自身的通用知识回答，并附带说明“（注：该答案基于 AI 通用知识，上传的文档中未包含此内容）”。"""

    # LangChain 1.x 现代化 create_agent
    graph_agent = create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=system_prompt,
    )

    result = graph_agent.invoke({"messages": [("user", user_question)]})
    final_output = result["messages"][-1].content
    return final_output, result["messages"]


# Streamlit 主界面
def main():
    st.set_page_config(
        page_title="PDF RAG 智能检索系统 (v1.x)",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 注入现代 CSS 样式 (区分旧版 UI)
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .tech-badge {
            background-color: rgba(255, 255, 255, 0.2);
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        .status-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header 渲染
    st.markdown(
        """
        <div class="main-header">
            <div class="tech-badge">⚡ LangChain 1.x & LangGraph 1.x</div>
            <h1 style="margin: 0; font-size: 2rem;">🧠 智能 PDF 知识库 RAG 系统</h1>
            <p style="margin-top: 0.5rem; opacity: 0.9;">基于最新 React Agent 引擎与向量检索构建架构</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # 初始化 Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 侧边栏：文档与索引控制
    with st.sidebar:
        st.header("⚙️ 文档控制台")

        db_ready = check_db_ready()
        if db_ready:
            st.success("🟢 向量数据库状态：已就绪")
        else:
            st.warning("🟡 向量数据库状态：未初始化")

        st.divider()

        st.subheader("📎 上传文档")
        pdf_docs = st.file_uploader(
            "选择 PDF 文件",
            accept_multiple_files=True,
            type=["pdf"],
            help="可同时选定并上传多个 PDF 文档",
        )

        if pdf_docs:
            st.caption(f"已选择 {len(pdf_docs)} 个文件：")
            for doc in pdf_docs:
                st.caption(f"• {doc.name}")

        process_btn = st.button(
            "🚀 开始构建向量索引",
            disabled=not pdf_docs,
            use_container_width=True,
            type="primary",
        )

        if process_btn and pdf_docs:
            with st.spinner("⏳ 正在提取与构建向量数据库..."):
                try:
                    raw_text = extract_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("❌ 未读取到有效文本内容！")
                    else:
                        chunks = chunk_text(raw_text)
                        create_vector_store(chunks)
                        st.success(f"✅ 处理完成！生成 {len(chunks)} 个片段。")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ 解析失败: {e}")

        st.divider()

        # 清除数据库操作
        if st.button("🗑️ 清空向量索引", use_container_width=True):
            if os.path.exists(FAISS_DB_DIR):
                shutil.rmtree(FAISS_DB_DIR)
                st.session_state.chat_history = []
                st.success("向量数据库已清理。")
                st.rerun()
            else:
                st.info("数据库已为空。")

    # 主功能区 Tabs
    tab_chat, tab_info = st.tabs(["💬 智能问答对话", "📊 知识库状态与检索详情"])

    with tab_chat:
        if not db_ready:
            st.info("👈 请先在左侧边栏上传 PDF 并点击【开始构建向量索引】")
        else:
            # 渲染历史对话记录
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # 聊天输入框
            user_query = st.chat_input("向知识库提问，例如：文档的核心观点是什么？")

            if user_query:
                # 显示用户消息
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_query}
                )
                with st.chat_message("user"):
                    st.markdown(user_query)

                # 生成回答
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Agent 正在调用工具检索并思考中..."):
                        try:
                            answer, steps = query_rag_agent(user_query)
                            st.markdown(answer)

                            # 存放助理回答
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer}
                            )
                        except Exception as e:
                            st.error(f"❌ 查询过程出现异常: {e}")

    with tab_info:
        st.subheader("📌 数据库索引配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("核心框架", "LangChain 1.x")
        with col2:
            st.metric("Agent 引擎", "LangGraph React")
        with col3:
            st.metric("向量引擎", "FAISS Local")

        st.markdown("---")
        st.markdown("##### 💡 技术架构优势与特性")
        st.markdown(
            """
        - **真正无 Warning 整合**：基于 `langchain-openai` 官方 partner 包连接阿里云 DashScope 通义千问。
        - **LangChain 1.x 标准响应**：采用 `create_agent` 代理范式，具备更强大的图控制扩展性。
        - **全新 Chat UI 体验**：使用 Streamlit 现代的原生 `st.chat_message` 对话组件，交互流畅细腻。
        """
        )


if __name__ == "__main__":
    main()
