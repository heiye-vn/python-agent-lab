import sys
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载同目录下的 .env 配置文件
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st  # 用来快速构建前端页面
from PyPDF2 import PdfReader  # PDF文档读取、处理的依赖库
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from langchain_classic.embeddings import init_embeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import (
    FAISS,
)  # 使用FAISS词向量数据库保存切分后短文档的词向量
from langchain_classic.tools.retriever import (
    create_retriever_tool,
)  # #RAG中的R，把RAG系统中的检索功能封装成工具，提供检索词向量功能
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

sys.stdout.reconfigure(encoding="utf-8")

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

# 初始化向量模型
embeddings = DashScopeEmbeddings(
    model="qwen3.7-text-embedding", dashscope_api_key=os.getenv("ALI_BAILIAN_API_KEY")
)


# 读取pdf上传的内容
def pdf_reader(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks, batch_size=16):
    db = None
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i : i + batch_size]
        if db is None:
            db = FAISS.from_texts(batch, embedding=embeddings)
        else:
            db.add_texts(batch)
    if db:
        db.save_local("faiss_db")


# 构建知识库回答逻辑链
def check_database_exists():
    """检查FAISS数据库是否存在"""
    return os.path.exists("faiss_db") and os.path.exists("faiss_db/index.faiss")


def user_input(user_question):
    if not check_database_exists():
        st.error("❌ 请先上传PDF文件并点击'Submit & Process'按钮来处理文档！")
        st.info("💡 步骤：1️⃣ 上传PDF → 2️⃣ 点击处理 → 3️⃣ 开始提问")
        return

    try:
        # 加载 FASISS 数据库
        new_db = FAISS.load_local(
            "faiss_db", embeddings, allow_dangerous_deserialization=True
        )

        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(
            retriever,
            "pdf_extractor",
            "This tool is to give answer to queries from the pdf",
        )
        get_conversational_chain(retrieval_chain, user_question)

    except Exception as e:
        st.error(f"❌ 加载数据库时出错: {str(e)}")  # 前端界面报错
        st.info("请重新处理PDF文件")  # 前端界面info提示


def get_conversational_chain(tools, querys):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是智能AI助手。请优先使用 `pdf_extractor` 工具检索上传的文档内容来回答用户问题。
1. 如果检索到的文档内容包含相关答案，请结合文档详细回答。
2. 如果检索到的文档内容不包含答案或问题与文档无关，请使用你自身的通用知识回答，并附带说明“（注：该答案基于 AI 通用知识，上传的文档中未包含此内容）”。""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)

    response = agent_executor.invoke({"input": querys})
    print(response)
    st.write("🤖 回答: ", response["output"])


def main():
    st.set_page_config("🤖 LangChain知识库系统开发")
    st.header("🤖 LangChain知识库系统开发")

    # 显示数据库状态
    col1, col2 = st.columns([3, 1])

    with col1:
        if check_database_exists():
            pass
        else:
            st.warning("⚠️ 请先上传并处理PDF文件")

    with col2:
        if st.button("🗑️ 清除数据库"):
            try:
                import shutil

                if os.path.exists("faiss_db"):
                    shutil.rmtree("faiss_db")
                st.success("数据库已清除")
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

    # 用户输入问题
    user_question = st.text_input(
        "💬 请输入问题",
        placeholder="例如：这个文档的主要内容是什么？",
        disabled=not check_database_exists(),
    )

    if user_question:
        if check_database_exists():
            with st.spinner("🤔 AI正在分析文档..."):
                user_input(user_question)
        else:
            st.error("❌ 请先上传并处理PDF文件！")

    # 侧边栏
    with st.sidebar:
        st.title("📁 文档管理")

        # 显示当前状态
        if check_database_exists():
            st.success("✅ 数据库状态：已就绪")
        else:
            st.info("📝 状态：等待上传PDF")

        st.markdown("---")

        # 文件上传
        pdf_doc = st.file_uploader(
            "📎 上传PDF文件",
            accept_multiple_files=True,
            type=["pdf"],
            help="支持上传多个PDF文件",
        )

        if pdf_doc:
            st.info(f"📄 已选择 {len(pdf_doc)} 个文件")
            for i, pdf in enumerate(pdf_doc, 1):
                st.write(f"{i}. {pdf.name}")

        # 处理按钮
        process_button = st.button(
            "🚀 提交并处理", disabled=not pdf_doc, use_container_width=True
        )

        if process_button:
            if pdf_doc:
                with st.spinner("📊 正在处理PDF文件..."):
                    try:
                        # 读取PDF内容
                        raw_text = pdf_reader(pdf_doc)

                        if not raw_text.strip():
                            st.error("❌ 无法从PDF中提取文本，请检查文件是否有效")
                            return

                        # 分割文本
                        text_chunks = get_chunks(raw_text)
                        st.info(f"📝 文本已分割为 {len(text_chunks)} 个片段")

                        # 创建向量数据库
                        vector_store(text_chunks)

                        st.success("✅ PDF处理完成！现在可以开始提问了")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 处理PDF时出错: {str(e)}")
            else:
                st.warning("⚠️ 请先选择PDF文件")

        # 使用说明
        with st.expander("💡 使用说明"):
            st.markdown(
                """
                **步骤：**
                1. 📎 上传一个或多个PDF文件
                2. 🚀 点击"Submit & Process"处理文档
                3. 💬 在主页面输入您的问题
                4. 🤖 AI将基于PDF内容回答问题
    
                **提示：**
                - 支持多个PDF文件同时上传
                - 处理大文件可能需要一些时间
                - 可以随时清除数据库重新开始
                """
            )


if __name__ == "__main__":
    main()
