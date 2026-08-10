import os
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import (
    PythonAstREPLTool,
)  # 从LangChain依赖库引入Python代码解释器
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser

load_dotenv(Path(__file__).parent / ".env")

csv_path = Path(__file__).parent / "global_cities_data.csv"
df = pd.read_csv(csv_path)
tool = PythonAstREPLTool(
    locals={"df": df}
)  # 传递给代码解释器的局部变量，这里是读取表格内容的pandas对象

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

# 添加提示词模板
system = f"""
你可以访问一个名为 `df` 的 pandas 数据框，你可以使用df.head().to_markdown() 查看数据集的基本信息， \
请根据用户提出的问题，编写 Python 代码来回答。只返回代码，不返回其他内容。只允许使用 pandas 和内置库。
"""

prompt = ChatPromptTemplate([("system", system), ("user", "{question}")])


parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

llm_with_tools = model.bind_tools([tool])  # 将工具与大模型绑定

llm_chain = prompt | llm_with_tools | parser | tool

response = llm_chain.invoke(
    "我有一张表，名为'df'，请帮我计算GDP_Billion_USD的最大值是多少，并输出对应的国家。"
)

print(response)
