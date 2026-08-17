import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
import json
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 文件中的环境变量
load_dotenv(Path(__file__).parent / ".env")

# SQLite 数据库文件路径
DB_PATH = Path(__file__).parent / "telco.db"


def ensure_db_exists():
    """如果数据库文件不存在，自动调用 init_database 初始化"""
    if not DB_PATH.exists():
        from init_db import init_database
        init_database()


ensure_db_exists()

# ----------------------------------------------------
# 1. SQL 查询工具（SQLite 版）
# ----------------------------------------------------
description_sql = """
当用户需要进行数据库查询工作时，请调用该函数。
该函数用于在本地 SQLite 数据库 (telco.db) 上运行 SQL 代码完成查询，
包含的表有：customer_info(客户基础信息，含姓名 customer_name)、customer_services(服务开通信息)、customer_churn(合约与流失信息)。
本函数只负责运行SQL代码并进行数据查询返回结果，若要将数据表提取到Python变量中，请使用 extract_data 函数。
"""


class SQLQuerySchema(BaseModel):
    sql_query: str = Field(description=description_sql)


@tool(args_schema=SQLQuerySchema)
def sql_inter(sql_query: str) -> str:
    """
    当用户需要进行数据库查询工作时，请调用该函数。
    该函数用于在本地 SQLite 数据库上运行一段 SQL 代码，完成数据查询相关工作。
    :param sql_query: 字符串形式的 SQL 查询语句
    :return: sql_query 在 SQLite 中的运行结果（JSON 字符串形式）
    """
    ensure_db_exists()
    try:
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row  # 支持字典形式获取结果
        with connection:
            cursor = connection.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
        connection.close()
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"SQL执行失败: {e}"


# ----------------------------------------------------
# 2. 数据提取工具（SQLite 版）
# ----------------------------------------------------
class ExtractQuerySchema(BaseModel):
    sql_query: str = Field(description="用于从 SQLite 提取数据的 SQL 查询语句。")
    df_name: str = Field(
        description="指定用于保存结果的 pandas 变量名称（字符串形式）。"
    )


@tool(args_schema=ExtractQuerySchema)
def extract_data(sql_query: str, df_name: str) -> str:
    """
    用于在 SQLite 数据库中提取数据表到当前 Python 内存环境中（存为 pandas DataFrame）。
    注意：本函数只负责数据提取并存为全局变量，若只需查询结果请使用 sql_inter。
    :param sql_query: 字符串形式的 SQL 查询语句
    :param df_name: 保存到本地环境的 pandas 变量名（字符串）
    :return: 提取与保存结果提示
    """
    ensure_db_exists()
    try:
        connection = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, connection)
        connection.close()
        globals()[df_name] = df
        return f"成功创建 pandas DataFrame `{df_name}`，包含 {len(df)} 行数据，列名为: {list(df.columns)}"
    except Exception as e:
        return f"数据提取执行失败: {e}"


# ----------------------------------------------------
# 3. Python 代码执行工具（REPL）
# ----------------------------------------------------
class PythonCodeInput(BaseModel):
    py_code: str = Field(
        description="一段合法的 Python 代码字符串，例如 'df.describe()' 或 'x = df['tenure'].mean()'"
    )


@tool(args_schema=PythonCodeInput)
def python_inter(py_code: str) -> str:
    """
    当用户需要执行 Python 程序并进行数据统计计算时，请调用该函数。
    该函数可以执行一段 Python 代码并返回结果。注意：本函数只能执行非绘图类代码，绘图请调用 fig_inter。
    """
    g = globals()
    try:
        # 尝试作为表达式求值
        return str(eval(py_code, g))
    except Exception:
        global_vars_before = set(g.keys())
        try:
            exec(py_code, g)
        except Exception as e:
            return f"代码执行时报错: {e}"
        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before
        if new_vars:
            result = {var: str(g[var]) for var in new_vars}
            return str(result)
        else:
            return "代码已顺利执行"


# ----------------------------------------------------
# 4. 可视化绘图工具
# ----------------------------------------------------
class FigCodeInput(BaseModel):
    py_code: str = Field(
        description="要执行的 Python 绘图代码，必须使用 matplotlib/seaborn 创建图像并赋值给变量"
    )
    fname: str = Field(
        description="图像对象的变量名，例如 'fig'，用于从代码中提取并保存为图片"
    )


@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """
    当用户需要使用 Python 进行可视化绘图任务时，请调用该函数。

    注意：
    1. 所有绘图代码必须创建一个图像对象，并将其赋值为指定变量名（例如 `fig`）。
    2. 必须使用 `fig = plt.figure()` 或 `fig, ax = plt.subplots()`。
    3. 不要使用 `plt.show()`。
    4. 请确保代码最后调用 `fig.tight_layout()`。
    5. 所有绘图代码中，坐标轴标签（xlabel、ylabel）、标题（title）、图例（legend）等文本内容，必须使用英文描述。
    """
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    local_vars = {"plt": plt, "pd": pd, "sns": sns}
    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        g = globals()
        exec(py_code, g, local_vars)
        g.update(local_vars)

        fig = local_vars.get(fname, None)
        if fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{fname}_{timestamp}.png"
            abs_path = images_dir / image_filename
            rel_path = f"images/{image_filename}"

            fig.savefig(str(abs_path), bbox_inches="tight")
            return f"图片已保存，路径为: {rel_path}"
        else:
            return f"未找到名为 `{fname}` 的图像对象，请确认变量名正确。"
    except Exception as e:
        return f"绘图执行失败: {e}"
    finally:
        plt.close("all")
        matplotlib.use(current_backend)


# ----------------------------------------------------
# 5. 系统提示词
# ----------------------------------------------------
prompt = """
你是一名经验丰富的智能数据分析助手，擅长帮助用户高效完成以下任务：

本地数据库为 SQLite (`telco.db`)，包含以下三张表：
1. `customer_info`: 客户基础信息（customer_id, customer_name, gender, senior_citizen, partner, dependents, tenure）
2. `customer_services`: 客户订购服务（customer_id, phone_service, multiple_lines, internet_service, online_security, tech_support, streaming_tv, streaming_movies）
3. `customer_churn`: 客户合约与流失（customer_id, contract, paperless_billing, payment_method, monthly_charges, total_charges, churn）

**工具使用指南与优先级：**
1. **数据库查询：**
   - 当用户需要查询表结构、统计某些指标或查看数据库概况时，调用 `sql_inter` 工具运行 SQLite SQL 语句。
2. **数据表提取：**
   - 当需要进行深度统计、复杂计算或绘图分析时，调用 `extract_data` 工具将所需表提取为 pandas DataFrame（如 `df_telco`）。
3. **Python 数据分析：**
   - 调用 `python_inter` 工具对提取后的 pandas DataFrame 执行统计计算与特征分析（非绘图任务）。
4. **可视化绘图：**
   - 调用 `fig_inter` 工具生成图表（必须给图像对象命名如 `fig`），图表中的标题和标签请使用英文。

**回答要求：**
- 所有回答使用**简体中文**，清晰、专业、数据驱动。
- 如果生成了图片，请务必在回答中以 Markdown 图片语法展示：`![图片描述](images/fig.png)`。
- 不要编造不存在的数据。
"""

# ----------------------------------------------------
# 6. 构建 Agent
# ----------------------------------------------------
tools = [python_inter, fig_inter, sql_inter, extract_data]

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

graph = create_agent(model=llm, tools=tools, system_prompt=prompt)
