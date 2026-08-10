import pandas as pd
from langchain_experimental.tools import (
    PythonAstREPLTool,
)

from pathlib import Path

csv_path = Path(__file__).parent / "global_cities_data.csv"
df = pd.read_csv(csv_path)
# 
tool = PythonAstREPLTool(locals={"df": df})
res = tool.invoke("df['GDP_Billion_USD'].mean()")  # 计算变量GDP的均值

print(res)
