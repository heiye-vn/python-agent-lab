import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------
# 解决本地系统代理导致的 SSL: UNEXPECTED_EOF_WHILE_READING 报错
# 如果本地开启了代理软件（如 127.0.0.1:7890），可能会阻断对 api.mem0.ai 的 HTTPS 握手
# ---------------------------------------------------------------------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

from mem0 import MemoryClient

api_key = os.getenv("MEM0_API_KEY")
if not api_key:
    raise ValueError("未检测到 MEM0_API_KEY，请检查 code/chapter16/.env 文件配置！")

print("=" * 65)
print("【Mem0 官方云端记忆客户端 (MemoryClient) 测试】")
print("=" * 65)

# 初始化云端客户端
client = MemoryClient(api_key=api_key)

# 1. 写入用户交互消息（Mem0 会自动调用 LLM 提炼出关键事实与偏好）
print("\n1. 正在向 Mem0 写入用户对话事实...")
add_result = client.add(
    [
        {
            "role": "user",
            "content": "你好，我是 Alice，我平时非常喜欢周末去爬山，并且饮食偏好清淡。",
        }
    ],
    user_id="alice",
)
print("写入成功！返回响应:", add_result)

# 2. 语义搜索记忆 (注意：Mem0 v2 最新规范需使用 filters 参数)
print("\n2. 正在执行语义记忆检索 (Query: 'Alice 喜欢做什么?')...")
search_results = client.search("Alice 喜欢做什么?", filters={"user_id": "alice"})

print(f"检索到 {len(search_results.get('results', search_results))} 条相关记忆：")
if isinstance(search_results, dict) and "results" in search_results:
    items = search_results["results"]
else:
    items = search_results

for idx, item in enumerate(items, start=1):
    print(f"  {idx}. [ID: {item.get('id')}] 提炼记忆: {item.get('memory')}")
    print(f"     分类标签: {item.get('categories')}")
    print(f"     创建时间: {item.get('created_at')}\n")

# 3. 获取用户所有长期记忆画像
print("\n3. 获取该用户的全部长期画像 (get_all)...")
all_memories = client.get_all(filters={"user_id": "alice"})
print(
    f"用户 alice 当前累计共有 {all_memories.get('count', len(all_memories))} 条长期记忆。"
)
