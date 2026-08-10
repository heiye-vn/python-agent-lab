import os
from pathlib import Path
import requests

from pypinyin import lazy_pinyin
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)


def city_to_pinyin(city_name):
    """
    将中文城市名转换为拼音 (不带声调)
    例如: '成都' -> 'chengdu', '西安' -> 'xian'
    """
    # lazy_pinyin 返回的是一个列表 ['cheng', 'du']
    # ''.join() 把它拼成字符串 'chengdu'
    pinyin_list = lazy_pinyin(city_name)
    return "".join(pinyin_list)


@tool
def get_weather(city_input: str) -> str:
    """查询指定城市或区县的实时天气信息。

    Args:
        city_input: 城市或区县名称，例如 '上海', '北京', '红塔区', '海淀区'
    """
    print(f"正在查询: {city_input} ...")

    # 1. 和风天气 GeoAPI 原生支持中文城市/区县搜索（如 '红塔区'、'海淀区'、'朝阳区'），无需转拼音
    location_query = city_input.strip()

    # 2. 获取 API KEY 和 API HOST
    api_key = os.getenv("HF_WEATHER_API_KEY")
    api_host = os.getenv("HF_WEATHER_HOST", "devapi.qweather.com").rstrip("/")

    if not api_key:
        return "[!] 错误: 未设置 HF_WEATHER_API_KEY 环境变量"

    # 3. 第一步：通过 GeoAPI 获取 Location ID
    if "geoapi.qweather.com" in api_host:
        geo_url = f"https://geoapi.qweather.com/v2/city/lookup?location={location_query}&key={api_key}"
    else:
        geo_url = f"https://{api_host}/geo/v2/city/lookup?location={location_query}&key={api_key}"

    try:
        geo_resp = requests.get(geo_url)
        if geo_resp.status_code != 200:
            return f"[!] GeoAPI 请求失败 (HTTP {geo_resp.status_code}): {geo_resp.text}"

        geo_data = geo_resp.json()

        if geo_data.get("code") == "200" and len(geo_data.get("location", [])) > 0:
            # 获取第一个匹配结果
            city_info = geo_data["location"][0]
            loc_id = city_info["id"]
            real_name = city_info["name"]
            adm1 = city_info.get("adm1", "")  # 省
            adm2 = city_info.get("adm2", "")  # 市

            full_location_name = f"{adm1} {adm2} {real_name}".strip()
            print(f"-> 匹配到位置: {full_location_name} (ID: {loc_id})")

            # 4. 第二步：查询实时天气
            if "devapi.qweather.com" in api_host:
                weather_url = f"https://devapi.qweather.com/v7/weather/now?location={loc_id}&key={api_key}"
            else:
                weather_url = (
                    f"https://{api_host}/v7/weather/now?location={loc_id}&key={api_key}"
                )

            weather_resp = requests.get(weather_url)
            if weather_resp.status_code != 200:
                return f"[!] WeatherAPI 请求失败 (HTTP {weather_resp.status_code}): {weather_resp.text}"

            weather_data = weather_resp.json()

            if weather_data.get("code") == "200":
                now = weather_data["now"]
                result = (
                    f"位置: {full_location_name}\n"
                    f"天气: {now['text']}\n"
                    f"温度: {now['temp']}°C\n"
                    f"湿度: {now['humidity']}%\n"
                    f"风向: {now['windDir']} {now['windScale']}级"
                )
                return result
            else:
                return f"天气数据获取失败，错误码: {weather_data.get('code')}"

        else:
            return f"[!] 未找到城市 '{city_input}'"

    except Exception as e:
        return f"[!] 网络请求出错: {e}"


# 绑定工具到大模型
llm_with_tools = model.bind_tools([get_weather])


if __name__ == "__main__":
    query = "请问上海今天天气如何？"
    print(f"用户提问: {query}")
    
    # 1. 大模型分析是否需要调用工具
    ai_msg = llm_with_tools.invoke(query)
    print("\n--- 大模型响应工具调用请求 ---")
    print(ai_msg.tool_calls)

    # 2. 若模型触发了工具调用，执行工具获取数据
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"\n--- 执行工具: {tool_call['name']} ---")
            tool_output = get_weather.invoke(tool_call["args"])
            print("工具返回内容:")
            print(tool_output)
