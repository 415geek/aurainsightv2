import os
import json
import math
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import requests
import pandas as pd

from meteostat import Point, Daily
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from openai import OpenAI


# ============================
# CONFIG
# ============================
def get_secret(key):
    # 优先尝试从 Streamlit Secrets 读取
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        # 回退到环境变量
        return os.getenv(key)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_MAPS_API_KEY")
YELP_API_KEY = get_secret("YELP_API_KEY")
CENSUS_API_KEY = get_secret("CENSUS_API_KEY")

PDF_STYLE_FILES = [
    "data/Aurainsight门店分析【东南风美食】.txt",
    "data/样本3.txt"
]



try:
    from meteostat import Point, Daily
except Exception as e:
    import streamlit as st
    st.error(f"Missing dependency: meteostat. Please check requirements.txt. Error: {e}")
    st.stop()
# ============================
# PDF STYLE LOADER
# ============================
def load_pdf_text(path):
    if not os.path.exists(path):
        st.warning(f"Warning: Style file not found: {path}. Skipping.")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

STYLE_CONTEXT = "\n".join([load_pdf_text(p) for p in PDF_STYLE_FILES])

# ============================
# GOOGLE PLACE
# ============================
def google_search(query):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_API_KEY}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("status") == "OK":
            return data["results"]
        else:
            # 返回错误状态以便调试
            return [{"error": f"Google API Error: {data.get('status')} - {data.get('error_message', '')}"}]
    except Exception as e:
        return [{"error": f"Request failed: {str(e)}"}]

# ... (中间代码保持不变) ...

# ============================
# AI REPORT ENGINE
# ============================
def generate_report(data, lang="zh"):
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 准备变量
    restaurant_name = data.get("place", {}).get("name", "Unknown Restaurant")
    restaurant_address = data.get("place", {}).get("formatted_address", "Unknown Address")
    input_data_str = json.dumps(data, ensure_ascii=False, indent=2)

    try:
        # 使用用户提供的 Prompt Template ID 调用
        # 注意：这通常需要特定的 OpenAI 库版本支持 'responses' 端点
        response = client.responses.create(
            prompt={
                "id": "pmpt_6971b3bd094081959997af7730098d45020d02ec1efab62b",
                "version": "2",
                "variables": {
                    "restaurant_name": restaurant_name,
                    "restaurant_address": restaurant_address,
                    "input_data": input_data_str
                }
            }
        )
        # 尝试标准返回结构
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            # 如果返回结构不同，尝试直接返回
            return str(response)

    except AttributeError:
        # 如果当前环境的 OpenAI 库不支持 client.responses
        return "❌ Error: 您的 OpenAI Python 库版本可能不支持 `client.responses.create`。请确认这是否为 Beta 功能或需要特定版本。"
    except Exception as e:
        return f"❌ 生成报告时发生错误: {str(e)}"

# ... (中间代码保持不变) ...

# ============================
# STREAMLIT UI
# ============================
st.title("AuraInsight · 商圈与增长分析系统")

# 1. 搜索与选择
address_input = st.text_input("请输入餐厅地址", placeholder="例如：2406 19th Ave, San Francisco")

if address_input:
    # 搜索逻辑
    if "last_query" not in st.session_state or st.session_state.last_query != address_input:
        # 自动追加 "restaurant" 以确保搜索的是商家而不是纯地址
        search_query = f"{address_input} restaurant"
        st.session_state.search_results = google_search(search_query)
        st.session_state.last_query = address_input
    
    results = st.session_state.get("search_results", [])

    # 错误处理逻辑
    if not results:
        st.warning("未找到匹配的餐厅，请尝试更详细的地址。")
    elif isinstance(results[0], dict) and "error" in results[0]:
        st.error(results[0]["error"])
    else:
        options = [f"{r['name']} | {r['formatted_address']}" for r in results]
        # 使用 key 保持 selectbox 状态
        idx = st.selectbox("请确认匹配的商家", range(len(options)), format_func=lambda i: options[i], key="selected_idx")
        
        if idx is not None:
            place = results[idx]
            # ... (后续代码) ...
