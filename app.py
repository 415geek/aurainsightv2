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
import openai


# ============================
# CONFIG
# ============================
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
YELP_API_KEY = os.getenv("YELP_API_KEY")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

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
    return requests.get(url, params=params).json()["results"]

# ============================
# YELP
# ============================
def yelp_match(name, lat, lng):
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {"term": name, "latitude": lat, "longitude": lng, "limit": 3}
    return requests.get(url, headers=headers, params=params).json()["businesses"]

# ============================
# METEOSTAT
# ============================
def get_weather(lat, lng, days=30):
    loc = Point(lat, lng)
    end = datetime.now()
    start = end - timedelta(days=days)
    df = Daily(loc, start, end).fetch()
    return df.reset_index()

# ============================
# NOAA CURRENT
# ============================
def noaa_forecast(lat, lng):
    url = f"https://api.weather.gov/points/{lat},{lng}"
    meta = requests.get(url).json()
    forecast_url = meta["properties"]["forecast"]
    return requests.get(forecast_url).json()

# ============================
# CENSUS
# ============================
def census_data(lat, lng):
    # simplified demo
    return {
        "population_est": "40,000–60,000",
        "asian_ratio": "40%–55%",
        "median_income": "$90k–$110k"
    }

# ============================
# AI REPORT ENGINE
# ============================
def generate_report(data, lang="zh"):
    prompt = f"""
你是商业分析咨询AI，请严格按照以下样板逻辑生成报告风格：

【样板参考】:
{STYLE_CONTEXT[:6000]}

【真实数据】:
{json.dumps(data, ensure_ascii=False, indent=2)}

要求：
- 报告结构必须与样板一致
- 标注 [FACT] [ASSUMPTION] [INFERENCE] [STRATEGY]
- 逻辑必须像麦肯锡顾问
- 风格必须专业、冷静、数据驱动
- 输出语言：{lang}
"""

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# ============================
# PDF EXPORT
# ============================
def export_pdf(text, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split("\n"):
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:120])
        y -= 14
    c.save()

# ============================
# STREAMLIT UI
# ============================
st.title("AuraInsight · 商圈与增长分析系统")

# 1. 搜索与选择
address_input = st.text_input("请输入餐厅地址", placeholder="例如：2406 19th Ave, San Francisco")

if address_input:
    # 搜索逻辑
    if "last_query" not in st.session_state or st.session_state.last_query != address_input:
        st.session_state.search_results = google_search(address_input)
        st.session_state.last_query = address_input
    
    results = st.session_state.get("search_results", [])

    if not results:
        st.warning("未找到匹配的餐厅，请尝试更详细的地址。")
    else:
        options = [f"{r['name']} | {r['formatted_address']}" for r in results]
        # 使用 key 保持 selectbox 状态
        idx = st.selectbox("请确认匹配的商家", range(len(options)), format_func=lambda i: options[i], key="selected_idx")
        
        if idx is not None:
            place = results[idx]
            
            # 2. 拉取数据 (使用 Session State 防止重复拉取)
            # 只有当选中的地点发生变化时，才重新拉取数据
            if "current_place_id" not in st.session_state or st.session_state.current_place_id != place["place_id"]:
                with st.spinner("正在拉取多维商业数据 (Google/Yelp/Weather/Census)..."):
                    lat = place["geometry"]["location"]["lat"]
                    lng = place["geometry"]["location"]["lng"]
                    
                    yelp_data = yelp_match(place["name"], lat, lng)
                    weather_hist = get_weather(lat, lng)
                    noaa = noaa_forecast(lat, lng)
                    census = census_data(lat, lng)
                    
                    st.session_state.fetched_data = {
                        "place": place,
                        "yelp": yelp_data,
                        "weather_history": weather_hist.tail(10).to_dict(),
                        "noaa_forecast": noaa,
                        "census": census
                    }
                    st.session_state.current_place_id = place["place_id"]
                    # 清除旧的报告
                    if "report_content" in st.session_state:
                        del st.session_state.report_content

            # 3. 显示商家概要
            data = st.session_state.fetched_data
            
            st.divider()
            st.subheader("📊 商家数据概要")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Google 评分**: {place.get('rating', 'N/A')} ({place.get('user_ratings_total', 0)} 条)")
            with col2:
                yelp_rating = data['yelp'][0]['rating'] if data['yelp'] else "N/A"
                yelp_count = data['yelp'][0]['review_count'] if data['yelp'] else 0
                st.error(f"**Yelp 评分**: {yelp_rating} ({yelp_count} 条)")
            with col3:
                st.success(f"**人口概况**: {data['census']['population_est']}")

            with st.expander("查看详细原始数据"):
                st.json(data)

            st.divider()

            # 4. 深度分析按钮
            col_btn, col_lang = st.columns([1, 1])
            with col_lang:
                lang = st.selectbox("报告语言", ["zh", "en"], key="report_lang")
            
            with col_btn:
                if st.button("🔍 开始深度分析 (生成报告)", type="primary"):
                    with st.spinner("AI 顾问正在根据所有数据点生成策略报告，请稍候..."):
                        report = generate_report(data, lang)
                        st.session_state.report_content = report
            
            # 5. 可编辑报告与导出
            if "report_content" in st.session_state:
                st.subheader("📝 深度分析报告 (可编辑)")
                
                # 用户可以在这里修改报告，修改后的内容会被返回给 user_edited_report
                user_edited_report = st.text_area(
                    "您可以直接修改下方的报告内容，修改后点击下载即可。",
                    value=st.session_state.report_content,
                    height=600
                )
                
                if st.button("📥 导出 PDF 分析报告"):
                    export_pdf(user_edited_report, "analysis_report.pdf")
                    with open("analysis_report.pdf", "rb") as pdf_file:
                        st.download_button(
                            label="点击下载 PDF",
                            data=pdf_file,
                            file_name="AuraInsight_Report.pdf",
                            mime="application/pdf"
                        )
                    st.success("PDF 已生成并准备下载！")
