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

restaurant_name = st.text_input("输入餐厅名称")
city = st.text_input("输入城市")

if restaurant_name and city:
    query = f"{restaurant_name} in {city}"
    results = google_search(query)
    
    if not results:
        st.warning("未找到匹配的餐厅，请尝试其他名称或城市。")
    else:
        options = [f"{r['name']} | {r['formatted_address']}" for r in results]
        idx = st.selectbox("选择匹配餐厅", range(len(options)), format_func=lambda i: options[i])
        
        if idx is not None:
            place = results[idx]

            lat = place["geometry"]["location"]["lat"]
            lng = place["geometry"]["location"]["lng"]

            if st.button("开始分析"):
                with st.spinner("拉取数据中..."):

                    yelp = yelp_match(place["name"], lat, lng)
                    weather_hist = get_weather(lat, lng)
                    noaa = noaa_forecast(lat, lng)
                    census = census_data(lat, lng)

                    data = {
                        "place": place,
                        "yelp": yelp,
                        "weather_history": weather_hist.tail(10).to_dict(),
                        "noaa_forecast": noaa,
                        "census": census
                    }

                    lang = st.selectbox("输出语言", ["zh", "en"])
                    report = generate_report(data, lang)

                    st.subheader("分析报告")
                    st.text_area("Report", report, height=500)

                    if st.button("导出PDF"):
                        export_pdf(report, "analysis_report.pdf")
                        st.success("PDF 已生成：analysis_report.pdf")
