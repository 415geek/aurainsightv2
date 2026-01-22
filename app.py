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
from textblob import TextBlob


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

def get_google_reviews(place_id):
    if not GOOGLE_API_KEY:
        return []
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    # 我们只需要 reviews 字段
    params = {
        "place_id": place_id,
        "fields": "reviews",
        "key": GOOGLE_API_KEY,
        "language": "zh-CN" # 尝试获取中文评论，或者根据需求不加此参数获取原语言
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("status") == "OK":
            return data.get("result", {}).get("reviews", [])
        return []
    except Exception:
        return []

def get_google_photo_url(photo_ref, max_width=400):
    if not photo_ref:
        return None
    base_url = "https://maps.googleapis.com/maps/api/place/photo"
    return f"{base_url}?maxwidth={max_width}&photo_reference={photo_ref}&key={GOOGLE_API_KEY}"

# ============================
# YELP
# ============================
def yelp_match(name, lat, lng):
    if not YELP_API_KEY:
        st.error("Yelp API Key is missing.")
        return []
    
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {"term": name, "latitude": lat, "longitude": lng, "limit": 3}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get("businesses", [])
        else:
            st.warning(f"Yelp API returned status: {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Yelp API call failed: {str(e)}")
        return []

def get_yelp_reviews(business_id):
    if not YELP_API_KEY:
        return []
    
    url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("reviews", [])
        return []
    except Exception:
        return []

def analyze_sentiment(reviews):
    if not reviews:
        return {"score": 0, "label": "No Data", "keywords": []}
    
    full_text = " ".join([r.get("text", "") for r in reviews])
    blob = TextBlob(full_text)
    sentiment_score = blob.sentiment.polarity
    
    # Simple labeling
    if sentiment_score > 0.3:
        label = "Positive 😊"
    elif sentiment_score < -0.1:
        label = "Negative 😞"
    else:
        label = "Neutral 😐"
        
    # Extract keywords (simple noun phrases)
    keywords = list(set([w.lower() for w in blob.noun_phrases if len(w) > 3]))[:5]
    
    return {
        "score": sentiment_score,
        "label": label,
        "keywords": keywords
    }


# ============================
# METEOSTAT
# ============================
def get_weather(lat, lng, days=30):
    try:
        loc = Point(lat, lng)
        end = datetime.now()
        start = end - timedelta(days=days)
        df = Daily(loc, start, end).fetch()
        return df.reset_index()
    except Exception as e:
        st.warning(f"Weather data fetch failed: {e}")
        return pd.DataFrame()

# ============================
# NOAA CURRENT
# ============================
def noaa_forecast(lat, lng):
    try:
        url = f"https://api.weather.gov/points/{lat},{lng}"
        # NOAA requires User-Agent
        headers = {"User-Agent": "AuraInsight-App/1.0"}
        meta_resp = requests.get(url, headers=headers)
        if meta_resp.status_code != 200:
            return {}
        
        meta = meta_resp.json()
        forecast_url = meta.get("properties", {}).get("forecast")
        if not forecast_url:
            return {}
            
        forecast_resp = requests.get(forecast_url, headers=headers)
        return forecast_resp.json() if forecast_resp.status_code == 200 else {}
    except Exception:
        return {}

# ============================
# CENSUS
# ============================
def census_data(lat, lng):
    # simplified demo placeholder
    return {
        "population_est": "40,000–60,000",
        "asian_ratio": "40%–55%",
        "median_income": "$90k–$110k"
    }

# ============================
# PDF EXPORT
# ============================
def export_pdf(text, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 40
    # Simple text wrapping logic
    for paragraph in text.split("\n"):
        # Split long lines roughly
        while len(paragraph) > 0:
            line = paragraph[:90] # Approx chars per line
            paragraph = paragraph[90:]
            if y < 40:
                c.showPage()
                y = height - 40
            # Register a font that supports utf-8 if needed, but for now standard
            c.drawString(40, y, line)
            y -= 14
    c.save()

# ... (中间代码保持不变) ...

# ============================
# AI REPORT ENGINE
# ============================
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def generate_report(data, lang="zh"):
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 准备变量
    restaurant_name = data.get("place", {}).get("name", "Unknown Restaurant")
    restaurant_address = data.get("place", {}).get("formatted_address", "Unknown Address")
    # 使用自定义序列化函数处理日期时间对象
    input_data_str = json.dumps(data, ensure_ascii=False, indent=2, default=json_serial)

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
            
            # 2. 确认按钮与动态进度条
            if st.button("🚀 确认并开始分析商家数据"):
                progress_bar = st.progress(0, text="正在初始化分析...")
                
                try:
                    lat = place["geometry"]["location"]["lat"]
                    lng = place["geometry"]["location"]["lng"]
                    
                                                            # 步骤 1: Yelp & Google 评论
                    
                                                            progress_bar.progress(25, text="正在获取 Yelp 与 Google 评论数据...")
                    
                                                            yelp_data = yelp_match(place["name"], lat, lng)
                    
                                                            
                    
                                                            # 1.1: 获取评论与情感分析
                    
                                                            yelp_reviews = []
                    
                                                            google_reviews = []
                    
                                                            
                    
                                                            # 获取 Google 评论
                    
                                                            try:
                    
                                                                google_reviews = get_google_reviews(place["place_id"])
                    
                                                            except Exception:
                    
                                                                pass
                    
                                        
                    
                                                            # 获取 Yelp 评论
                    
                                                            if yelp_data:
                    
                                                                try:
                    
                                                                    first_biz_id = yelp_data[0]['id']
                    
                                                                    yelp_reviews = get_yelp_reviews(first_biz_id)
                    
                                                                except Exception:
                    
                                                                    pass
                    
                                                            
                    
                                                            # 合并评论进行分析
                    
                                                            # 注意：Google 评论对象也有 'text' 字段，与 Yelp 结构兼容
                    
                                                            all_reviews = google_reviews + yelp_reviews
                    
                                                            sentiment_result = analyze_sentiment(all_reviews)
                    
                                                            
                    
                                                            # 步骤 2: 天气
                    progress_bar.progress(50, text="正在获取历史与预测天气数据...")
                    weather_hist = get_weather(lat, lng)
                    noaa = noaa_forecast(lat, lng)
                    
                    # 步骤 3: 人口普查
                    progress_bar.progress(75, text="正在查询商圈人口普查数据...")
                    census = census_data(lat, lng)
                    
                    # 完成
                    st.session_state.fetched_data = {
                        "place": place,
                        "yelp": yelp_data,
                        "yelp_reviews": yelp_reviews,
                        "google_reviews": google_reviews,
                        "sentiment": sentiment_result,
                        "weather_history": weather_hist.tail(10).to_dict(),
                        "noaa_forecast": noaa,
                        "census": census
                    }
                    st.session_state.current_place_id = place["place_id"]
                    
                    # 清除旧的深度报告
                    if "report_content" in st.session_state:
                        del st.session_state.report_content
                        
                    progress_bar.progress(100, text="数据拉取完成！")
                    
                except Exception as e:
                    st.error(f"数据拉取过程中发生错误: {str(e)}")
                    progress_bar.empty()

            # 3. 显示商家概要 (仅当数据已拉取时显示)
            if "fetched_data" in st.session_state and st.session_state.current_place_id == place["place_id"]:
                data = st.session_state.fetched_data
                
                st.divider()
                st.subheader("📊 商家数据概要")
                
                # Photos
                if "photos" in place:
                    p_cols = st.columns(min(3, len(place["photos"])))
                    for i, photo in enumerate(place["photos"][:3]):
                        url = get_google_photo_url(photo.get("photo_reference"))
                        if url:
                            with p_cols[i]:
                                st.image(url, use_column_width=True)
                                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Google 评分**: {place.get('rating', 'N/A')} ({place.get('user_ratings_total', 0)} 条)")
                with col2:
                    yelp_rating = data['yelp'][0]['rating'] if data['yelp'] else "N/A"
                    yelp_count = data['yelp'][0]['review_count'] if data['yelp'] else 0
                    st.error(f"**Yelp 评分**: {yelp_rating} ({yelp_count} 条)")
                with col3:
                    st.success(f"**商圈人口 (3英里)**: {data['census']['population_est']}")

                # Sentiment
                if data.get("sentiment"):
                    st.markdown("#### 💬 评论情感洞察")
                    sent = data["sentiment"]
                    s_col1, s_col2 = st.columns([1, 2])
                    with s_col1:
                        st.metric("情感倾向", sent.get("label", "N/A"), f"{sent.get('score', 0):.2f}")
                    with s_col2:
                        if sent.get("keywords"):
                            st.write("**热门关键词:**")
                            st.write(" ".join([f"`{k}`" for k in sent["keywords"]]))
                        else:
                            st.write("暂无足够评论提取关键词")

                with st.expander("查看详细原始数据"):
                    st.json(data)

                st.divider()

                # 4. 深度分析按钮
                col_btn, col_lang = st.columns([1, 1])
                with col_lang:
                    lang = st.selectbox("报告语言", ["zh", "en"], key="report_lang")
                
                with col_btn:
                    if st.button("🔍 生成深度AI策略报告", type="primary"):
                        # 初始化进度条
                        report_progress = st.progress(0, text="正在启动 AI 引擎...")
                        
                        try:
                            # 阶段 1: 准备上下文
                            report_progress.progress(20, text="正在整合商家数据与商圈信息...")
                            # 模拟一点处理时间，让用户看清提示
                            import time
                            time.sleep(0.5) 
                            
                            # 阶段 2: 构建 Prompt
                            report_progress.progress(40, text="正在构建高维分析模型...")
                            
                            # 阶段 3: 调用 API (这是最耗时的部分)
                            report_progress.progress(60, text="AI 顾问正在生成策略报告 (这可能需要 30-60 秒)...")
                            
                            # 这里的调用是同步的，会阻塞直到返回
                            report = generate_report(data, lang)
                            
                            # 阶段 4: 处理响应
                            report_progress.progress(90, text="正在格式化报告内容...")
                            st.session_state.report_content = report
                            
                            # 完成
                            report_progress.progress(100, text="报告生成完毕！")
                            
                        except Exception as e:
                            report_progress.empty()
                            st.error(f"报告生成失败: {str(e)}")
            
            # 5. 可编辑报告与导出
            if "report_content" in st.session_state and st.session_state.current_place_id == place["place_id"]:
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
