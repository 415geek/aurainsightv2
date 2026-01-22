import os
import json
import math
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import requests
import pandas as pd
import nltk
import time
import concurrent.futures
import io

# Ensure TextBlob corpora are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from meteostat import Point, Daily
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from openai import OpenAI
from textblob import TextBlob
import concurrent.futures
import re
import io

# ============================
# FEATURE FLAGS & CONFIG (v2.1)
# ============================
ENABLE_DATA_UPLOAD_PIPELINE = os.getenv("ENABLE_DATA_UPLOAD_PIPELINE", "true").lower() == "true"
# 模式：prompt_bound (使用 OpenAI 后台绑定的模型) | explicit_model (显式指定 ID)
AURAINSIGHT_MODEL_MODE = os.getenv("AURAINSIGHT_MODEL_MODE", "prompt_bound")
AURAINSIGHT_MODEL_ID = os.getenv("AURAINSIGHT_MODEL_ID", "gpt-4o")
AURAINSIGHT_PROMPT_ID = os.getenv("AURAINSIGHT_PROMPT_ID", "pmpt_6971b3bd094081959997af7730098d45020d02ec1efab62b")

# ============================
# DATA PIPELINE (CANONICAL SCHEMA v2.1)
# ============================
class DataPipeline:
    COLUMN_MAP = {
        '日期': 'date', 'date': 'date', '时间': 'date', 'day': 'date',
        '订单量': 'orders', '单量': 'orders', 'orders': 'orders', 'order_count': 'orders',
        '营收': 'revenue', '实收': 'revenue', 'revenue': 'revenue', 'sales': 'revenue', '金额': 'revenue',
        '渠道': 'channel', '来源': 'channel', 'channel': 'channel', 'platform': 'channel',
        '取消率': 'cancel_rate', '退单率': 'cancel_rate',
        '备餐时间': 'prep_time', '出餐时间': 'prep_time'
    }

    @classmethod
    def parse_file(cls, uploaded_file):
        fname = uploaded_file.name
        ext = fname.split('.')[-1].lower()
        
        try:
            if ext in ['csv']:
                df = pd.read_csv(uploaded_file)
            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif ext in ['pdf']:
                from PyPDF2 import PdfReader
                reader = PdfReader(uploaded_file)
                text = "".join([p.extract_text() for p in reader.pages])
                return {"type": "text", "content": text, "source": fname}
            elif ext in ['docx']:
                import docx
                doc = docx.Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
                return {"type": "text", "content": text, "source": fname}
            elif ext in ['png', 'jpg', 'jpeg', 'webp']:
                return {"type": "image_placeholder", "content": "OCR Pending v2.2", "source": fname}
            else:
                return {"error": f"不支持的格式: {ext}"}
            
            # 基础列名映射
            df = df.rename(columns=lambda x: cls.COLUMN_MAP.get(str(x).lower().strip(), x))
            return {"type": "table", "data": df, "source": fname}
        except Exception as e:
            return {"error": f"解析失败 ({fname}): {str(e)}"}

    @classmethod
    def build_canonical_schema(cls, files):
        operational_data = {
            "time_series": [],
            "channels_summary": [],
            "data_quality_report": {"missing_fields": [], "warnings": []},
            "traceability": []
        }
        
        all_dfs = []
        for f in files:
            res = cls.parse_file(f)
            if "error" in res:
                operational_data["data_quality_report"]["warnings"].append(res["error"])
                continue
            if res["type"] == "table":
                df = res["data"]
                # 尝试日期标准化
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    except: pass
                
                # 清洗数值
                for col in ['orders', 'revenue']:
                    if col in df.columns:
                        df[col] = df[col].apply(cls.clean_numeric)
                
                all_dfs.append({"df": df, "source": res["source"]})

        if not all_dfs:
            return operational_data

        # 合并数据
        merged = pd.concat([d["df"] for d in all_dfs], ignore_index=True)
        
        # 计算 AOV (DERIVED)
        if 'revenue' in merged.columns and 'orders' in merged.columns:
            merged['aov'] = (merged['revenue'] / merged['orders']).replace([float('inf'), -float('inf')], 0).fillna(0)
            operational_data["traceability"].append({"field": "aov", "source": "derived: revenue/orders", "tag": "DERIVED_DATA"})

        # 生成 time_series
        if 'date' in merged.columns:
            ts_cols = [c for c in ['date', 'channel', 'orders', 'revenue', 'aov', 'cancel_rate', 'prep_time'] if c in merged.columns]
            operational_data["time_series"] = merged[ts_cols].to_dict('records')
            
        # 质量报告
        needed = ['date', 'orders', 'revenue']
        operational_data["data_quality_report"]["missing_fields"] = [f for f in needed if f not in merged.columns]
        
        return operational_data

    @staticmethod
    def clean_numeric(val):
        if pd.isna(val): return 0
        if isinstance(val, (int, float)): return val
        clean_val = re.sub(r'[^\d\.]', '', str(val))
        try: return float(clean_val)
        except: return 0

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

def get_yelp_details(business_id):
    if not YELP_API_KEY:
        return {}
    
    url = f"https://api.yelp.com/v3/businesses/{business_id}"
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}

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

def generate_report(data, lang="zh", operational_data=None):
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 准备变量
    restaurant_name = data.get("place", {}).get("name", "Unknown Restaurant")
    restaurant_address = data.get("place", {}).get("formatted_address", "Unknown Address")
    
    # 构建 Payload (Canonical Schema v2.1)
    payload = {
        "restaurant_profile": data.get("place"),
        "weather": {"history": data.get("weather_history"), "forecast": data.get("noaa_forecast")},
        "operational_data": operational_data if operational_data else "MISSING - USE INDUSTRY PRIORS",
        "timestamp": datetime.now().isoformat()
    }
    
    input_data_str = json.dumps(payload, ensure_ascii=False, indent=2, default=json_serial)
    
    # 强制量化指令 (v2.1)
    system_instruction = f"""
    You are AuraInsight v2.1 (Quantitative Mode). 
    
    MANDATORY STANDARDS:
    1. If 'operational_data' is present, you MUST use [VERIFIED] data for all core KPIs.
    2. Every forecast (Orders/Revenue) MUST include a P10, P50, and P90 confidence interval.
    3. You MUST provide the specific formula used for your prediction (e.g., Bayesian Linear Regression with Weather Coefficients).
    4. If weather history is present, calculate the correlation coefficient between rain/temp and order volume.
    5. Output Language: {"Chinese" if lang == "zh" else "English"}.
    """
    
    input_payload = input_data_str + f"\n\n[SYSTEM_DIRECTIVE]: {system_instruction}"

    try:
        # 模型路由逻辑
        if AURAINSIGHT_MODEL_MODE == "explicit_model":
            # 模式 B：显式强控
            response = client.responses.create(
                model=AURAINSIGHT_MODEL_ID,
                prompt={
                    "variables": {
                        "restaurant_name": restaurant_name,
                        "restaurant_address": restaurant_address,
                        "input_data": input_payload
                    }
                }
            )
            # 记录模型日志
            st.info(f"Using Explicit Model: {AURAINSIGHT_MODEL_ID}")
        else:
            # 模式 A：Prompt 绑定 (默认)
            response = client.responses.create(
                prompt={
                    "id": AURAINSIGHT_PROMPT_ID,
                    "version": "2",
                    "variables": {
                        "restaurant_name": restaurant_name,
                        "restaurant_address": restaurant_address,
                        "input_data": input_payload
                    }
                }
            )
            st.info("Using Prompt-Bound Model Mapping")

        
        # 2. 轮询状态，直到完成 (OpenAI Responses API 是异步的)
        import time
        max_retries = 30 # 最多等待 60 秒 (2s * 30)
        retries = 0
        final_response = response
        
        while retries < max_retries:
            # 检查当前状态
            # 如果状态已经是 completed 或 failed，退出循环
            if hasattr(final_response, 'status'):
                if final_response.status == 'completed':
                    break
                if final_response.status in ['failed', 'incomplete', 'cancelled']:
                    return f"❌ AI 响应失败 (状态: {final_response.status})"
            
            # 等待并重新获取状态
            time.sleep(2)
            final_response = client.responses.retrieve(final_response.id)
            retries += 1
            
        # 3. 解析最终生成的文本内容
        text_content = ""
        if hasattr(final_response, 'output') and isinstance(final_response.output, list):
            for item in final_response.output:
                # 寻找类型为 message 的输出项
                if hasattr(item, 'content') and isinstance(item.content, list):
                    for part in item.content:
                        # 处理文本块
                        if hasattr(part, 'text'):
                            # 有些版本是 part.text.value，有些是 part.text
                            if hasattr(part.text, 'value'):
                                text_content += part.text.value
                            elif isinstance(part.text, str):
                                text_content += part.text
                        elif isinstance(part, dict) and 'text' in part:
                            t = part['text']
                            text_content += t.get('value', t) if isinstance(t, dict) else str(t)
        
        if text_content:
            return text_content
            
        # 兜底显示：如果轮询超时或解析失败
        if hasattr(final_response, 'model_dump_json'):
            return f"⚠️ 报告生成超时或解析失败。原始 JSON：\n\n{final_response.model_dump_json(indent=2, ensure_ascii=False)}"
        
        return f"⚠️ 无法获取报告内容。状态: {getattr(final_response, 'status', 'unknown')}"

    except AttributeError as ae:
        return f"❌ OpenAI 库版本不支持此操作或 API 结构已变更: {str(ae)}"
    except Exception as e:
        return f"❌ 生成报告时发生错误: {str(e)}"

# ... (中间代码保持不变) ...

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="AuraInsight v2.1", layout="wide")

# Sidebar Monitoring
with st.sidebar:
    st.image("https://maps.gstatic.com/mapfiles/place_api/icons/v1/png_71/restaurant-71.png", width=50)
    st.title("System Status")
    st.caption(f"Model Mode: {AURAINSIGHT_MODEL_MODE}")
    st.caption(f"Pipeline: {'Enabled' if ENABLE_DATA_UPLOAD_PIPELINE else 'Disabled'}")
    if st.toggle("Debug Info"):
        st.json({"model_id": AURAINSIGHT_MODEL_ID, "prompt_id": AURAINSIGHT_PROMPT_ID})

st.title("AuraInsight · 量化增长分析系统 v2.1")

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

                    # 获取 Yelp 评论与详情
                    yelp_details = {}
                    if yelp_data:
                        try:
                            first_biz_id = yelp_data[0]['id']
                            yelp_reviews = get_yelp_reviews(first_biz_id)
                            # 获取商家详情以拉取更多图片
                            yelp_details = get_yelp_details(first_biz_id)
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
                        "yelp_details": yelp_details,
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
                
                # Photos Gallery
                all_photo_urls = []
                
                # Google Photos (Up to 6)
                if "photos" in place:
                    for photo in place["photos"][:6]:
                        url = get_google_photo_url(photo.get("photo_reference"), max_width=800)
                        if url:
                            all_photo_urls.append(("Google", url))
                
                # Yelp Photos
                if data.get("yelp_details") and "photos" in data["yelp_details"]:
                    for url in data["yelp_details"]["photos"]:
                        all_photo_urls.append(("Yelp", url))
                
                if all_photo_urls:
                    st.markdown("#### 📸 门店实景与菜品预览")
                    # 使用 3 列网格展示图片
                    cols = st.columns(3)
                    for i, (source, url) in enumerate(all_photo_urls):
                        with cols[i % 3]:
                            st.image(url, caption=f"来源: {source}", use_column_width=True)
                                
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
                            report_progress.progress(10, text="正在整合商家数据与商圈信息...")
                            time.sleep(0.5)
                            
                            # 阶段 2: 构建 Prompt
                            report_progress.progress(30, text="正在构建高维分析模型...")
                            
                            # 阶段 3: 异步调用 API 并动态更新文字
                            loading_texts = [
                                "正在通过 GPT-4o 进行深度推理...",
                                "正在分析 Yelp 与 Google 评论情感趋势...",
                                "正在结合 Meteostat 历史天气数据进行回归分析...",
                                "正在交叉比对 Census 商圈人口统计数据...",
                                "正在生成麦肯锡风格的战略建议...",
                                "正在优化报告格式与排版..."
                            ]
                            
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(generate_report, data, lang)
                                
                                # 循环更新进度条文字，直到任务完成
                                idx = 0
                                progress_val = 30
                                while not future.done():
                                    if idx < len(loading_texts):
                                        current_text = loading_texts[idx]
                                    else:
                                        # 如果所有预设文案都显示完了，不再循环，而是显示通用等待提示
                                        current_text = "正在进行最终的深度逻辑整合，请耐心等待..."
                                    
                                    # 让进度条缓慢增加，但不到 100%
                                    if progress_val < 90:
                                        progress_val += 1
                                    
                                    report_progress.progress(progress_val, text=f"AI 顾问工作流: {current_text}")
                                    time.sleep(1.5) # 每 1.5 秒切换一次文字
                                    idx += 1
                                
                                # 获取结果
                                report = future.result()
                            
                            # 阶段 4: 处理响应
                            report_progress.progress(95, text="正在最终格式化报告内容...")
                            st.session_state.report_content = report
                            
                            # 完成
                            report_progress.progress(100, text="报告生成完毕！")
                            
                        except Exception as e:
                            report_progress.empty()
                            st.error(f"报告生成失败: {str(e)}")
            
            # 5. 可编辑报告与导出
            if "report_content" in st.session_state and st.session_state.current_place_id == place["place_id"]:
                st.divider()
                st.subheader("📝 深度分析报告 (可编辑)")
                
                # 用户可以在这里修改报告
                user_edited_report = st.text_area(
                    "您可以直接修改下方的报告内容，修改后点击下载即可。",
                    value=st.session_state.report_content,
                    height=500,
                    key="report_area"
                )
                
                # 导出按钮
                col_exp1, col_exp2 = st.columns([1, 1])
                with col_exp1:
                    if st.button("📥 导出 PDF 分析报告"):
                        export_pdf(user_edited_report, "analysis_report.pdf")
                        with open("analysis_report.pdf", "rb") as pdf_file:
                            st.download_button(
                                label="点击下载 PDF",
                                data=pdf_file,
                                file_name="AuraInsight_Report.pdf",
                                mime="application/pdf"
                            )
                        st.success("PDF 已生成！")

                # ============================
                # 阶段 1.2: 补充数据上传区域 (闭环核心)
                # ============================
                if ENABLE_DATA_UPLOAD_PIPELINE:
                    st.divider()
                    st.markdown("### 📊 补充运营数据（数据闭环）")
                    st.info("💡 上传真实数据（POS/外卖平台导出）后，AI 将重新清洗并校准模型结论，提供更高精度的报告。")
                    
                    uploaded_files = st.file_uploader(
                        "支持 CSV, XLSX, TXT (支持多文件同时上传)", 
                        accept_multiple_files=True,
                        type=['csv', 'xlsx', 'xls', 'txt']
                    )
                    
                    if uploaded_files:
                        op_data = DataPipeline.build_canonical_schema(uploaded_files)
                        
                        # 可视化反馈
                        v_col1, v_col2 = st.columns([2, 1])
                        with v_col1:
                            st.write("**数据质量审计**")
                            if op_data["data_quality_report"]["missing_fields"]:
                                st.warning(f"缺失关键字段: {', '.join(op_data['data_quality_report']['missing_fields'])}")
                            if op_data["time_series"]:
                                st.success(f"解析成功: 获取到 {len(op_data['time_series'])} 条记录")
                                st.dataframe(pd.DataFrame(op_data["time_series"]).head(10))
                        
                        with v_col2:
                            st.write("**字段来源追溯**")
                            for item in op_data["traceability"]:
                                st.caption(f"{item['field']}: {item['tag']} ({item['source']})")
                        
                        # 重新生成按钮
                        if st.button("🔄 注入真实数据并重新建模生成报告", type="primary"):
                            with st.progress(0, text="正在同步量化模型..."):
                                new_report = generate_report(data, lang, operational_data=op_data)
                                st.session_state.report_content = new_report
                                st.rerun()


                    # Admin 回滚开关 (隐藏)
                    if st.toggle("Admin: 使用旧模型版本 (Rollback Mode)", value=False):
                        st.session_state.use_legacy_model = True
                    else:
                        st.session_state.use_legacy_model = False

