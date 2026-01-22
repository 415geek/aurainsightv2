import streamlit as st
import os
import requests
import openai
from datetime import datetime, timedelta
from meteostat import Point, Daily

# 页面配置
st.set_page_config(page_title="API Health Check", page_icon="🔧", layout="wide")

st.title("🔧 系统 API 健康检查诊断")
st.markdown("此页面用于验证所有外部服务的连通性和 API Key 配置情况。")

# 获取 Keys
def get_secret(key):
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.getenv(key)

GOOGLE_API_KEY = get_secret("GOOGLE_MAPS_API_KEY")
YELP_API_KEY = get_secret("YELP_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def check_google():
    st.subheader("1. Google Maps API (Places)")
    if not GOOGLE_API_KEY:
        st.error("❌ 环境变量 `GOOGLE_MAPS_API_KEY` 未设置")
        return False
    
    try:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        # 测试查询：旧金山市政厅
        params = {"query": "San Francisco City Hall", "key": GOOGLE_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if data.get("status") == "OK":
            st.success(f"✅ 连接成功! 找到 {len(data.get('results', []))} 个结果。")
            with st.expander("查看原始响应"):
                st.json(data["results"][0] if data["results"] else {})
            return True
        else:
            st.error(f"❌ API 响应错误: {data.get('status')}")
            st.error(f"错误信息: {data.get('error_message')}")
            return False
    except Exception as e:
        st.error(f"❌ 请求异常: {e}")
        return False

def check_yelp():
    st.subheader("2. Yelp Fusion API")
    if not YELP_API_KEY:
        st.error("❌ 环境变量 `YELP_API_KEY` 未设置")
        return False
        
    try:
        url = "https://api.yelp.com/v3/businesses/search"
        headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
        # 测试查询：旧金山的咖啡厅
        params = {"term": "coffee", "location": "San Francisco", "limit": 1}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            st.success(f"✅ 连接成功! 找到商家: {data['businesses'][0]['name']}")
            return True
        else:
            st.error(f"❌ API 错误 (Status {r.status_code}): {r.text}")
            return False
    except Exception as e:
        st.error(f"❌ 请求异常: {e}")
        return False

def check_openai():
    st.subheader("3. OpenAI API (GPT-4o)")
    if not OPENAI_API_KEY:
        st.error("❌ 环境变量 `OPENAI_API_KEY` 未设置")
        return False
        
    try:
        # 简单测试请求
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
            max_tokens=10,
            temperature=0
        )
        content = resp.choices[0].message.content
        st.success(f"✅ 连接成功! 模型回复: {content}")
        return True
    except Exception as e:
        st.error(f"❌ OpenAI API 调用失败: {e}")
        return False

def check_weather():
    st.subheader("4. Meteostat & NOAA (Weather)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Meteostat (Python Lib)**")
        try:
            # 测试旧金山坐标
            start = datetime.now() - timedelta(days=7)
            end = datetime.now()
            sf_point = Point(37.7749, -122.4194)
            data = Daily(sf_point, start, end)
            data = data.fetch()
            
            if not data.empty:
                st.success(f"✅ 获取到 {len(data)} 条历史天气记录")
            else:
                st.warning("⚠️ 库调用成功但未返回数据 (可能是地点/时间问题)")
        except Exception as e:
            st.error(f"❌ Meteostat 错误: {e}")

    with col2:
        st.markdown("**NOAA API (Public)**")
        try:
            # 测试旧金山 Grid
            url = "https://api.weather.gov/points/37.7749,-122.4194"
            r = requests.get(url, headers={"User-Agent": "AuraInsight-Test"}, timeout=10)
            if r.status_code == 200:
                st.success("✅ NOAA Metadata 获取成功")
            else:
                st.error(f"❌ NOAA 错误: {r.status_code}")
        except Exception as e:
            st.error(f"❌ 请求异常: {e}")

if st.button("🚀 开始全面诊断", type="primary"):
    with st.spinner("正在逐个测试接口连接..."):
        g_ok = check_google()
        st.divider()
        y_ok = check_yelp()
        st.divider()
        o_ok = check_openai()
        st.divider()
        check_weather()
        
    if g_ok and y_ok and o_ok:
        st.balloons()
        st.success("🎉 恭喜！核心 API 均配置正确且工作正常。")
    else:
        st.warning("⚠️ 部分 API 存在问题，请检查上方的错误提示。")
