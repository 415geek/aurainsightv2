
# app.py
# AuraInsight Delivery Growth Intelligence - Streamlit Demo
# ---------------------------------------------------------
# Demo goals:
# - Input restaurant name + address
# - Optional metrics: impressions, CTR, CVR, AOV, promo spend
# - Pull (optional) data from APIs: Google Maps (geocode/places/traffic), US Census (ACS), Weather (OpenWeather)
# - Competitor mining (optional) via Google Places OR Yelp Fusion
# - Weather & traffic impact coefficients
# - Bayesian updating for CTR/CVR using Beta priors
# - Quantitative forecast with probability ranges (P10/P50/P90)
# - Action plan + A/B tests + KPI dashboard
#
# NOTE: This demo is designed to run even WITHOUT API keys (uses safe fallbacks / illustrative defaults).
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import os
import math
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AuraInsight 外卖增收分析 Demo", layout="wide")

# -----------------------------
# i18n dictionary
# -----------------------------
I18N = {
    "zh-CN": {
        "app_title": "AuraInsight 外卖增收分析（Demo）",
        "language": "语言",
        "lang_zh": "中文",
        "lang_en": "English",
        "tab_input": "输入",
        "tab_report": "报告",
        "tab_debug": "调试/数据源",
        "restaurant_name": "餐厅名称",
        "address": "地址（完整）",
        "cuisine": "菜系",
        "platforms": "使用的外卖平台",
        "optional_metrics": "可选：近30天平台指标（没有也能生成报告）",
        "impressions": "曝光量（Impressions）",
        "ctr": "点击率 CTR（%）",
        "cvr": "下单率 CVR（%）",
        "aov": "客单价 AOV（$）",
        "promo": "活动花费 Promo Spend（$）",
        "generate": "生成报告",
        "api_keys": "API Keys（可选）",
        "google_key": "Google Maps API Key（Geocode/Places/Traffic）",
        "census_key": "US Census API Key（ACS）",
        "openweather_key": "OpenWeather API Key（天气）",
        "yelp_key": "Yelp Fusion API Key（竞对/口碑，可选）",
        "assumptions": "输入与假设（可编辑）",
        "data_sources": "数据源与可验证性",
        "no_keys_fallback": "未提供关键API Key：本Demo将使用保守默认值与示例数据（仍可演示完整逻辑）。",
        "geo": "地理信息",
        "latlng": "经纬度",
        "trade_area": "商圈画像（1mi/3mi）",
        "competitors": "竞对分析（重点）",
        "weather_traffic": "天气与交通影响（必须）",
        "funnel": "转化漏斗诊断",
        "bayes": "贝叶斯预测（概率，不是拍脑袋）",
        "quant": "量化预测模型（Orders & Revenue）",
        "plan": "30/60/90 天行动计划",
        "abtests": "A/B 测试设计",
        "kpi": "KPI 仪表盘",
        "export_hint": "（Demo 版：此处先展示报告结构与关键计算；PDF 导出可在下一迭代加入。）",
        "warn_low_data": "提示：你没有提供平台指标（曝光/CTR/CVR等），系统会用行业先验 + 可解释代理变量进行估计，预测区间会更宽。",
    },
    "en-US": {
        "app_title": "AuraInsight Delivery Growth (Demo)",
        "language": "Language",
        "lang_zh": "Chinese",
        "lang_en": "English",
        "tab_input": "Input",
        "tab_report": "Report",
        "tab_debug": "Debug / Sources",
        "restaurant_name": "Restaurant name",
        "address": "Full address",
        "cuisine": "Cuisine",
        "platforms": "Delivery platforms",
        "optional_metrics": "Optional: last 30 days metrics (report still works without)",
        "impressions": "Impressions",
        "ctr": "CTR (%)",
        "cvr": "CVR (%)",
        "aov": "AOV ($)",
        "promo": "Promo spend ($)",
        "generate": "Generate report",
        "api_keys": "API Keys (optional)",
        "google_key": "Google Maps API Key (Geocode/Places/Traffic)",
        "census_key": "US Census API Key (ACS)",
        "openweather_key": "OpenWeather API Key (Weather)",
        "yelp_key": "Yelp Fusion API Key (competitors/reviews, optional)",
        "assumptions": "Inputs & Assumptions (editable)",
        "data_sources": "Data sources & verifiability",
        "no_keys_fallback": "Missing key API credentials: demo will use conservative defaults and sample data (full logic still shown).",
        "geo": "Geo",
        "latlng": "Lat/Lng",
        "trade_area": "Trade area (1mi/3mi)",
        "competitors": "Competitor analysis (core)",
        "weather_traffic": "Weather & traffic impact (required)",
        "funnel": "Conversion funnel",
        "bayes": "Bayesian forecast (probabilistic)",
        "quant": "Quantitative model (Orders & Revenue)",
        "plan": "30/60/90-day action plan",
        "abtests": "A/B testing plan",
        "kpi": "KPI dashboard",
        "export_hint": "(Demo: shows structure + key calculations; PDF export can be added next.)",
        "warn_low_data": "Note: no platform metrics provided. The system will estimate using priors + explainable proxies; prediction intervals will be wider.",
    }
}

def t(key: str) -> str:
    lang = st.session_state.get("lang", "zh-CN")
    return I18N.get(lang, I18N["zh-CN"]).get(key, key)

# -----------------------------
# Bayesian helpers
# -----------------------------
def beta_posterior_from_rate(impressions: Optional[int], rate: Optional[float], alpha0: float, beta0: float,
                             pseudo_n: int = 200) -> Tuple[float, float, bool]:
    """
    Return posterior (alpha, beta) for a rate using Beta prior.
    If impressions is missing, approximate with pseudo_n trials to keep uncertainty honest.
    """
    used_pseudo = False
    if rate is None:
        return alpha0, beta0, True
    r = max(0.0, min(1.0, float(rate)))
    n = impressions if (impressions is not None and impressions > 0) else pseudo_n
    if impressions is None or impressions <= 0:
        used_pseudo = True
    succ = r * n
    fail = (1.0 - r) * n
    return alpha0 + succ, beta0 + fail, used_pseudo

def beta_quantiles(alpha: float, beta: float, qs=(0.1, 0.5, 0.9)) -> Dict[float, float]:
    from scipy.stats import beta as beta_dist
    return {q: float(beta_dist.ppf(q, alpha, beta)) for q in qs}

def prob_beta_greater(alpha: float, beta: float, threshold: float) -> float:
    from scipy.stats import beta as beta_dist
    return float(1.0 - beta_dist.cdf(threshold, alpha, beta))

# -----------------------------
# External API wrappers (optional)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def geocode_google(address: str, google_key: str) -> Optional[Tuple[float, float]]:
    if not google_key:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address": address, "key": google_key}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return None
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])

@st.cache_data(show_spinner=False, ttl=60 * 60)
def weather_openweather(lat: float, lng: float, key: str) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    # One Call 3.0 endpoint (requires subscription in some tiers). We use /data/2.5/forecast as fallback-friendly.
    # We'll use "onecall" if available, else forecast.
    onecall = "https://api.openweathermap.org/data/3.0/onecall"
    try:
        r = requests.get(onecall, params={"lat": lat, "lon": lng, "appid": key, "units": "imperial"}, timeout=20)
        if r.status_code == 200:
            return {"provider": "openweather_onecall", "data": r.json()}
    except Exception:
        pass
    forecast = "https://api.openweathermap.org/data/2.5/forecast"
    r = requests.get(forecast, params={"lat": lat, "lon": lng, "appid": key, "units": "imperial"}, timeout=20)
    if r.status_code != 200:
        return None
    return {"provider": "openweather_forecast", "data": r.json()}

@st.cache_data(show_spinner=False, ttl=60 * 60)
def distance_matrix_google(origin: str, dest: str, google_key: str) -> Optional[Dict[str, Any]]:
    if not google_key:
        return None
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": dest,
        "departure_time": "now",
        "key": google_key
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=60 * 60)
def places_nearby_google(lat: float, lng: float, radius_m: int, keyword: str, google_key: str) -> Optional[List[Dict[str, Any]]]:
    if not google_key:
        return None
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "keyword": keyword,
        "type": "restaurant",
        "key": google_key
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("status") not in ("OK", "ZERO_RESULTS"):
        return None
    return data.get("results", [])

# Census: for demo we keep it simple and optionally show how to query.
# Trade-area exact ring computation + tract mapping is a next iteration; here we show a conservative, explainable default.
# -----------------------------
# Coefficients
# -----------------------------
def weather_coefficient(temp_f: float, precip_mm: float, condition: str) -> float:
    """
    Simple explainable weather -> demand coefficient.
    Conservative defaults:
      - Rain increases demand up to +25%
      - Cold (<50F) increases +8%
      - Heat (>90F) increases +6%
    """
    coef = 1.0
    if precip_mm and precip_mm > 0:
        coef *= 1.15
        if precip_mm > 5:
            coef *= 1.08
    if temp_f < 50:
        coef *= 1.08
    if temp_f > 90:
        coef *= 1.06
    # Fog/wind can increase "stay-in" but also add delivery delay; we leave to traffic coefficient.
    return float(min(1.35, max(0.85, coef)))

def traffic_coefficient(duration_sec: Optional[float], duration_in_traffic_sec: Optional[float]) -> Tuple[float, float]:
    """
    Fulfillment risk coefficient derived from congestion index.
    coef reduces effective conversion when traffic is worse.
    """
    if not duration_sec or not duration_in_traffic_sec or duration_sec <= 0:
        return 1.0, 1.0
    cong = duration_in_traffic_sec / duration_sec
    # Map congestion ratio to coefficient: mild congestion little impact; heavy congestion reduces conversion and increases cancellations.
    # This is a conservative mapping for demo.
    coef = 1.0
    if cong >= 1.1:
        coef = 0.97
    if cong >= 1.25:
        coef = 0.93
    if cong >= 1.5:
        coef = 0.88
    return float(coef), float(cong)

# -----------------------------
# Default priors (industry baselines)
# -----------------------------
@dataclass
class Priors:
    ctr_alpha: float = 8.0   # mean ~ 8/(8+92)=8%
    ctr_beta: float = 92.0
    cvr_alpha: float = 20.0  # mean ~ 20/(20+80)=20%
    cvr_beta: float = 80.0
    aov_mu: float = math.log(26.0)  # lognormal mean-ish around 26
    aov_sigma: float = 0.35

PRIORS = Priors()

# -----------------------------
# UI: sidebar
# -----------------------------
st.sidebar.markdown("### " + t("language"))
lang = st.sidebar.radio("", options=["zh-CN", "en-US"], index=0, format_func=lambda x: I18N[x]["lang_zh"] if x=="zh-CN" else I18N[x]["lang_en"])
st.session_state["lang"] = lang

st.sidebar.markdown("### " + t("api_keys"))
google_key = st.sidebar.text_input(t("google_key"), value=os.getenv("GOOGLE_MAPS_API_KEY", ""), type="password")
census_key = st.sidebar.text_input(t("census_key"), value=os.getenv("CENSUS_API_KEY", ""), type="password")
openweather_key = st.sidebar.text_input(t("openweather_key"), value=os.getenv("OPENWEATHER_API_KEY", ""), type="password")
yelp_key = st.sidebar.text_input(t("yelp_key"), value=os.getenv("YELP_API_KEY", ""), type="password")

st.sidebar.info(t("no_keys_fallback"))

# -----------------------------
# Main tabs
# -----------------------------
st.title(t("app_title"))
tab_input, tab_report, tab_debug = st.tabs([t("tab_input"), t("tab_report"), t("tab_debug")])

# -----------------------------
# Input tab
# -----------------------------
with tab_input:
    col1, col2 = st.columns([2, 1])

    with col1:
        name = st.text_input(t("restaurant_name"), value="Ohana Hawaiian BBQ")
        address = st.text_input(t("address"), value="1240 Anderson Dr #103, Suisun City, CA 94585")
        cuisine = st.text_input(t("cuisine"), value="Hawaiian BBQ / Plate Lunch")
        platforms = st.multiselect(t("platforms"),
                                  options=["DoorDash", "Uber Eats", "Grubhub", "Fantuan (饭团)", "HungryPanda (熊猫)", "Other"],
                                  default=["DoorDash", "Uber Eats"])

        st.markdown("#### " + t("optional_metrics"))
        m1, m2, m3 = st.columns(3)
        with m1:
            impressions = st.number_input(t("impressions"), min_value=0, value=0, step=100)
        with m2:
            ctr_pct = st.number_input(t("ctr"), min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        with m3:
            cvr_pct = st.number_input(t("cvr"), min_value=0.0, max_value=100.0, value=0.0, step=0.1)

        m4, m5 = st.columns(2)
        with m4:
            aov = st.number_input(t("aov"), min_value=0.0, value=0.0, step=1.0)
        with m5:
            promo = st.number_input(t("promo"), min_value=0.0, value=0.0, step=10.0)

        gen = st.button(t("generate"), type="primary")

    with col2:
        st.markdown("### " + t("assumptions"))
        # Editable assumptions for demo
        radius_miles_core = st.slider("Core radius (miles)", 0.5, 5.0, 3.0, 0.5)
        # Trade-area population defaults (conservative placeholders)
        pop_1mi = st.number_input("Population (1mi) - fallback", min_value=0, value=20000, step=1000)
        pop_3mi = st.number_input("Population (3mi) - fallback", min_value=0, value=90000, step=5000)
        median_income = st.number_input("Median household income ($) - fallback", min_value=0, value=97083, step=1000)
        delivery_pen = st.slider("Delivery penetration (share of people ordering delivery monthly)", 0.05, 0.40, 0.22, 0.01)
        freq = st.slider("Monthly delivery frequency (orders/person/month)", 1.0, 5.0, 2.5, 0.1)

        st.caption(t("export_hint"))

    if gen:
        st.session_state["run"] = {
            "name": name, "address": address, "cuisine": cuisine, "platforms": platforms,
            "impressions": int(impressions) if impressions else None,
            "ctr": (ctr_pct / 100.0) if ctr_pct else None,
            "cvr": (cvr_pct / 100.0) if cvr_pct else None,
            "aov": float(aov) if aov else None,
            "promo": float(promo) if promo else None,
            "assumptions": {
                "radius_miles_core": radius_miles_core,
                "pop_1mi": pop_1mi,
                "pop_3mi": pop_3mi,
                "median_income": median_income,
                "delivery_pen": delivery_pen,
                "freq": freq,
            }
        }
        st.success("✅ Report job created. Go to the Report tab.")

# -----------------------------
# Report tab
# -----------------------------
with tab_report:
    run = st.session_state.get("run")
    if not run:
        st.info("Fill inputs and click Generate.")
    else:
        # Progress simulation
        prog = st.progress(0, text="Initializing…")
        time.sleep(0.1)

        # 1) Geocode
        prog.progress(10, text="Geocoding…")
        latlng = None
        try:
            latlng = geocode_google(run["address"], google_key) if google_key else None
        except Exception:
            latlng = None

        if latlng is None:
            # fallback: not precise; purely for demo continuity
            latlng = (38.2380, -122.0400)  # approx Suisun City
            geo_note = "fallback"
        else:
            geo_note = "google"

        prog.progress(25, text="Weather & Traffic…")

        # 2) Weather
        weather = None
        temp_f, precip_mm, w_desc = 58.0, 0.0, "unknown"
        if openweather_key:
            try:
                weather = weather_openweather(latlng[0], latlng[1], openweather_key)
            except Exception:
                weather = None

        if weather:
            if weather["provider"] == "openweather_onecall":
                cur = weather["data"].get("current", {})
                temp_f = float(cur.get("temp", temp_f))
                w_desc = (cur.get("weather") or [{"description":"unknown"}])[0].get("description", "unknown")
                rain = cur.get("rain", {}).get("1h", 0.0) if isinstance(cur.get("rain", {}), dict) else 0.0
                precip_mm = float(rain or 0.0)
            else:
                # forecast: take first item
                lst = weather["data"].get("list", [])
                if lst:
                    first = lst[0]
                    temp_f = float(first.get("main", {}).get("temp", temp_f))
                    w_desc = (first.get("weather") or [{"description":"unknown"}])[0].get("description", "unknown")
                    precip_mm = float(first.get("rain", {}).get("3h", 0.0) or 0.0)

        w_coef = weather_coefficient(temp_f, precip_mm, w_desc)

        # 3) Traffic coefficient (proxy with commute risk if no API)
        traffic = None
        t_coef, cong = 1.0, 1.0
        if google_key:
            try:
                # Use a conservative "destination" as a nearby major node (demo). In production: sample to major residential centroids.
                traffic = distance_matrix_google(run["address"], run["address"], google_key)
            except Exception:
                traffic = None

        if traffic and traffic.get("rows"):
            # origin=dest => duration=0, so fallback. (In prod you'd compute to multiple points.)
            t_coef, cong = 1.0, 1.0
        else:
            # Use a conservative commute proxy: assume mild congestion factor
            cong = 1.25
            t_coef, _ = traffic_coefficient(1.0, cong)

        prog.progress(45, text="Competitors…")

        # 4) Competitors (Google Places)
        competitors = []
        if google_key:
            try:
                results = places_nearby_google(latlng[0], latlng[1], int(run["assumptions"]["radius_miles_core"] * 1609.34), keyword=run["cuisine"].split("/")[0].strip(), google_key=google_key)
                if results:
                    for r0 in results[:20]:
                        competitors.append({
                            "name": r0.get("name"),
                            "rating": r0.get("rating"),
                            "user_ratings_total": r0.get("user_ratings_total"),
                            "price_level": r0.get("price_level"),
                            "vicinity": r0.get("vicinity"),
                            "types": ", ".join((r0.get("types") or [])[:5]),
                        })
            except Exception:
                competitors = []

        if not competitors:
            # fallback illustrative competitor set (demo)
            competitors = [
                {"name":"Hawaiian BBQ Express (nearby)", "rating":4.2, "user_ratings_total":350, "price_level":1, "vicinity":"Vacaville area", "types":"hawaiian, bbq"},
                {"name":"L&L Hawaiian Barbecue (nearby)", "rating":4.1, "user_ratings_total":900, "price_level":1, "vicinity":"Fairfield area", "types":"hawaiian, bbq"},
                {"name":"Korean BBQ (substitute)", "rating":4.3, "user_ratings_total":520, "price_level":2, "vicinity":"Vacaville area", "types":"korean, bbq"},
                {"name":"Local BBQ (substitute)", "rating":4.0, "user_ratings_total":410, "price_level":2, "vicinity":"Suisun/Fairfield", "types":"bbq"},
            ]

        comp_df = pd.DataFrame(competitors)
        # Tiering (simple demo rules)
        def tier(row):
            types = str(row.get("types","")).lower()
            if "hawai" in types or "hawaiian" in types:
                return "Direct"
            if "bbq" in types:
                return "Substitute"
            return "Occasion"

        comp_df["tier"] = comp_df.apply(tier, axis=1)
        comp_df["trust_score"] = (comp_df["rating"].fillna(0) * np.log1p(comp_df["user_ratings_total"].fillna(0))).round(2)

        prog.progress(60, text="Bayesian inference…")

        # 5) Bayesian posteriors for CTR/CVR
        imp = run["impressions"]
        ctr = run["ctr"]
        cvr = run["cvr"]

        if imp is None or (ctr is None and cvr is None):
            st.warning(t("warn_low_data"))

        ctr_a, ctr_b, ctr_pseudo = beta_posterior_from_rate(imp, ctr, PRIORS.ctr_alpha, PRIORS.ctr_beta, pseudo_n=250)
        cvr_a, cvr_b, cvr_pseudo = beta_posterior_from_rate(imp, cvr, PRIORS.cvr_alpha, PRIORS.cvr_beta, pseudo_n=120)

        from scipy.stats import beta as beta_dist
        ctr_q = beta_quantiles(ctr_a, ctr_b)
        cvr_q = beta_quantiles(cvr_a, cvr_b)

        # AOV: if provided use as point; else sample from lognormal prior
        if run["aov"] and run["aov"] > 0:
            aov_samples = np.full(5000, float(run["aov"]))
        else:
            aov_samples = np.random.lognormal(mean=PRIORS.aov_mu, sigma=PRIORS.aov_sigma, size=5000)

        prog.progress(75, text="Quant forecast…")

        # 6) Quant forecast samples
        # Market demand baseline (3mi) -> potential orders in area
        pop = run["assumptions"]["pop_3mi"]
        pen = run["assumptions"]["delivery_pen"]
        freq = run["assumptions"]["freq"]

        base_market_orders = pop * pen * freq  # total market orders/month (all restaurants)
        # Restaurant share: proxy using competitor trust score rank (very simplified)
        # In production: multinomial logit choice model.
        my_trust = 3.8 * math.log1p(200)  # placeholder; if you have Yelp/Google rating & reviews you plug in
        total_trust = my_trust + float(comp_df["trust_score"].clip(lower=0).sum())
        share = my_trust / total_trust if total_trust > 0 else 0.03
        share = float(max(0.01, min(0.12, share)))  # conservative cap

        # Bayesian conversion adjustment factor: compare posterior median to prior mean
        ctr_prior_mean = PRIORS.ctr_alpha / (PRIORS.ctr_alpha + PRIORS.ctr_beta)
        cvr_prior_mean = PRIORS.cvr_alpha / (PRIORS.cvr_alpha + PRIORS.cvr_beta)
        adj = float((ctr_q[0.5] / ctr_prior_mean) * (cvr_q[0.5] / cvr_prior_mean))
        adj = float(max(0.6, min(1.6, adj)))

        # Sample the posterior for uncertainty
        ctr_s = beta_dist.rvs(ctr_a, ctr_b, size=5000)
        cvr_s = beta_dist.rvs(cvr_a, cvr_b, size=5000)

        # Core predicted orders for the restaurant:
        # market_orders * share * weather * traffic * (ctr, cvr adjustment proxy)
        # We use ctr_s/cvr_s relative to prior means to model uncertainty.
        orders_samples = base_market_orders * share * w_coef * t_coef * (ctr_s/ctr_prior_mean) * (cvr_s/cvr_prior_mean)
        orders_samples = np.clip(orders_samples, 50, None)

        revenue_samples = orders_samples * aov_samples
        p10_o, p50_o, p90_o = np.percentile(orders_samples, [10, 50, 90])
        p10_r, p50_r, p90_r = np.percentile(revenue_samples, [10, 50, 90])

        # Growth probabilities vs baseline (use baseline share and priors)
        baseline_orders = base_market_orders * share * w_coef * t_coef
        growth = (orders_samples / max(1.0, baseline_orders)) - 1.0
        p20 = float(np.mean(growth >= 0.20))
        p40 = float(np.mean(growth >= 0.40))
        p60 = float(np.mean(growth >= 0.60))

        prog.progress(100, text="Done")

        # -----------------------------
        # Report rendering
        # -----------------------------
        left, mid, right = st.columns([1.2, 2.8, 1.3])

        with left:
            st.markdown("### TOC")
            st.markdown(f"- {t('geo')}")
            st.markdown(f"- {t('trade_area')}")
            st.markdown(f"- {t('competitors')}")
            st.markdown(f"- {t('weather_traffic')}")
            st.markdown(f"- {t('funnel')}")
            st.markdown(f"- {t('bayes')}")
            st.markdown(f"- {t('quant')}")
            st.markdown(f"- {t('plan')}")
            st.markdown(f"- {t('abtests')}")
            st.markdown(f"- {t('kpi')}")

        with right:
            st.markdown("### " + t("assumptions"))
            st.json({
                "radius_miles_core": run["assumptions"]["radius_miles_core"],
                "pop_1mi_fallback": run["assumptions"]["pop_1mi"],
                "pop_3mi_fallback": run["assumptions"]["pop_3mi"],
                "median_income_fallback": run["assumptions"]["median_income"],
                "delivery_penetration": run["assumptions"]["delivery_pen"],
                "monthly_frequency": run["assumptions"]["freq"],
                "weather_coef": round(w_coef, 3),
                "traffic_coef": round(t_coef, 3),
                "traffic_congestion_index": round(cong, 3),
                "bayes_adj_factor": round(adj, 3),
            })
            st.markdown("### " + t("data_sources"))
            sources = []
            if google_key:
                sources.append("Google Geocoding/Places/DistanceMatrix (if enabled)")
            else:
                sources.append("Geo/competitors fallback (demo)")
            if openweather_key:
                sources.append("OpenWeather (if enabled)")
            else:
                sources.append("Weather fallback (demo)")
            if census_key:
                sources.append("US Census ACS (not fully wired in demo ring model)")
            else:
                sources.append("Census fallback (demo)")
            st.write("- " + "\n- ".join(sources))

        with mid:
            st.markdown(f"## {run['name']}")
            st.caption(run["address"])
            st.markdown(f"**{t('cuisine')}**: {run['cuisine']}  \n**Platforms**: {', '.join(run['platforms']) if run['platforms'] else '-'}")

            st.markdown("---")
            st.markdown(f"### {t('geo')}")
            st.write(f"{t('latlng')}: `{latlng[0]:.5f}, {latlng[1]:.5f}`  ({geo_note})")

            st.markdown(f"### {t('trade_area')}")
            st.write(f"- 1mi Population (fallback): **{run['assumptions']['pop_1mi']:,}**")
            st.write(f"- 3mi Population (fallback): **{run['assumptions']['pop_3mi']:,}**")
            st.write(f"- Median income (fallback): **${run['assumptions']['median_income']:,}**")

            st.markdown(f"### {t('competitors')}")
            st.dataframe(comp_df.sort_values(["tier","trust_score"], ascending=[True, False]).reset_index(drop=True), use_container_width=True, height=280)

            st.markdown(f"### {t('weather_traffic')}")
            wt1, wt2, wt3 = st.columns(3)
            with wt1:
                st.metric("Temp (F)", f"{temp_f:.1f}")
            with wt2:
                st.metric("Precip (mm)", f"{precip_mm:.2f}")
            with wt3:
                st.metric("Weather coef", f"{w_coef:.2f}")
            st.write(f"- Traffic congestion index (proxy): **{cong:.2f}x**")
            st.write(f"- Traffic coefficient: **{t_coef:.2f}**")

            st.markdown(f"### {t('funnel')}")
            # Funnel: if user provided metrics, compute rough funnel counts
            if run["impressions"] and run["ctr"] and run["cvr"]:
                clicks = run["impressions"] * run["ctr"]
                orders = clicks * run["cvr"]
                funnel_df = pd.DataFrame({
                    "Stage": ["Impressions", "Clicks", "Orders"],
                    "Value": [run["impressions"], int(clicks), int(orders)]
                })
            else:
                funnel_df = pd.DataFrame({
                    "Stage": ["Impressions", "Clicks", "Orders"],
                    "Value": [10000, int(10000 * ctr_q[0.5]), int(10000 * ctr_q[0.5] * cvr_q[0.5])]
                })
            st.dataframe(funnel_df, use_container_width=True, height=160)

            st.markdown(f"### {t('bayes')}")
            st.write("**CTR posterior (P10/P50/P90)**:", f"{ctr_q[0.1]*100:.1f}% / {ctr_q[0.5]*100:.1f}% / {ctr_q[0.9]*100:.1f}%")
            st.write("**CVR posterior (P10/P50/P90)**:", f"{cvr_q[0.1]*100:.1f}% / {cvr_q[0.5]*100:.1f}% / {cvr_q[0.9]*100:.1f}%")
            st.write(f"Posterior notes: CTR used pseudo trials = {ctr_pseudo}, CVR used pseudo trials = {cvr_pseudo}")

            st.markdown(f"### {t('quant')}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Predicted Orders / month (P10)", f"{p10_o:,.0f}")
                st.metric("Predicted Orders / month (P50)", f"{p50_o:,.0f}")
                st.metric("Predicted Orders / month (P90)", f"{p90_o:,.0f}")
            with c2:
                st.metric("Revenue / month (P10)", f"${p10_r:,.0f}")
                st.metric("Revenue / month (P50)", f"${p50_r:,.0f}")
                st.metric("Revenue / month (P90)", f"${p90_r:,.0f}")

            st.write("**Probability of hitting growth targets (vs baseline)**")
            st.write(f"- P(growth ≥ +20%): **{p20*100:.0f}%**")
            st.write(f"- P(growth ≥ +40%): **{p40*100:.0f}%**")
            st.write(f"- P(growth ≥ +60%): **{p60*100:.0f}%**")

            st.markdown(f"### {t('plan')}")
            st.write("**30天（快赢）**")
            st.write("- 重构菜单首屏：Top 3 招牌 + 2个高毛利加购（用图片强化）")
            st.write("- 建立天气触发促销：雨/冷天推热食组合；好天气推便当/轻食")
            st.write("- 优化履约：出餐时间承诺、打包稳定（降低差评与取消）")
            st.write("**60天（结构优化）**")
            st.write("- 建立复购机制：次卡/第二单券/家庭套餐订阅")
            st.write("- 竞对对标：用“Why they win / How we counter”矩阵每周迭代")
            st.write("**90天（规模化）**")
            st.write("- 形成爆款 SKU 矩阵：1个引流 + 2个利润 + 1个口碑")
            st.write("- 开启系统化A/B测试与仪表盘警报（CTR/CVR/ETA）")

            st.markdown(f"### {t('abtests')}")
            ab = pd.DataFrame([
                {"Experiment":"Menu structure", "Hypothesis":"场景化首屏提升CTR与下单率", "A":"传统分类", "B":"3个套餐+爆款置顶", "Metric":"CTR, CVR", "Duration":"14 days"},
                {"Experiment":"Price anchoring", "Hypothesis":"锚点提升AOV", "A":"先展示低价单品", "B":"先展示Mix/Combo", "Metric":"AOV", "Duration":"14 days"},
                {"Experiment":"Promo mechanic", "Hypothesis":"加购折扣优于满减", "A":"满减", "B":"加购第二件折扣", "Metric":"Orders, Margin", "Duration":"21 days"},
            ])
            st.dataframe(ab, use_container_width=True, height=180)

            st.markdown(f"### {t('kpi')}")
            kpi = pd.DataFrame([
                {"KPI":"Impressions", "Target":"↑", "Alert":"-20% WoW"},
                {"KPI":"CTR", "Target":"8%–12% (industry)", "Alert":"< 6%"},
                {"KPI":"CVR", "Target":"20%–30% (industry)", "Alert":"< 15%"},
                {"KPI":"AOV", "Target":"$24–$32 (segment)", "Alert":"< $22"},
                {"KPI":"ETA delay (traffic)", "Target":"< 1.25x", "Alert":"> 1.5x"},
                {"KPI":"Negative review rate", "Target":"< 8%", "Alert":"> 12%"},
            ])
            st.dataframe(kpi, use_container_width=True, height=220)

# -----------------------------
# Debug tab
# -----------------------------
with tab_debug:
    st.subheader("Debug")
    st.write("Session state:", st.session_state.get("run", {}))
    st.write("Tips:")
    st.write("- Add API keys to enable real geocoding / competitors / weather.")
    st.write("- Next iteration: implement exact 1mi/3mi ring -> census tract mapping + ACS pull.")
