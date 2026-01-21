# app.py
# Streamlit: 输入餐厅地址/店名 -> Google Places 匹配 -> 用户确认 -> 拉取公开数据
# (Google/Yelp/Census+TIGER/NOAA+Meteostat/WalkScore可选) -> 调用 OpenAI 生成深度报告 -> 多语言 -> 导出PDF
#
# ✅ 重点：天气已改为 NOAA + Meteostat（无需注册、稳定、适合分析）
#
# ---------------------------
# 安装依赖
# ---------------------------
# pip install streamlit requests python-dateutil pandas reportlab meteostat
# 可选（更漂亮PDF）: pip install markdown2 weasyprint
#
# ---------------------------
# 环境变量
# ---------------------------
# GOOGLE_MAPS_API_KEY=...
# YELP_API_KEY=...
# OPENAI_API_KEY=...
# OPENAI_MODEL=gpt-4o-mini   (按你账号可用模型调整)
# CENSUS_API_KEY=...         (可选：无key也能用部分Census，但建议有)
# WALKSCORE_API_KEY=...      (可选)
#
# ---------------------------
# 免责声明（产品级做法）
# ---------------------------
# - 1mi/3mi 人口等“半径商圈”是基于 Census Tract/County 密度近似估算，不等同于精确环形叠加统计。
# - 竞品清单来自 Yelp/Google 周边检索，可能漏掉未收录或新店。
#
# Author: you + me (严谨版)

import os
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

from meteostat import Point, Daily

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="AuraInsight · 餐厅商圈与增长分析", layout="wide")

# -----------------------------
# ENV
# -----------------------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
YELP_API_KEY = os.getenv("YELP_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
WALKSCORE_API_KEY = os.getenv("WALKSCORE_API_KEY", "")

USER_AGENT = "AuraInsight-Analyzer/1.0"

DEFAULT_RADII_MI = [1, 3]  # 样板：1 mile / 3 mile


# -----------------------------
# Errors
# -----------------------------
class APIError(Exception):
    pass


# -----------------------------
# HTTP helpers + cache
# -----------------------------
def _req_json(
    method: str,
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    timeout: int = 30,
    retries: int = 2,
    backoff: float = 1.4,
) -> dict:
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)

    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=timeout,
            )
            if r.status_code == 429:
                time.sleep(backoff ** (i + 1))
                continue
            if r.status_code >= 400:
                raise APIError(f"{url} -> HTTP {r.status_code}: {r.text[:500]}")
            if "application/json" in r.headers.get("content-type", ""):
                return r.json()
            # NOAA有时 text/json
            return json.loads(r.text)
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i + 1))
    raise APIError(f"Request failed after retries: {url} | {repr(last_err)}")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)  # 24h
def cached_http_json(method: str, url: str, params_key: str, headers_key: str, body_key: str) -> dict:
    params = json.loads(params_key) if params_key else None
    headers = json.loads(headers_key) if headers_key else None
    body = json.loads(body_key) if body_key else None
    return _req_json(method, url, headers=headers, params=params, json_body=body)


def http_json_cached(method: str, url: str, params: Optional[dict] = None, headers: Optional[dict] = None, body: Optional[dict] = None) -> dict:
    return cached_http_json(
        method=method,
        url=url,
        params_key=json.dumps(params or {}, sort_keys=True),
        headers_key=json.dumps(headers or {}, sort_keys=True),
        body_key=json.dumps(body or {}, sort_keys=True),
    )


# -----------------------------
# Google Places
# -----------------------------
@dataclass
class PlaceCandidate:
    name: str
    place_id: str
    address: str
    lat: float
    lng: float
    types: List[str]
    rating: Optional[float]
    user_ratings_total: Optional[int]


def google_text_search(query: str) -> List[PlaceCandidate]:
    if not GOOGLE_MAPS_API_KEY:
        raise APIError("Missing GOOGLE_MAPS_API_KEY")
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_MAPS_API_KEY}
    data = http_json_cached("GET", url, params=params)

    results = data.get("results", []) or []
    out: List[PlaceCandidate] = []
    for r in results[:10]:
        loc = (r.get("geometry") or {}).get("location") or {}
        out.append(
            PlaceCandidate(
                name=r.get("name", "") or "",
                place_id=r.get("place_id", "") or "",
                address=r.get("formatted_address", "") or "",
                lat=float(loc.get("lat", 0.0) or 0.0),
                lng=float(loc.get("lng", 0.0) or 0.0),
                types=r.get("types") or [],
                rating=r.get("rating"),
                user_ratings_total=r.get("user_ratings_total"),
            )
        )
    return out


def google_place_details(place_id: str) -> dict:
    if not GOOGLE_MAPS_API_KEY:
        raise APIError("Missing GOOGLE_MAPS_API_KEY")
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = ",".join(
        [
            "name",
            "place_id",
            "formatted_address",
            "geometry",
            "types",
            "rating",
            "user_ratings_total",
            "opening_hours",
            "website",
            "formatted_phone_number",
            "price_level",
            "business_status",
        ]
    )
    params = {"place_id": place_id, "fields": fields, "key": GOOGLE_MAPS_API_KEY, "language": "en"}
    data = http_json_cached("GET", url, params=params)
    if data.get("status") != "OK":
        raise APIError(f"Google Place Details failed: {data.get('status')} {data.get('error_message','')}")
    return data["result"]


def google_nearby_search(lat: float, lng: float, radius_m: int, keyword: str = "", type_: str = "restaurant") -> List[dict]:
    if not GOOGLE_MAPS_API_KEY:
        raise APIError("Missing GOOGLE_MAPS_API_KEY")
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"location": f"{lat},{lng}", "radius": radius_m, "type": type_, "key": GOOGLE_MAPS_API_KEY}
    if keyword:
        params["keyword"] = keyword
    data = http_json_cached("GET", url, params=params)
    return (data.get("results") or [])[:20]


# -----------------------------
# Yelp Fusion
# -----------------------------
def yelp_headers() -> dict:
    if not YELP_API_KEY:
        raise APIError("Missing YELP_API_KEY")
    return {"Authorization": f"Bearer {YELP_API_KEY}", "User-Agent": USER_AGENT}


def yelp_business_search(name: str, lat: float, lng: float) -> Optional[dict]:
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "term": name,
        "latitude": lat,
        "longitude": lng,
        "limit": 5,
        "sort_by": "best_match",
    }
    data = http_json_cached("GET", url, params=params, headers=yelp_headers())
    businesses = data.get("businesses") or []
    return businesses[0] if businesses else None


def yelp_business_details(business_id: str) -> dict:
    url = f"https://api.yelp.com/v3/businesses/{business_id}"
    return http_json_cached("GET", url, headers=yelp_headers())


def yelp_business_reviews(business_id: str) -> dict:
    url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
    return http_json_cached("GET", url, headers=yelp_headers())


def yelp_competitors(lat: float, lng: float, radius_m: int = 4800, categories: Optional[str] = None) -> List[dict]:
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "latitude": lat,
        "longitude": lng,
        "radius": min(radius_m, 40000),
        "limit": 20,
        "sort_by": "rating",
    }
    if categories:
        params["categories"] = categories
    data = http_json_cached("GET", url, params=params, headers=yelp_headers())
    return data.get("businesses") or []


# -----------------------------
# Weather: NOAA + Meteostat
# -----------------------------
def noaa_points(lat: float, lng: float) -> dict:
    url = f"https://api.weather.gov/points/{lat:.4f},{lng:.4f}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    return http_json_cached("GET", url, headers=headers)


def noaa_forecast(lat: float, lng: float) -> dict:
    p = noaa_points(lat, lng)
    forecast_url = (p.get("properties") or {}).get("forecast")
    if not forecast_url:
        raise APIError("NOAA points missing forecast url")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    return http_json_cached("GET", forecast_url, headers=headers)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def meteostat_daily(lat: float, lng: float, days: int = 365) -> pd.DataFrame:
    # Meteostat: 学术级历史数据（无需key）
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    location = Point(lat, lng)
    df = Daily(location, start, end).fetch()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # 字段：tavg/tmin/tmax/prcp/snow/wspd/...
    return df


def summarize_weather(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "days": 0,
            "rain_days": None,
            "heavy_rain_days": None,
            "hot_days": None,
            "cold_days": None,
            "avg_tavg_c": None,
            "total_prcp_mm": None,
        }

    days = len(df)
    # prcp: mm (Meteostat)
    rain_days = int((df["prcp"].fillna(0) > 0.5).sum())
    heavy_rain_days = int((df["prcp"].fillna(0) > 10).sum())
    # tmax/tmin: °C
    hot_days = int((df["tmax"].fillna(-999) > 30).sum())   # >86°F
    cold_days = int((df["tmin"].fillna(999) < 5).sum())    # <41°F
    avg_tavg = float(df["tavg"].dropna().mean()) if df["tavg"].notna().any() else None
    total_prcp = float(df["prcp"].fillna(0).sum())

    return {
        "days": days,
        "rain_days": rain_days,
        "heavy_rain_days": heavy_rain_days,
        "hot_days": hot_days,
        "cold_days": cold_days,
        "avg_tavg_c": avg_tavg,
        "total_prcp_mm": total_prcp,
    }


# -----------------------------
# Census: FCC -> GEOIDs -> ACS + TIGER land area
# -----------------------------
def fcc_block_geoid(lat: float, lng: float) -> dict:
    # FCC Census Block API: lat/lon -> state/county/tract/block FIPS
    url = "https://geo.fcc.gov/api/census/block/find"
    params = {"latitude": lat, "longitude": lng, "format": "json"}
    return http_json_cached("GET", url, params=params)


def tiger_tract_land_area(state: str, county: str, tract: str) -> Optional[int]:
    # TIGERweb: get ALAND for tract (m^2)
    # layer: Census Tracts
    # https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/2/query
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/2/query"
    geoid = f"{state}{county}{tract}"
    params = {
        "where": f"GEOID='{geoid}'",
        "outFields": "ALAND,GEOID,NAME",
        "f": "json",
    }
    data = http_json_cached("GET", url, params=params)
    feats = data.get("features") or []
    if not feats:
        return None
    attrs = feats[0].get("attributes") or {}
    aland = attrs.get("ALAND")
    return int(aland) if aland is not None else None


def tiger_county_land_area(state: str, county: str) -> Optional[int]:
    # TIGERweb Counties layer
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/County/MapServer/0/query"
    geoid = f"{state}{county}"
    params = {
        "where": f"GEOID='{geoid}'",
        "outFields": "ALAND,GEOID,NAME",
        "f": "json",
    }
    data = http_json_cached("GET", url, params=params)
    feats = data.get("features") or []
    if not feats:
        return None
    attrs = feats[0].get("attributes") or {}
    aland = attrs.get("ALAND")
    return int(aland) if aland is not None else None


def acs_tract_profile(state: str, county: str, tract: str) -> dict:
    # ACS 5-year: tract-level key vars
    # variables:
    # B01003_001E total population
    # B19013_001E median household income
    # DP05_0037PE Asian alone percent (DP05_0037PE)
    # DP05_0071PE Hispanic percent (DP05_0071PE)
    # DP05_0033PE White alone percent (DP05_0033PE)
    # DP02_0001E households (DP02_0001E)
    #
    # DP* tables are in /profile endpoint.
    base = "https://api.census.gov/data/2022/acs/acs5/profile"
    vars_ = [
        "DP05_0001E",   # Total population (profile)
        "DP02_0001E",   # Households
        "DP03_0062E",   # Median household income (approx; DP03 varies, this is "Median household income" in profile)
        "DP05_0033PE",  # White %
        "DP05_0037PE",  # Asian %
        "DP05_0071PE",  # Hispanic %
        "NAME",
    ]
    params = {
        "get": ",".join(vars_),
        "for": f"tract:{tract}",
        "in": f"state:{state} county:{county}",
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY

    data = http_json_cached("GET", base, params=params)
    # First row headers, second row values
    if not isinstance(data, list) or len(data) < 2:
        raise APIError("Census ACS response unexpected")
    headers = data[0]
    values = data[1]
    out = dict(zip(headers, values))
    return out


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def estimate_radius_population(
    radius_miles: float,
    tract_pop: Optional[int],
    tract_aland_m2: Optional[int],
    county_pop: Optional[int],
    county_aland_m2: Optional[int],
) -> Tuple[Optional[int], str]:
    """
    半径人口近似：
    - 1mi：优先用 tract 密度估算并上限=tract_pop
    - 3mi：优先用 county 密度估算（范围更大），避免tract过小导致失真
    """
    r_m = radius_miles * 1609.344
    area_circle = math.pi * (r_m ** 2)

    def density(pop: Optional[int], aland: Optional[int]) -> Optional[float]:
        if pop is None or aland is None or aland <= 0:
            return None
        return pop / aland  # people per m^2

    if radius_miles <= 1.5:
        d = density(tract_pop, tract_aland_m2)
        if d is None:
            return None, "density_missing"
        est = int(d * area_circle)
        if tract_pop is not None:
            est = min(est, tract_pop)
        return max(est, 0), "tract_density"
    else:
        d = density(county_pop, county_aland_m2)
        if d is None:
            return None, "density_missing"
        est = int(d * area_circle)
        return max(est, 0), "county_density"


def census_bundle_from_latlng(lat: float, lng: float) -> dict:
    fcc = fcc_block_geoid(lat, lng)
    block = (fcc.get("Block") or {})
    fips = block.get("FIPS")
    if not fips or len(fips) < 11:
        raise APIError("FCC did not return valid FIPS")
    # FIPS: SSCCCTTTTTTBBBB -> take needed pieces
    state = fips[0:2]
    county = fips[2:5]
    tract = fips[5:11]

    acs = acs_tract_profile(state, county, tract)

    tract_pop = safe_int(acs.get("DP05_0001E"))
    households = safe_int(acs.get("DP02_0001E"))
    med_income = safe_int(acs.get("DP03_0062E"))
    pct_white = safe_float(acs.get("DP05_0033PE"))
    pct_asian = safe_float(acs.get("DP05_0037PE"))
    pct_hisp = safe_float(acs.get("DP05_0071PE"))

    tract_aland = tiger_tract_land_area(state, county, tract)
    county_aland = tiger_county_land_area(state, county)

    # county population：用 ACS5（non-profile endpoint）更稳，但这里用 profile county 也行
    # 为简化：用同profile接口 county层
    base = "https://api.census.gov/data/2022/acs/acs5/profile"
    params = {"get": "DP05_0001E,NAME", "for": f"county:{county}", "in": f"state:{state}"}
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    county_data = http_json_cached("GET", base, params=params)
    county_pop = None
    if isinstance(county_data, list) and len(county_data) >= 2:
        headers = county_data[0]
        values = county_data[1]
        row = dict(zip(headers, values))
        county_pop = safe_int(row.get("DP05_0001E"))

    return {
        "fcc": fcc,
        "state_fips": state,
        "county_fips": county,
        "tract": tract,
        "tract_name": acs.get("NAME"),
        "tract_pop": tract_pop,
        "households": households,
        "median_household_income": med_income,
        "pct_white": pct_white,
        "pct_asian": pct_asian,
        "pct_hispanic": pct_hisp,
        "tract_aland_m2": tract_aland,
        "county_pop": county_pop,
        "county_aland_m2": county_aland,
    }


# -----------------------------
# WalkScore (optional)
# -----------------------------
def walkscore(lat: float, lng: float, address: str) -> Optional[dict]:
    if not WALKSCORE_API_KEY:
        return None
    url = "https://api.walkscore.com/score"
    params = {
        "format": "json",
        "address": address,
        "lat": lat,
        "lon": lng,
        "transit": 1,
        "bike": 1,
        "wsapikey": WALKSCORE_API_KEY,
    }
    try:
        return http_json_cached("GET", url, params=params)
    except Exception:
        return None


# -----------------------------
# OpenAI (via REST) - generate & translate
# -----------------------------
def openai_headers() -> dict:
    if not OPENAI_API_KEY:
        raise APIError("Missing OPENAI_API_KEY")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def openai_chat(prompt_system: str, prompt_user: str, temperature: float = 0.3, max_tokens: int = 3500) -> str:
    """
    Uses Chat Completions compatible endpoint. If your account requires Responses API,
    you can switch. This one works for many setups.
    """
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": OPENAI_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
    }
    data = _req_json("POST", url, headers=openai_headers(), json_body=body, timeout=90, retries=1)
    choices = data.get("choices") or []
    if not choices:
        raise APIError("OpenAI returned no choices")
    return (choices[0].get("message") or {}).get("content") or ""


def translate_text(text: str, target_lang: str) -> str:
    # target_lang: "zh" or "en"
    if target_lang not in ("zh", "en"):
        return text
    sys = "You are a professional business report translator. Preserve structure, headings, tables, [FACT]/[INFERENCE]/[ASSUMPTION]/[STRATEGY] tags. Do not add or remove content."
    if target_lang == "zh":
        user = f"Translate to Simplified Chinese:\n\n{text}"
    else:
        user = f"Translate to English:\n\n{text}"
    return openai_chat(sys, user, temperature=0.1, max_tokens=3500)


# -----------------------------
# Report builder: data bundle -> prompt -> report markdown
# -----------------------------
def miles_to_meters(mi: float) -> int:
    return int(mi * 1609.344)


def build_data_bundle(place: dict) -> dict:
    name = place.get("name", "")
    address = place.get("formatted_address", "")
    geom = (place.get("geometry") or {}).get("location") or {}
    lat = float(geom.get("lat", 0.0) or 0.0)
    lng = float(geom.get("lng", 0.0) or 0.0)

    # Yelp
    yelp_match = None
    yelp_details = None
    yelp_reviews = None
    competitors_yelp = []
    try:
        if YELP_API_KEY:
            yelp_match = yelp_business_search(name, lat, lng)
            if yelp_match and yelp_match.get("id"):
                yelp_details = yelp_business_details(yelp_match["id"])
                yelp_reviews = yelp_business_reviews(yelp_match["id"])
            competitors_yelp = yelp_competitors(lat, lng, radius_m=miles_to_meters(3))
    except Exception as e:
        competitors_yelp = competitors_yelp or []
        yelp_details = yelp_details or None
        yelp_reviews = yelp_reviews or None

    # Google nearby competitors (backup)
    competitors_google = []
    try:
        competitors_google = google_nearby_search(lat, lng, radius_m=miles_to_meters(3), type_="restaurant")
    except Exception:
        competitors_google = []

    # Census
    census = {}
    try:
        census = census_bundle_from_latlng(lat, lng)
    except Exception as e:
        census = {"error": str(e)}

    # Radius estimates
    radius_stats = []
    for r in DEFAULT_RADII_MI:
        est_pop, method = estimate_radius_population(
            radius_miles=r,
            tract_pop=census.get("tract_pop"),
            tract_aland_m2=census.get("tract_aland_m2"),
            county_pop=census.get("county_pop"),
            county_aland_m2=census.get("county_aland_m2"),
        )
        radius_stats.append({"radius_miles": r, "est_population": est_pop, "method": method})

    # Weather: NOAA forecast + Meteostat history
    noaa_fc = None
    try:
        noaa_fc = noaa_forecast(lat, lng)
    except Exception as e:
        noaa_fc = {"error": str(e)}

    hist_df = meteostat_daily(lat, lng, days=365)
    weather_summary = summarize_weather(hist_df)

    # WalkScore optional
    ws = None
    try:
        ws = walkscore(lat, lng, address)
    except Exception:
        ws = None

    return {
        "place": place,
        "lat": lat,
        "lng": lng,
        "google": {
            "rating": place.get("rating"),
            "user_ratings_total": place.get("user_ratings_total"),
            "types": place.get("types"),
            "price_level": place.get("price_level"),
            "business_status": place.get("business_status"),
            "phone": place.get("formatted_phone_number"),
            "website": place.get("website"),
            "opening_hours": (place.get("opening_hours") or {}).get("weekday_text"),
        },
        "yelp": {
            "match": yelp_match,
            "details": yelp_details,
            "reviews": yelp_reviews,
            "competitors": competitors_yelp,
        },
        "competitors_google": competitors_google,
        "census": census,
        "radius_stats": radius_stats,
        "weather": {
            "noaa_forecast": noaa_fc,
            "meteostat_days": 365,
            "meteostat_summary": weather_summary,
        },
        "walkscore": ws,
        # (可扩展) safegraph / crime / traffic / POI density
    }


def compact_competitors(yelp_list: List[dict], google_list: List[dict]) -> List[dict]:
    """
    给模型的竞品输入必须“短而结构化”，否则prompt爆炸。
    """
    out = []
    # Yelp top 10
    for b in (yelp_list or [])[:10]:
        out.append(
            {
                "source": "yelp",
                "name": b.get("name"),
                "rating": b.get("rating"),
                "review_count": b.get("review_count"),
                "price": b.get("price"),
                "distance_m": b.get("distance"),
                "categories": [c.get("title") for c in (b.get("categories") or [])[:2]],
            }
        )
    # Google top 10
    for r in (google_list or [])[:10]:
        out.append(
            {
                "source": "google",
                "name": r.get("name"),
                "rating": r.get("rating"),
                "user_ratings_total": r.get("user_ratings_total"),
                "vicinity": r.get("vicinity"),
                "types": (r.get("types") or [])[:3],
            }
        )
    # 去重（按name）
    seen = set()
    uniq = []
    for x in out:
        n = (x.get("name") or "").strip().lower()
        if not n:
            continue
        if n in seen:
            continue
        seen.add(n)
        uniq.append(x)
    return uniq[:18]


def compact_noaa_forecast(noaa: dict) -> List[dict]:
    """
    NOAA forecast periods -> 简化（未来3-6个时段）
    """
    props = (noaa or {}).get("properties") or {}
    periods = props.get("periods") or []
    simple = []
    for p in periods[:6]:
        simple.append(
            {
                "name": p.get("name"),
                "temperature": p.get("temperature"),
                "temperatureUnit": p.get("temperatureUnit"),
                "windSpeed": p.get("windSpeed"),
                "shortForecast": p.get("shortForecast"),
            }
        )
    return simple


def make_report_prompt(bundle: dict, lang: str = "zh") -> Tuple[str, str]:
    place = bundle["place"]
    name = place.get("name", "")
    address = place.get("formatted_address", "")
    lat, lng = bundle["lat"], bundle["lng"]

    competitors = compact_competitors(bundle["yelp"].get("competitors") or [], bundle.get("competitors_google") or [])
    noaa_simple = compact_noaa_forecast(bundle["weather"].get("noaa_forecast") or {})
    weather_summary = bundle["weather"].get("meteostat_summary") or {}

    census = bundle.get("census") or {}
    radius_stats = bundle.get("radius_stats") or []

    # 用于“样板逻辑”输出的系统提示
    system = f"""
你是一个“餐厅商圈与增长分析”专家顾问，输出必须像专业咨询报告：结构清晰、推理严谨、可执行。
强制要求：
- 必须使用并保留标签：[FACT] [INFERENCE] [ASSUMPTION] [STRATEGY]
- 结论必须基于输入数据；没有数据就明确标注[ASSUMPTION]，不要编造具体数值。
- 报告必须包含：1) Trade Area Intelligence（1mi/3mi） 2) 竞对与替代结构 3) 转化漏斗诊断 4) 天气/季节/交通影响 5) 30/60/90天动作清单
- 报告语言：{"简体中文" if lang=="zh" else "English"}。标题与段落也要对应语言。
- 字体格式：用Markdown标题/表格呈现关键矩阵；避免超长无结构段落。
"""

    # 用户提示：把数据喂给模型（短而关键）
    user = f"""
请为以下餐厅生成《商圈与增长分析报告》，风格对齐我提供的样板（偏“麦肯锡式”但直白可落地）。

餐厅信息（来自Google Places）：
- 门店：{name}
- 地址：{address}
- 坐标：{lat:.5f},{lng:.5f}
- Google评分/评论数：{bundle["google"].get("rating")} / {bundle["google"].get("user_ratings_total")}
- 类型：{bundle["google"].get("types")}
- 价格等级(price_level)：{bundle["google"].get("price_level")}
- 营业状态：{bundle["google"].get("business_status")}
- 电话：{bundle["google"].get("phone")}
- 网站：{bundle["google"].get("website")}

Yelp（若匹配到）：
- Yelp匹配：{(bundle["yelp"].get("match") or {}).get("name")}
- Yelp评分/评论数：{(bundle["yelp"].get("details") or {}).get("rating")} / {(bundle["yelp"].get("details") or {}).get("review_count")}
- Yelp价位：{(bundle["yelp"].get("details") or {}).get("price")}
- Yelp类别：{[(c.get("title")) for c in ((bundle["yelp"].get("details") or {}).get("categories") or [])[:4]]}
- Yelp近3条评论摘录（如有）：
{[(r.get("text") or "")[:180] for r in ((bundle["yelp"].get("reviews") or {}).get("reviews") or [])[:3]]}

人口与消费能力（Census ACS + TIGER，可能为近似）：
- tract：{census.get("tract_name")} (state_fips={census.get("state_fips")}, county_fips={census.get("county_fips")}, tract={census.get("tract")})
- tract人口：{census.get("tract_pop")}
- 家庭户数：{census.get("households")}
- 家庭收入中位数（USD，可能为空）：{census.get("median_household_income")}
- 族裔比例（%）：White={census.get("pct_white")}, Asian={census.get("pct_asian")}, Hispanic={census.get("pct_hispanic")}
- 1mi/3mi 半径人口估算（基于密度近似，不是精确环统计）：
{radius_stats}

步行/交通（可选WalkScore）：
{bundle.get("walkscore")}

天气（NOAA预测 + Meteostat历史365天汇总）：
- NOAA未来预测（简化6条）：
{noaa_simple}
- Meteostat历史汇总（365天）：
{weather_summary}
解释提示：Meteostat温度为摄氏，prcp为毫米。

竞对池（Yelp/Google周边检索合并去重，最多18条）：
{competitors}

必须输出的关键结构（严格遵守）：
一、商圈人口与消费能力模型（Trade Area Intelligence）
- 1mi/3mi：人口、收入、族裔（如果缺字段，给区间[ASSUMPTION]并解释为何）
- 给出“潜在订单容量模型”的可计算公式，并用你给的数据/假设算出区间

二、竞对与替代性结构分析
- 竞对定义：抢同一顿饭预算/同一场景
- 分A/B/C类（直接/替代/体验大店或目的地）
- 至少挑3-6个竞对写“为什么会抢单 + DT如何反制”的结构
- 输出一个量化矩阵表（评分/评数/心智/场景/反制点）

三、转化漏斗诊断模型
- 订单 = 曝光 × CTR × CVR × 复购率
- 给行业健康值区间，并结合本店线上信任信号（评分/评数/评论内容）推断断点
- 断点必须落到“可操作动作”

四、天气/季节/交通影响
- 用Meteostat雨天/高温/低温天数来论证季节性
- 给“雨天外卖提升”之类结论必须标[ASSUMPTION]，并给合理区间
- 给出可执行的运营动作（菜单/配送/时段定价/促销）

五、30/60/90天行动清单（必须量化KPI）
- 每条动作：目标指标、预期影响路径、执行成本等级（低/中/高）
- 目标要写得像内部决策会使用的版本

输出必须是Markdown（可直接导出PDF），不要输出代码。
"""

    return system.strip(), user.strip()


# -----------------------------
# PDF export
# -----------------------------
def markdown_to_pdf_bytes(markdown_text: str, title: str = "Report") -> bytes:
    """
    优先：weasyprint（更漂亮）
    兜底：reportlab（保证能导出）
    """
    # Try WeasyPrint
    try:
        import markdown2
        from weasyprint import HTML, CSS

        html_body = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks"])
        html = f"""
        <html>
          <head>
            <meta charset="utf-8">
            <style>
              body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.35; }}
              h1, h2, h3 {{ margin: 0.6em 0 0.3em; }}
              table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0; }}
              th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
              th {{ background: #f5f5f5; }}
              code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }}
              .meta {{ color: #666; font-size: 12px; }}
            </style>
          </head>
          <body>
            {html_body}
          </body>
        </html>
        """
        pdf = HTML(string=html).write_pdf(stylesheets=[CSS(string="")])
        return pdf
    except Exception:
        pass

    # Fallback: ReportLab simple
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    # Optional: load a CJK font if available in system (best effort)
    # If you have a font file, set FONT_PATH env to it.
    font_name = "Helvetica"
    font_path = os.getenv("FONT_PATH", "")
    if font_path and os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("CustomFont", font_path))
            font_name = "CustomFont"
        except Exception:
            font_name = "Helvetica"

    import io
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setTitle(title)

    x = 50
    y = height - 50
    c.setFont(font_name, 10)

    # naive wrap lines (Markdown shown as plain text in fallback)
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip("\n")
        if not line:
            y -= 12
            continue
        # wrap
        for seg in wrap_text(line, max_chars=95):
            if y < 60:
                c.showPage()
                c.setFont(font_name, 10)
                y = height - 50
            c.drawString(x, y, seg)
            y -= 12

    c.save()
    buffer.seek(0)
    return buffer.read()


def wrap_text(s: str, max_chars: int = 90) -> List[str]:
    out = []
    while len(s) > max_chars:
        out.append(s[:max_chars])
        s = s[max_chars:]
    out.append(s)
    return out


# -----------------------------
# UI
# -----------------------------
st.title("AuraInsight · 餐厅商圈与增长分析报告生成器")
st.caption("输入地址/店名 → 选中正确商家 → 拉取公开数据 → 生成深度分析报告（支持中英 & PDF导出）")

with st.sidebar:
    st.header("设置")
    lang = st.selectbox("报告语言", ["中文", "English"], index=0)
    lang_code = "zh" if lang == "中文" else "en"

    st.subheader("API 状态")
    st.write("Google Places:", "✅" if GOOGLE_MAPS_API_KEY else "❌")
    st.write("Yelp:", "✅" if YELP_API_KEY else "❌")
    st.write("OpenAI:", "✅" if OPENAI_API_KEY else "❌")
    st.write("Census key:", "✅" if CENSUS_API_KEY else "⚠️(可选)")
    st.write("WalkScore:", "✅" if WALKSCORE_API_KEY else "⚠️(可选)")
    st.divider()
    st.markdown(
        """
**天气数据来源**  
- NOAA (forecast) + Meteostat (历史365天)  
无需注册，适合做“雨天/高温/季节性”的量化分析。
"""
    )

query = st.text_input("餐厅地址 / 店名（建议：店名 + 城市）", value="")
colA, colB = st.columns([1, 1])

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "selected_place_id" not in st.session_state:
    st.session_state.selected_place_id = None
if "report_zh" not in st.session_state:
    st.session_state.report_zh = None
if "report_en" not in st.session_state:
    st.session_state.report_en = None
if "bundle" not in st.session_state:
    st.session_state.bundle = None


with colA:
    if st.button("🔎 搜索匹配商家", use_container_width=True, disabled=not query.strip()):
        try:
            with st.spinner("Google Places 搜索中..."):
                cands = google_text_search(query.strip())
            if not cands:
                st.warning("没有找到匹配结果，请尝试更具体的输入（店名 + 城市 + 州）。")
            st.session_state.candidates = cands
            st.session_state.selected_place_id = None
            st.session_state.report_zh = None
            st.session_state.report_en = None
            st.session_state.bundle = None
        except Exception as e:
            st.error(f"搜索失败：{e}")

with colB:
    clear = st.button("🧹 清空", use_container_width=True)
    if clear:
        st.session_state.candidates = []
        st.session_state.selected_place_id = None
        st.session_state.report_zh = None
        st.session_state.report_en = None
        st.session_state.bundle = None
        st.rerun()

# Candidate selection
if st.session_state.candidates:
    st.subheader("选择正确的商家（确认后再开始分析）")
    labels = []
    for i, c in enumerate(st.session_state.candidates):
        labels.append(
            f"{i+1}. {c.name} | {c.address} | ⭐{c.rating or 'NA'} ({c.user_ratings_total or 'NA'})"
        )
    idx = st.selectbox("匹配结果", list(range(len(labels))), format_func=lambda i: labels[i])
    chosen = st.session_state.candidates[idx]
    st.session_state.selected_place_id = chosen.place_id

    st.info(f"已选择：**{chosen.name}**  —  {chosen.address}")

    # Show quick map
    try:
        st.map(pd.DataFrame([{"lat": chosen.lat, "lon": chosen.lng}]).rename(columns={"lon": "lon"}))
    except Exception:
        pass

    # Generate report
    if st.button("🚀 开始分析并生成报告", type="primary", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("缺少 OPENAI_API_KEY，无法生成报告。")
        else:
            try:
                with st.spinner("拉取数据中（人口/竞品/天气）..."):
                    place = google_place_details(chosen.place_id)
                    bundle = build_data_bundle(place)
                    st.session_state.bundle = bundle

                with st.spinner("调用模型生成报告（长文）..."):
                    sys_prompt, user_prompt = make_report_prompt(bundle, lang="zh")  # 先生成中文基稿最稳
                    report_zh = openai_chat(sys_prompt, user_prompt, temperature=0.25, max_tokens=3500)
                    st.session_state.report_zh = report_zh

                # 若用户想英文：用翻译（更稳定，不会跑格式）
                if lang_code == "en":
                    with st.spinner("翻译成英文..."):
                        st.session_state.report_en = translate_text(report_zh, "en")

                st.success("报告生成完成。")
            except Exception as e:
                st.error(f"生成失败：{e}")

# Display report
report_to_show = None
if lang_code == "zh" and st.session_state.report_zh:
    report_to_show = st.session_state.report_zh
elif lang_code == "en":
    if st.session_state.report_en:
        report_to_show = st.session_state.report_en
    elif st.session_state.report_zh:
        # 还没翻译就现场翻译
        try:
            with st.spinner("翻译成英文..."):
                st.session_state.report_en = translate_text(st.session_state.report_zh, "en")
            report_to_show = st.session_state.report_en
        except Exception as e:
            st.error(f"翻译失败：{e}")
            report_to_show = st.session_state.report_zh

if report_to_show:
    st.divider()
    st.subheader("生成的报告（可直接导出PDF）")
    st.markdown(report_to_show)

    # Download PDF
    try:
        place_name = ""
        if st.session_state.bundle and st.session_state.bundle.get("place"):
            place_name = st.session_state.bundle["place"].get("name") or "Restaurant"
        file_name = f"{place_name}_TradeArea_Growth_Report_{lang_code}.pdf".replace(" ", "_")
        pdf_bytes = markdown_to_pdf_bytes(report_to_show, title=file_name)
        st.download_button(
            "⬇️ 下载 PDF",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"PDF导出失败（可先复制Markdown）：{e}")

# Debug panel (optional)
with st.expander("（可选）查看原始数据包/调试", expanded=False):
    st.write("bundle keys:", list((st.session_state.bundle or {}).keys()))
    st.json(st.session_state.bundle or {})
