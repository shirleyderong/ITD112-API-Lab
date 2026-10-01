import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import json
import hashlib
import time
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
import pandas as pd
from datetime import date

# Page Config
st.set_page_config(page_title="ITD112 Laboratory Exercise 1: API Integration", layout="wide")
st.title("Integrating APIs: Soil and Weather Data")
st.write("Inputs, request, response, parsing, joining, analysis one step at a time.")

# Constants & Default Sites
SITES = pd.DataFrame([
    ("Iligan City", 8.200, 124.300),
    ("Malaybalay", 8.150, 125.130),
    ("Valencia", 7.900, 125.090),
    ("Davao (Calinan)", 7.190, 125.460),
    ("General Santos", 6.150, 125.150),
], columns=["site", "lat", "lon"])

SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOIL_PROPERTIES = ["clay", "sand", "silt", "phh2o", "soc"]
SOIL_DEPTH = "0-5cm"
DAILY_VARS = ["temperature_2m_mean", "precipitation_sum", "et0_fao_evapotranspiration"]
TIMEZONE = "Asia/Manila"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COLORS = {"sky": "#0F6E7A", "sky_pale": "#A3C4D3", "sun": "#E6A11D"}

# User Inputs
chosen = st.multiselect("Sites", SITES["site"].tolist(), default=SITES["site"].tolist()[:3])
start = st.date_input("Start date", date(2025, 1, 1))
end = st.date_input("End date", date(2025, 12, 31))
fetch = st.button("Run the integration", type="primary")
# --- REQUEST ---
def full_url(url, params):
    """The exact URL that will be sent, with the query string encoded."""
    return requests.Request("GET", url, params=params).prepare().url

def build_weather_request(sites, start, end):
    """Open-Meteo accepts comma-separated coordinates, so one request covers every site."""
    params = {
        "latitude": ",".join(str(v) for v in sites["lat"]),
        "longitude": ",".join(str(v) for v in sites["lon"]),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    return {"api": "Open-Meteo", "label": "All sites", "url": WEATHER_URL,
            "params": params, "full_url": full_url(WEATHER_URL, params)}

def build_forecast_request(sites):
    """Fetch the next 16 days of forecast data."""
    params = {
        "latitude": ",".join(str(v) for v in sites["lat"]),
        "longitude": ",".join(str(v) for v in sites["lon"]),
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
        "forecast_days": 16
    }
    return {"api": "Open-Meteo Forecast", "label": "All sites", "url": FORECAST_URL,
            "params": params, "full_url": full_url(FORECAST_URL, params)}

def build_soil_request(site, lat, lon):
    """SoilGrids takes one point per request. 'property' repeats once per soil property."""
    params = [("lat", lat), ("lon", lon), ("depth", SOIL_DEPTH), ("value", "mean")]
    params += [("property", p) for p in SOIL_PROPERTIES] # repeated keys
    return {"api": "SoilGrids", "label": site, "url": SOIL_URL,
            "params": params, "full_url": full_url(SOIL_URL, params)}


CACHE_DIR = Path("cache")
SOILGRIDS_PAUSE_S = 12

@st.cache_resource
def http_session():
    """Session that retries on rate limiting (429) and server errors (5xx)."""
    retry = Retry(total=3, backoff_factor=2,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "itd112-lab1/1.0"
    return s

def send(req):
    """GET the request and return the response record. A response saved on disk is reused."""
    key = hashlib.sha1(
        (req["url"] + json.dumps(req["params"], sort_keys=True, default=str)).encode()
    ).hexdigest()
    
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        # cache hit - no network at all
        text = path.read_text(encoding="utf-8")
        return {"status": None, "from_cache": True, "data": json.loads(text)}
        
    r = http_session().get(req["url"], params=req["params"], timeout=60)
    r.raise_for_status()
    data = r.json()
    
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"status": r.status_code, "from_cache": False, "data": data}

# Filter sites based on selection
sites = SITES[SITES["site"].isin(chosen)].reset_index(drop=True)

# --- JSON PARSE ---
def parse_weather(data, sites):
    """Each site's 'daily' block holds parallel arrays; each array becomes a column."""
    results = data if isinstance(data, list) else [data] # one site returns an object
    frames = []
    for site, res in zip(sites["site"], results):
        df = pd.DataFrame(res["daily"])
        df.insert(0, "site", site)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def parse_soil(data, site):
    """One row per soil property. SoilGrids stores integers; dividing by d_factor gives conventional units."""
    rows = []
    for layer in data["properties"]["layers"]:
        raw = layer["depths"][0]["values"]["mean"]
        unit = layer["unit_measure"]
        rows.append({
            "site": site,
            "property": layer["name"],
            "raw_mean": raw,
            "mapped_units": unit.get("mapped_units"),
            "d_factor": unit["d_factor"],
            "value": np.nan if raw is None else raw / unit["d_factor"],
            "target_units": unit.get("target_units"),
        })
    return pd.DataFrame(rows)

def soil_table(soil_long, sites):
    """Pivot to one row per site, one column per property, and attach coordinates."""
    wide = (soil_long.pivot(index="site", columns="property", values="value")
            .reindex(columns=SOIL_PROPERTIES).reset_index())
    wide.columns.name = None
    return sites.merge(wide, on="site", how="left")

# --- INTEGRATE ---
def integrate(weather, soil):
    daily = weather.merge(soil, on="site", how="left") # the join
    daily["time"] = pd.to_datetime(daily["time"])
    daily["month"] = daily["time"].dt.month
    daily["water_balance_mm"] = (daily["precipitation_sum"] - daily["et0_fao_evapotranspiration"])
    
    summary = (daily.groupby("site", sort=False)
               .agg(rain_mm=("precipitation_sum", "sum"),
                    et0_mm=("et0_fao_evapotranspiration", "sum"),
                    water_balance_mm=("water_balance_mm", "sum"),
                    temp_mean_c=("temperature_2m_mean", "mean"))
               .reset_index().merge(soil, on="site"))
    return daily, summary

# --- CHARTS ---
def chart_q2(daily, site):
    """Monthly rainfall bars vs reference ET0 line, on the same mm axis."""
    m = (daily[daily["site"] == site].groupby("month")
         [["precipitation_sum", "et0_fao_evapotranspiration"]].sum()
         .reindex(range(1, 13), fill_value=0))
    
    deficit = m["precipitation_sum"] < m["et0_fao_evapotranspiration"]
    
    fig = go.Figure()
    fig.add_bar(x=MONTHS, y=m["precipitation_sum"], name="Rainfall (surplus month)",
                marker_color=np.where(deficit, COLORS["sky_pale"], COLORS["sky"]))
    fig.add_scatter(x=MONTHS, y=m["et0_fao_evapotranspiration"], name="Reference ET0",
                    mode="lines+markers", line=dict(color=COLORS["sun"], width=2.5))
    fig.update_yaxes(title="mm per month", rangemode="tozero")
    return fig

def chart_q4(summary):
    """100% stacked bar chart of topsoil texture."""
    tex = summary.set_index("site")[["sand", "silt", "clay"]].dropna()
    # Normalise before stacking so they sum to 100%
    tex = tex.div(tex.sum(axis=1), axis=0).mul(100).sort_values("clay")
    
    fig = px.bar(tex, orientation="h", title="Topsoil Texture (Normalized to 100%)",
                 labels={"value": "Percentage (%)", "site": "Site", "variable": "Texture"},
                 color_discrete_sequence=["#D4B483", "#C1666B", "#4281A4"])
    return fig

def chart_q5(summary):
    """Bubble scatter + trend line: Does soil moderate water stress?"""
    s = summary.dropna(subset=["clay", "soc"])
    fig = go.Figure()
    
    # Base scatter plot
    fig.add_scatter(
        x=s["clay"], y=s["water_balance_mm"], mode="markers+text", text=s["site"],
        textposition="top center",
        marker=dict(size=14 + 26 * s["soc"] / s["soc"].max(), color=COLORS["sky"]) # bubble size = SOC
    )
    
    # Add trend line only if we have enough points
    if len(s) >= 5:
        slope, intercept = np.polyfit(s["clay"], s["water_balance_mm"], 1)
        r = np.corrcoef(s["clay"], s["water_balance_mm"])[0, 1]
        fig.add_scatter(x=s["clay"], y=slope * s["clay"] + intercept, mode="lines",
                        name=f"Linear fit (r={r:.2f}, n={len(s)})", line=dict(dash="dash", color="gray"))
                        
    fig.add_hline(y=0, annotation_text="rainfall = ET0", line_dash="dot", line_color="red")
    fig.update_layout(title="Clay content vs Water Balance (Bubble Size = Organic Carbon)",
                      xaxis_title="Clay (%)", yaxis_title="Water Balance (mm)")
                      
    return fig

def chart_q1(daily, site):
    """Q1: How does temperature change over time? (7-day rolling mean)"""
    df = daily[daily["site"] == site].copy()
    df = df.sort_values("time")
    df["temp_rolling_7d"] = df["temperature_2m_mean"].rolling(window=7).mean()
    
    fig = px.line(df, x="time", y="temp_rolling_7d", 
                  title=f"7-Day Rolling Mean Temperature ({site})",
                  labels={"time": "Date", "temp_rolling_7d": "Temperature (°C)", "type": "Record Type"},
                  color_discrete_map={"Archive": COLORS["sun"], "Forecast": COLORS["sky"]})
    return fig

def chart_q3(daily, site):
    """Q3: How variable is daily rainfall? (Box plot by month)"""
    df = daily[daily["site"] == site]
    fig = px.box(df, x="month", y="precipitation_sum", 
                 title=f"Daily Rainfall Variability by Month ({site})",
                 labels={"month": "Month (1-12)", "precipitation_sum": "Daily Rainfall (mm)"},
                 color_discrete_sequence=[COLORS["sky"]])
    # Convert x-axis to show categorical months instead of continuous numbers
    fig.update_xaxes(tickmode='array', tickvals=list(range(1, 13)), ticktext=MONTHS)
    return fig

def chart_q6(summary):
    """Q6: Where are the sites, and how do they differ? (Map: color=SOC, size=rainfall)"""
    # Uses px.scatter_map per Plotly 5.24+ requirements
    fig = px.scatter_map(summary, lat="lat", lon="lon", hover_name="site",
                         color="soc", size="rain_mm", 
                         color_continuous_scale="Viridis", size_max=20, zoom=5,
                         title="Site Map (Color = Organic Carbon, Size = Total Rainfall)")
    # Fallback/styling for the map
    fig.update_layout(map_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    return fig
    
    # Add trend line only if we have enough points
    if len(s) >= 5:
        slope, intercept = np.polyfit(s["clay"], s["water_balance_mm"], 1)
        r = np.corrcoef(s["clay"], s["water_balance_mm"])[0, 1]
        fig.add_scatter(x=s["clay"], y=slope * s["clay"] + intercept, mode="lines",
                        name=f"Linear fit (r={r:.2f}, n={len(s)})", line=dict(dash="dash", color="gray"))
                        
    fig.add_hline(y=0, annotation_text="rainfall = ET0", line_dash="dot", line_color="red")
    fig.update_layout(title="Clay content vs Water Balance (Bubble Size = Organic Carbon)",
                      xaxis_title="Clay (%)", yaxis_title="Water Balance (mm)")
    return fig

# --- EXECUTION BLOCK ---
if fetch:
    # Validate inputs before calling APIs
    if sites.empty:
        st.error("Please select at least one site.")
    elif start > end:
        st.error("Start date must be on or before the end date.")
    else:
        st.subheader("Fetching Data...")
        
        # 1. Fetch Weather (1 combined request)
        weather_req = build_weather_request(sites, start, end)
        forecast_req = build_forecast_request(sites)

        with st.spinner("Calling Open-Meteo Archive..."):
            weather_res = send(weather_req)
        with st.spinner("Calling Open-Meteo Forecast..."):
            forecast_res = send(forecast_req)

        st.success(f"Weather data ready! (Archive Cached: {weather_res['from_cache']})")

        # 2. Fetch SoilGrids (Looping through sites)
        soil_reqs = [build_soil_request(row["site"], row["lat"], row["lon"]) for _, row in sites.iterrows()]
        soil_res = []

        for i, req in enumerate(soil_reqs):
            with st.spinner(f"Calling SoilGrids for {req['label']}..."):
                try:
                    res = send(req)
                    soil_res.append(res)
                    if not res["from_cache"] and i < len(soil_reqs) - 1:
                        time.sleep(SOILGRIDS_PAUSE_S)
                except requests.RequestException as exc:
                    st.error(f"SoilGrids failed for {req['label']}: {exc}. The API is a beta service and is sometimes paused. Try again later.")
                    st.stop()
                    
        st.success("Soil data ready!")
        st.subheader("Parsing and Integrating...")

        weather_df = parse_weather(weather_res["data"], sites)
        weather_df["type"] = "Archive"

        forecast_df = parse_weather(forecast_res["data"], sites)
        forecast_df["type"] = "Forecast"

        combined_weather = pd.concat([weather_df, forecast_df], ignore_index=True)
        combined_weather = combined_weather.drop_duplicates(subset=["site", "time"]).reset_index(drop=True)

        soil_frames = [parse_soil(res["data"], req["label"]) for res, req in zip(soil_res, soil_reqs)]
        soil_long = pd.concat(soil_frames, ignore_index=True)
        soil_wide = soil_table(soil_long, sites)

        missing = soil_wide.loc[soil_wide["clay"].isna(), "site"].tolist()
        if missing:
            st.warning(f"SoilGrids returned null for: {', '.join(missing)}. The map has no value at that exact point (water, a built-up area, or a gap in the map). The request still succeeded; try moving the point slightly.")

        daily, summary = integrate(combined_weather, soil_wide)

        st.success("Integration complete!")

        st.write("### Integrated Summary Table (One row per site)")
        st.dataframe(summary, use_container_width=True)

        # --- RENDER CHARTS ---
        st.divider()
        st.header("Analysis & Charts")
        
        first_site = sites.iloc[0]["site"]
        
        st.subheader("Q1: How does temperature change over time?")
        st.plotly_chart(chart_q1(daily, first_site), use_container_width=True)
        
        st.subheader(f"Q2: Does rainfall meet evaporative demand? ({first_site})")
        st.plotly_chart(chart_q2(daily, first_site), use_container_width=True)
        
        st.subheader("Q3: How variable is daily rainfall?")
        st.plotly_chart(chart_q3(daily, first_site), use_container_width=True)
        
        st.subheader("Q4: What is the topsoil texture at each site?")
        st.plotly_chart(chart_q4(summary), use_container_width=True)
        
        st.subheader("Q5: Does soil moderate water stress?")
        st.plotly_chart(chart_q5(summary), use_container_width=True)
        
        st.subheader("Q6: Where are the sites, and how do they differ?")
        st.plotly_chart(chart_q6(summary), use_container_width=True)

        # --- EXPORT ---
        st.divider()
        st.header("Export Data")
        
        b1, b2, _ = st.columns([1, 1, 2])
        b1.download_button("Download site summary (CSV)", summary.to_csv(index=False),
                           "site_summary.csv", "text/csv", use_container_width=True)
        b2.download_button("Download daily records (CSV)", daily.drop(columns="month").to_csv(index=False),
                           "integrated_daily.csv", "text/csv", use_container_width=True)
                           
        SOURCE_NOTE = (f"Sources: SoilGrids 2.0 (ISRIC, CC BY 4.0), {SOIL_DEPTH} mean; "
                       "Open-Meteo Historical Weather API (CC BY 4.0).")
        st.caption(SOURCE_NOTE)