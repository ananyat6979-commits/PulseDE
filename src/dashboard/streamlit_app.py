"""Streamlit dashboard — real-time financial sentiment intelligence.

Features:
  - Live feed via WebSocket → `st.experimental_rerun` pattern.
  - Sentiment timeline (hourly rollup, Plotly).
  - Ticker heatmap (sentiment × confidence).
  - Uncertainty distribution violin plot.
  - Entity/ticker frequency bar chart.
  - Model confidence calibration reliability diagram.
  - Raw article table with colour-coded sentiment badges.
  - Sidebar filters: source, date range, market impact, min confidence.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import websocket  # websocket-client

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8080/v1"
WS_URL = "ws://localhost:8080/ws/realtime"
API_TOKEN = "dev-token"       # Replace with proper auth in production
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

SENTIMENT_COLOURS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#6b7280",
}

st.set_page_config(
    page_title="PulseDE — Financial Sentiment Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "live_queue" not in st.session_state:
    st.session_state.live_queue: deque = deque(maxlen=200)
if "ws_started" not in st.session_state:
    st.session_state.ws_started = False


# ── WebSocket background thread ────────────────────────────────────────────────
def _start_ws_listener() -> None:
    def on_message(ws: object, message: str) -> None:
        try:
            data = json.loads(message)
            st.session_state.live_queue.appendleft(data)
        except Exception:
            pass

    def _run() -> None:
        while True:
            try:
                ws = websocket.WebSocketApp(WS_URL, on_message=on_message)
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception:
                time.sleep(5)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


if not st.session_state.ws_started:
    _start_ws_listener()
    st.session_state.ws_started = True


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_recent(n: int = 300) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/sentiment/latest", params={"n": n}, headers=HEADERS, timeout=5)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_hourly(hours: int = 48) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/sentiment/hourly", params={"hours": hours}, headers=HEADERS, timeout=5)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 PulseDE")
    st.caption("Real-Time Financial Sentiment Intelligence v2.0")
    st.divider()

    time_window = st.selectbox("Time window", ["1h", "6h", "24h", "48h", "7d"], index=2)
    hours_map = {"1h": 1, "6h": 6, "24h": 24, "48h": 48, "7d": 168}
    hours = hours_map[time_window]

    min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)
    sentiment_filter = st.multiselect(
        "Sentiment", ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"]
    )
    impact_filter = st.multiselect(
        "Market impact", ["high", "medium", "low", "unknown"],
        default=["high", "medium", "low", "unknown"]
    )
    show_uncertain = st.toggle("Show uncertain predictions", value=True)
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)

    st.divider()
    ticker_query = st.text_input("Ticker lookup", placeholder="AAPL")
    if ticker_query:
        try:
            r = requests.get(f"{API_BASE}/sentiment/ticker/{ticker_query.upper()}", headers=HEADERS, timeout=5)
            if r.ok:
                t = r.json()
                st.metric("Positive %", f"{t['positive_pct']*100:.1f}%")
                st.metric("Negative %", f"{t['negative_pct']*100:.1f}%")
                st.metric("Articles", t["article_count"])
        except Exception:
            st.warning("Ticker not found or API unavailable")

    st.divider()
    st.caption("Model stack")
    st.code("ProsusAI/finbert\nyiyanghkust/finbert-tone\ndistilroberta-finetuned-financial", language=None)

# ── Main content ───────────────────────────────────────────────────────────────
st.title("📡 PulseDE — Financial Sentiment Intelligence")

# Health badge
try:
    health = requests.get(f"{API_BASE}/health", timeout=2).json()
    badge = "🟢 API Online" if health["status"] == "ok" else "🟡 API Degraded"
except Exception:
    badge = "🔴 API Offline"
st.caption(badge)

df_raw = load_recent(n=500)
df_hourly = load_hourly(hours=hours)

if df_raw.empty:
    st.warning("No data available. Ensure the pipeline is running.")
    st.stop()

# Apply sidebar filters
df = df_raw.copy()
df["published_at"] = pd.to_datetime(df["published_at"])
df["ensemble_confidence"] = df["ensemble_confidence"].astype(float)
cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
df = df[df["published_at"] >= cutoff]
df = df[df["ensemble_confidence"] >= min_confidence]
df = df[df["ensemble_sentiment"].isin(sentiment_filter)]
df = df[df["market_impact"].isin(impact_filter)]
if not show_uncertain:
    df = df[~df["is_uncertain"]]

# ── KPI cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
n = len(df)
pos_pct = (df["ensemble_sentiment"] == "positive").mean() * 100 if n > 0 else 0
neg_pct = (df["ensemble_sentiment"] == "negative").mean() * 100 if n > 0 else 0
avg_conf = df["ensemble_confidence"].mean() * 100 if n > 0 else 0
avg_unc = df["ensemble_uncertainty"].mean() * 100 if n > 0 else 0

col1.metric("Articles", f"{n:,}")
col2.metric("Positive", f"{pos_pct:.1f}%", delta=f"{pos_pct - 50:.1f}pp vs neutral")
col3.metric("Negative", f"{neg_pct:.1f}%")
col4.metric("Avg confidence", f"{avg_conf:.1f}%")
col5.metric("Avg uncertainty", f"{avg_unc:.1f}%")

st.divider()

# ── Charts row 1 ───────────────────────────────────────────────────────────────
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Sentiment over time")
    if not df_hourly.empty:
        dh = pd.DataFrame(df_hourly)
        dh["bucket"] = pd.to_datetime(dh["bucket"])
        dh_pivot = dh.groupby(["bucket", "ensemble_sentiment"])["article_count"].sum().reset_index()
        fig_time = px.bar(
            dh_pivot,
            x="bucket",
            y="article_count",
            color="ensemble_sentiment",
            color_discrete_map=SENTIMENT_COLOURS,
            barmode="stack",
            labels={"article_count": "Articles", "bucket": "Hour", "ensemble_sentiment": "Sentiment"},
            template="plotly_white",
        )
        fig_time.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=True)
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Hourly rollup not yet available — data accumulating.")

with c2:
    st.subheader("Sentiment mix")
    if n > 0:
        counts = df["ensemble_sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(
            counts,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLOURS,
            hole=0.55,
            template="plotly_white",
        )
        fig_pie.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

# ── Charts row 2 ───────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader("Confidence distribution by sentiment")
    fig_violin = px.violin(
        df,
        y="ensemble_confidence",
        x="ensemble_sentiment",
        color="ensemble_sentiment",
        color_discrete_map=SENTIMENT_COLOURS,
        box=True,
        points="outliers",
        template="plotly_white",
        labels={"ensemble_confidence": "Confidence", "ensemble_sentiment": "Sentiment"},
    )
    fig_violin.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

with c4:
    st.subheader("Uncertainty vs confidence")
    fig_scatter = px.scatter(
        df.sample(min(len(df), 300)),
        x="ensemble_confidence",
        y="ensemble_uncertainty",
        color="ensemble_sentiment",
        color_discrete_map=SENTIMENT_COLOURS,
        size_max=8,
        opacity=0.7,
        hover_data=["headline", "source"],
        template="plotly_white",
        labels={
            "ensemble_confidence": "Confidence",
            "ensemble_uncertainty": "Uncertainty (MC Dropout entropy)",
        },
    )
    fig_scatter.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Ticker heatmap ─────────────────────────────────────────────────────────────
st.subheader("Ticker sentiment heatmap")
if "tickers" in df.columns:
    ticker_rows = []
    for _, row in df.iterrows():
        tickers = row.get("tickers") or []
        if isinstance(tickers, str):
            import ast
            tickers = ast.literal_eval(tickers)
        for t in tickers:
            ticker_rows.append({
                "ticker": t,
                "sentiment": row["ensemble_sentiment"],
                "confidence": row["ensemble_confidence"],
            })
    if ticker_rows:
        df_tickers = pd.DataFrame(ticker_rows)
        pivot = df_tickers.groupby(["ticker", "sentiment"]).size().unstack(fill_value=0)
        top_tickers = pivot.sum(axis=1).nlargest(20).index
        pivot = pivot.loc[top_tickers]
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            labels=dict(x="Sentiment", y="Ticker", color="Count"),
            template="plotly_white",
            aspect="auto",
        )
        fig_heat.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No ticker data in this window")

# ── Live feed table ────────────────────────────────────────────────────────────
st.subheader("Live article feed")

live_data = list(st.session_state.live_queue)
display_df = pd.DataFrame(live_data if live_data else df.head(50).to_dict("records"))

if not display_df.empty and "ensemble_sentiment" in display_df.columns:
    cols_to_show = [
        c for c in [
            "published_at", "headline", "source", "ensemble_sentiment",
            "ensemble_confidence", "ensemble_uncertainty", "market_impact",
            "tickers", "is_uncertain",
        ]
        if c in display_df.columns
    ]

    def _colour_row(row: pd.Series) -> list[str]:
        base = SENTIMENT_COLOURS.get(str(row.get("ensemble_sentiment", "")), "")
        bg = f"background-color: {base}22" if base else ""
        return [bg] * len(row)

    styled = (
        display_df[cols_to_show]
        .head(100)
        .style.apply(_colour_row, axis=1)
        .format(
            {
                "ensemble_confidence": "{:.1%}",
                "ensemble_uncertainty": "{:.3f}",
            },
            na_rep="-",
        )
    )
    st.dataframe(styled, use_container_width=True, height=400)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
