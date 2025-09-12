# -*- coding: utf-8 -*-
# streamlit_tc.py
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------- Configuración de la página ----------
st.set_page_config(page_title="TC Paralelo Dashboard", layout="wide")

# ---------- Constantes ----------
API_URL_DEFAULT = "https://dolarboliviahoy.com/api/getHistoricalData"

# ---------- Utilidades ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    roll_up = gain.rolling(window).mean()
    roll_down = loss.rolling(window).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

@st.cache_data(show_spinner=False)
def fetch_data(api_url: str) -> pd.DataFrame:
    """
    Descarga datos desde la API y calcula columnas requeridas:
    trend, EMA20/50/200, log_return, RSI14.
    """
    r = requests.get(api_url, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data)
    # Normalización de nombres
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c: "date"})
                break
    if "avg_price" not in df.columns:
        for c in df.columns:
            cl = c.lower()
            if "avg" in cl or "price" in cl:
                df = df.rename(columns={c: "avg_price"})
                break

    if "date" not in df.columns or "avg_price" not in df.columns:
        raise ValueError("No se encontraron columnas 'date' y 'avg_price' en la respuesta de la API.")

    # Tipos y orden
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df = df.dropna(subset=["date", "avg_price"]).sort_values("date").reset_index(drop=True)

    # Tendencia lineal simple
    n = len(df)
    if n >= 2:
        x = np.arange(n, dtype=float)
        y = df["avg_price"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        df["trend"] = slope * x + intercept
    else:
        df["trend"] = df["avg_price"]

    # EMAs
    df["EMA20"] = ema(df["avg_price"], 20)
    df["EMA50"] = ema(df["avg_price"], 50)
    df["EMA200"] = ema(df["avg_price"], 200)

    # Retorno log
    df["log_return"] = np.log(df["avg_price"]).diff()

    # RSI
    df["RSI14"] = compute_rsi(df["avg_price"], 14)

    return df

# ---------- Sidebar (controles) ----------
st.sidebar.header("Controles")
api_url = st.sidebar.text_input("API URL", value=os.getenv("API_URL", API_URL_DEFAULT))
days_back = st.sidebar.slider("Ventana (días) a mostrar", min_value=30, max_value=720, value=180, step=30)

col_em1, col_em2, col_em3 = st.sidebar.columns(3)
show_ema20 = col_em1.checkbox("EMA 20", value=True)
show_ema50 = col_em2.checkbox("EMA 50", value=False)
show_ema200 = col_em3.checkbox("EMA 200", value=False)

show_vol = st.sidebar.checkbox("Volatilidad (log-return)", value=True)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True)

refresh_sec = st.sidebar.number_input(
    "Auto-refresh (segundos)",
    min_value=0, max_value=600, value=60, step=5,
    help="Coloca 0 para desactivar el auto-refresh."
)
if refresh_sec > 0:
    st_autorefresh(interval=refresh_sec * 1000, key="datarefresh")

# ---------- Carga de datos ----------
info_box = st.empty()
info_box.info("Cargando datos...")
try:
    df = fetch_data(api_url)
    info_box.empty()
except Exception as e:
    info_box.error(f"Error al obtener datos: {e}")
    st.stop()

if df.empty:
    st.warning("La API devolvió un dataset vacío.")
    st.stop()

# ---------- Filtro por rango ----------
max_date = df["date"].max()
min_date = max_date - timedelta(days=int(days_back))
df_view = df[df["date"].between(min_date, max_date)].copy()

if df_view.empty:
    st.warning("No hay datos en el rango seleccionado.")
    st.stop()

# ---------- Métricas principales ----------
col1, col2, col3, col4 = st.columns(4)

last_price = df_view["avg_price"].iloc[-1]
prev_price = df_view["avg_price"].iloc[-2] if len(df_view) > 1 else last_price
chg = last_price - prev_price
chg_pct = (chg / prev_price * 100.0) if prev_price else 0.0

# Volatilidad anualizada (log)
vol_ann = df_view["log_return"].std() * math.sqrt(252) * 100.0 if df_view["log_return"].notna().any() else 0.0

wk_window = df_view[df_view["date"] >= (max_date - timedelta(days=7))]
wk_chg_pct = (wk_window["avg_price"].iloc[-1] / wk_window["avg_price"].iloc[0] - 1) * 100.0 if len(wk_window) > 1 else 0.0

col1.metric("Último precio (BOB/USD)", f"{last_price:,.3f}", f"{chg:+.3f}")
col2.metric("Variación % diaria", f"{chg_pct:+.2f}%")
col3.metric("Volatilidad anualizada (log)", f"{vol_ann:.2f}%")
col4.metric("Cambio 7 días", f"{wk_chg_pct:+.2f}%")

st.markdown("---")

# ---------- Gráfico 1: Precio + Tendencia + EMAs ----------
st.subheader("Tipo de cambio paralelo (BOB/USD) + Tendencia y EMAs")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df_view["date"], df_view["avg_price"], label="Precio Promedio (BOB/USD)")
ax1.plot(df_view["date"], df_view["trend"], label="Tendencia Lineal", linestyle="--", linewidth=1)

if show_ema20:
    ax1.plot(df_view["date"], df_view["EMA20"], label="EMA 20")
if show_ema50:
    ax1.plot(df_view["date"], df_view["EMA50"], label="EMA 50")
if show_ema200:
    ax1.plot(df_view["date"], df_view["EMA200"], label="EMA 200")

ax1.set_xlabel("Fecha")
ax1.set_ylabel("Tipo de cambio")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best")
st.pyplot(fig1, clear_figure=True)

# ---------- Gráfico 2: Volatilidad (log-return) ----------
if show_vol:
    st.subheader("Volatilidad diaria (retorno logarítmico)")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(df_view["date"], df_view["log_return"])
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Log Return")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

# ---------- Gráfico 3: RSI ----------
if show_rsi:
    st.subheader("RSI (14) con zonas 70/30")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(df_view["date"], df_view["RSI14"], label="RSI (14)")
    ax3.axhline(70, linestyle="--", linewidth=1)
    ax3.axhline(30, linestyle="--", linewidth=1)
    ax3.set_xlabel("Fecha")
    ax3.set_ylabel("RSI")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")
    st.pyplot(fig3, clear_figure=True)

st.caption("Fuente: API histórica. Las métricas se calculan sobre la ventana seleccionada.")
