# -*- coding: utf-8 -*-
"""
Dashboard Streamlit: Análisis de tipo de cambio paralelo (BOB/USD)
- Descarga datos de dolarboliviahoy.com
- Calcula tendencia (regresión), volatilidad (retorno log)
- Calcula EMAs (20/50/200) y RSI(14)
- Grafica: Precio+Tendencia+EMAs, Volatilidad y RSI

Nota: Mantiene la lógica de tu script original sin alterar los cálculos.
"""

import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress
import streamlit as st

# -------------------------
# 0) Configuración de página
# -------------------------
st.set_page_config(
    page_title="TC Paralelo Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# 1) Funciones de indicadores (idénticas en esencia a tu script)
# -------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI estilo Wilder usando medias móviles exponenciales (EWMA). 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_emas(df: pd.DataFrame, price_col: str = "avg_price", spans=(20, 50, 200)) -> pd.DataFrame:
    for s in spans:
        df[f"EMA{s}"] = df[price_col].ewm(span=s, adjust=False).mean()
    return df


# -------------------------
# 2) Descargar datos (cacheado)
# -------------------------
URL = "https://dolarboliviahoy.com/api/getHistoricalData"

@st.cache_data(ttl=60*60)
def fetch_data() -> pd.DataFrame:
    resp = requests.get(URL, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Error {resp.status_code} al acceder a la API")
    data = resp.json()
    df = pd.DataFrame(data)
    # Tipos
    for col in ["buy_average_price", "sell_average_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "buy_average_price", "sell_average_price"])\
           .sort_values("date")
    # Precio promedio diario
    df["avg_price"] = df[["buy_average_price", "sell_average_price"]].mean(axis=1)
    return df


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Tendencia lineal (regresión sobre timestamps en segundos)
    timestamps = (df["date"].astype("int64") // 10**9).astype(float)
    slope, intercept, r_value, p_value, std_err = linregress(timestamps, df["avg_price"])
    df["trend"] = intercept + slope * timestamps

    # Retorno log y volatilidad global (std * 100)
    df["log_return"] = np.log(df["avg_price"] / df["avg_price"].shift(1))
    volatility = df["log_return"].std(skipna=True) * 100

    # EMAs y RSI
    df = add_emas(df, "avg_price", spans=(20, 50, 200))
    df["RSI14"] = compute_rsi(df["avg_price"], period=14)
    df["RSI_state"] = np.select(
        [df["RSI14"] >= 70, df["RSI14"] <= 30],
        ["Sobrecompra", "Sobreventa"],
        default="Neutral",
    )
    return df, float(volatility)


# -------------------------
# 3) Carga y preparación
# -------------------------
try:
    base_df = fetch_data()
except Exception as e:
    st.error(f"No se pudo obtener datos: {e}")
    st.stop()

df, volatility = enrich_indicators(base_df.copy())

# -------------------------
# 4) Sidebar (filtros y opciones)
# -------------------------
# Botón de actualización manual
st.sidebar.header("Menú")
if st.sidebar.button("Actualizar Datos"):
    fetch_data.clear()   # limpia la cache de esa función
    st.rerun()           # vuelve a ejecutar la app
    
st.sidebar.header("Filtros")
min_date, max_date = df["date"].min().date(), df["date"].max().date()

# Rango de fechas para visualizar (no altera los cálculos originales; solo filtra lo mostrado)
selected_range = st.sidebar.date_input(
    "Rango de fechas",
    (max_date.replace(year=max_date.year-1) if (max_date.year - min_date.year) >= 1 else min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = map(pd.to_datetime, selected_range)
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

mask = (df["date"] >= start_date) & (df["date"] <= end_date)
df_view = df.loc[mask].copy()

st.sidebar.markdown("---")
show_table = st.sidebar.checkbox("Mostrar tabla de datos", value=False)

# Descargar CSV con indicadores
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.sidebar.download_button(
    label="Descargar CSV (completo)",
    data=csv_buf.getvalue(),
    file_name="historial_usdt_bob_con_indicadores.csv",
    mime="text/csv",
)

# -------------------------
# 5) Encabezado y KPIs
# -------------------------
st.title("Tipo de cambio paralelo BOB/USD — Dashboard")
st.caption("Fuente: dolarboliviahoy.com | Indicadores: Tendencia, EMAs (20/50/200), RSI(14) y volatilidad")

if not df_view.empty:
    last = df.dropna(subset=["avg_price"]).iloc[-1]
    prev = df.dropna(subset=["avg_price"]).iloc[-2] if len(df) > 1 else last
    last_price = float(last["avg_price"]) if pd.notna(last["avg_price"]) else np.nan
    d1_change = (last_price / float(prev["avg_price"]) - 1) * 100 if pd.notna(prev["avg_price"]) else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio Promedio (BOB/USD)", f"{last_price:.4f}")
    col2.metric("Variación diaria", f"{d1_change:+.2f}%")
    col3.metric("Volatilidad (std log-return)", f"{volatility:.2f}%")
    col4.metric("RSI(14)", f"{float(last['RSI14']):.1f}" if pd.notna(last["RSI14"]) else "-")

# -------------------------
# 6) Gráfico 1: Precio + Tendencia + EMAs
# -------------------------
with st.container():
    st.subheader("Precio con Tendencia y EMAs")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_view["date"], df_view["avg_price"], label="Precio Promedio (BOB/USD)")
    ax1.plot(df_view["date"], df_view["trend"], label="Tendencia Lineal", linestyle="--")
    for s in (20, 50, 200):
        ax1.plot(df_view["date"], df_view[f"EMA{s}"], label=f"EMA {s}")
    ax1.set_title(f"Tipo de cambio paralelo + EMAs | Volatilidad global: {volatility:.2f}%")
    ax1.set_xlabel("Fecha"); ax1.set_ylabel("Tipo de cambio")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="best")
    fig1.tight_layout()
    st.pyplot(fig1)

# -------------------------
# 7) Gráfico 2: Volatilidad diaria (retorno log)
# -------------------------
    with st.container():
        st.subheader("Volatilidad diaria (retorno logarítmico)")
        fig2, ax2 = plt.subplots(figsize=(12, 3.8))
        sns.lineplot(x="date", y="log_return", data=df_view, ax=ax2)
        ax2.set_xlabel("Fecha"); ax2.set_ylabel("Log Return")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

# -------------------------
# 8) Gráfico 3: RSI (14)
# -------------------------
    with st.container():
        st.subheader("RSI (14) con zonas 70/30")
        fig3, ax3 = plt.subplots(figsize=(12, 3.8))
        ax3.plot(df_view["date"], df_view["RSI14"], label="RSI (14)")
        ax3.axhline(70, linestyle="--", linewidth=1)
        ax3.axhline(30, linestyle="--", linewidth=1)
        ax3.set_xlabel("Fecha"); ax3.set_ylabel("RSI")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)

# -------------------------
# 9) Tabla
# -------------------------
if show_table:
    st.subheader("Datos con indicadores")
    st.dataframe(
        df_view[[
            "date","buy_average_price","sell_average_price","avg_price",
            "trend","EMA20","EMA50","EMA200","RSI14","RSI_state","log_return"
        ]].rename(columns={
            "date": "Fecha",
            "buy_average_price": "CompraProm",
            "sell_average_price": "VentaProm",
            "avg_price": "PrecioProm",
            "trend": "Tendencia",
        }),
        use_container_width=True,
        hide_index=True,
    )

# -------------------------
# 10) Pie de página
# -------------------------
st.caption("© Hecho con Streamlit. Este panel solo visualiza y no constituye recomendación financiera.")
