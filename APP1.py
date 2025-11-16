# app.py
# Real-time BTC Predictor — Ensemble + resilient fetch + TA + safe training
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time
import math
import warnings
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# optional libraries
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

try:
    from catboost import CatBoostRegressor
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Real-time Predictor — Ensemble + TA", layout="wide")

# ---------------------------
# plotting imports (robust)
# ---------------------------
HAVE_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    HAVE_MPL = False

HAVE_PLOTLY = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    HAVE_PLOTLY = False

# ---------------------------
# small helper plotting functions
# ---------------------------
def plot_series(plot_placeholder, x, y, preds_dict=None, title=""):
    """
    x: list of datetimes or indexes
    y: list of floats (actual close)
    preds_dict: {"XGB": (pred_value, pred_time), "RF": (...), ...}
    """
    try:
        if HAVE_MPL:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(x, y, label="Close", linewidth=1.5)
            ax.scatter([x[-1]], [y[-1]], color="orange", label="Last Close")
            if preds_dict:
                markers = ["*", "X", "D", "P", "v"]
                for i,(k,(pv,pt)) in enumerate(preds_dict.items()):
                    ax.scatter([pt], [pv], label=f"{k} pred: {pv:,.2f}", marker=markers[i%len(markers)], s=100)
            ax.set_title(title)
            ax.legend(loc="upper left", fontsize="small")
            plot_placeholder.pyplot(fig)
        elif HAVE_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Close"))
            fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", name="Last Close", marker=dict(color="orange",size=8)))
            if preds_dict:
                for k,(pv,pt) in preds_dict.items():
                    fig.add_trace(go.Scatter(x=[pt], y=[pv], mode="markers", name=f"{k} pred: {pv:,.2f}", marker_symbol="star", marker_size=12))
            fig.update_layout(title=title, xaxis_title="time", yaxis_title="price", height=420)
            plot_placeholder.plotly_chart(fig)
        else:
            # fallback
            s = pd.Series(y, index=x)
            plot_placeholder.line_chart(s)
            if preds_dict:
                lines = [f"{k}: {v[0]:.2f} at {v[1]}" for k,v in preds_dict.items()]
                plot_placeholder.write("Predictions: " + ", ".join(lines))
    except Exception as e:
        plot_placeholder.text(f"Plot error: {e}")

def plot_feature_importances(plot_placeholder, feat_names, importances):
    try:
        fi = pd.DataFrame({"feature": feat_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False).head(25)
        if HAVE_MPL:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.barh(fi["feature"][::-1], fi["importance"][::-1])
            ax.set_title("Top feature importances")
            plot_placeholder.pyplot(fig)
        elif HAVE_PLOTLY:
            fig = px.bar(fi.iloc[:25][::-1], x="importance", y="feature", orientation="h", title="Top feature importances")
            plot_placeholder.plotly_chart(fig)
        else:
            plot_placeholder.dataframe(fi)
    except Exception as e:
        plot_placeholder.text(f"Feature importance error: {e}")

# ---------------------------
# endpoints + resilient fetcher (Binance + CoinGecko fallback)
# ---------------------------
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H = "https://api.binance.com/api/v3/ticker/24hr"
COINGECKO_RANGE = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"

from requests.adapters import HTTPAdapter, Retry

def _make_requests_session(retries=3, backoff_factor=0.3, status_forcelist=(429,500,502,503,504)):
    """
    Create a requests.Session with Retry configured.
    NOTE: intentionally exclude 451 from retry list so we don't keep retrying
    'Unavailable For Legal Reasons' responses — those should immediately fall back.
    """
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # set a sensible user agent so some providers don't block us
    s.headers.update({"User-Agent": "real-time-btc-predictor/1.0 (+https://example.com)"})
    return s

def now_ms():
    return int(time.time() * 1000)
def ms_to_s(ms):
    return int(ms / 1000)

# Rate-limit aware CoinGecko fetcher (retries, respects Retry-After)
def fetch_klines_coingecko(symbol="BTCUSDT", interval="1m", start_time_ms=None, end_time_ms=None, max_attempts=5):
    """
    Rate-limit-aware CoinGecko range fetcher (uses a requests.Session).
    - Exits immediately on HTTP 451 (legal block) and returns empty DF so the app can use other fallbacks.
    - Retries on 429 respecting Retry-After header.
    - Caps backoff wait to avoid extremely long sleeps.
    """
    session = _make_requests_session(retries=3, backoff_factor=0.5)
    now = now_ms()
    if end_time_ms is None:
        end_time_ms = now
    end_s = ms_to_s(min(end_time_ms, now))
    if start_time_ms is None:
        start_s = end_s - 7*24*3600  # 7 days default
    else:
        start_s = ms_to_s(start_time_ms)

    url = COINGECKO_RANGE
    params = {"vs_currency": "usd", "from": int(start_s), "to": int(end_s)}
    attempt = 0
    backoff = 1.0
    MAX_BACKOFF = 8.0  # cap backoff so sleeps don't grow unbounded

    while attempt < max_attempts:
        attempt += 1
        try:
            r = session.get(url, params=params, timeout=20)
            # If CoinGecko returns 451 -> legal block, don't retry, fall back immediately
            if r.status_code == 451:
                st.warning(f"CoinGecko returned HTTP 451 (unavailable for legal reasons). Not retrying; falling back (attempt {attempt}/{max_attempts}).")
                df_empty = pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
                df_empty._source = "coingecko"
                return df_empty

            if r.status_code == 429:
                # Rate limited: honor Retry-After if present, else small backoff
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        wait = min(float(ra), MAX_BACKOFF)
                    except Exception:
                        wait = min(backoff, MAX_BACKOFF)
                else:
                    wait = min(backoff, MAX_BACKOFF)
                st.warning(f"CoinGecko 429 rate limit. Sleeping {wait:.1f}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait)
                backoff = min(backoff * 2.0, MAX_BACKOFF)
                continue

            # For other non-200 statuses, treat as hard failure and return empty to trigger Binance fallback upstream
            if r.status_code != 200:
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text[:400]
                st.warning(f"CoinGecko request failed (status {r.status_code}). Server message: {msg}")
                df_empty = pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
                df_empty._source = "coingecko"
                return df_empty

            j = r.json()
            prices = j.get("prices", [])
            volumes = j.get("total_volumes", [])
            if not prices:
                df_empty = pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
                df_empty._source = "coingecko"
                return df_empty

            dfp = pd.DataFrame(prices, columns=["ts_ms","price"])
            dfv = pd.DataFrame(volumes, columns=["ts_ms","volume"]) if volumes else None
            dfp["open_time"] = pd.to_datetime(dfp["ts_ms"], unit="ms")
            dfp["open"] = dfp["price"]; dfp["high"] = dfp["price"]; dfp["low"] = dfp["price"]; dfp["close"] = dfp["price"]
            if dfv is not None and not dfv.empty:
                df = pd.merge_asof(dfp.sort_values("ts_ms"), dfv.sort_values("ts_ms"), on="ts_ms", direction="nearest")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            else:
                df = dfp
                df["volume"] = 0.0
            df = df[["open_time","open","high","low","close","volume"]].reset_index(drop=True)
            df._source = "coingecko"
            return df

        except requests.RequestException as e:
            # Network error — retry a few times, then return empty DF
            st.warning(f"CoinGecko network error on attempt {attempt}: {e}. Sleeping {min(backoff, MAX_BACKOFF):.1f}s before retry.")
            time.sleep(min(backoff, MAX_BACKOFF))
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue
        except Exception as e:
            st.error(f"Unexpected CoinGecko parsing error: {e}")
            df_err = pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
            df_err._source = "coingecko"
            return df_err

    # exhausted attempts
    st.warning("CoinGecko fallback exhausted retries. Returning empty DataFrame so other fallbacks can be used.")
    df_empty = pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
    df_empty._source = "coingecko"
    return df_empty

# Robust Binance klines fetcher with fallback
def fetch_klines_resilient(symbol="BTCUSDT", interval="1m", limit=500, start_time_ms=None, end_time_ms=None, days=None):
    """
    Binance-first fetcher with effectively unlimited retries on transient errors.
    - Retries Binance indefinitely (capped backoff).
    - Falls back to CoinGecko only on 'hard' failures (451 or non-retriable 4xx).
    - Keeps behavior for paginated historical requests, but each request to Binance will loop until success or hard failure.
    """
    binance_session = _make_requests_session(retries=3, backoff_factor=0.4)  # session config; retry won't be infinite here — we use explicit loops
    now = now_ms()
    if end_time_ms is None:
        end_time_ms = now
    else:
        end_time_ms = min(end_time_ms, now)

    if start_time_ms is None:
        if days:
            start_time_ms = end_time_ms - int(days * 24 * 3600 * 1000)
        else:
            start_time_ms = None

    # Helper to perform a GET to Binance with unlimited-ish retry loop (bounded backoff)
    def _binance_get_with_retries(url, params, timeout=15):
        attempt = 0
        backoff = 0.5
        MAX_BACKOFF = 8.0
        while True:
            attempt += 1
            try:
                r = binance_session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                # network/connection error -> retry
                st.warning(f"Binance network error (attempt {attempt}): {e}. Sleeping {min(backoff,MAX_BACKOFF):.1f}s then retrying.")
                time.sleep(min(backoff, MAX_BACKOFF))
                backoff = min(backoff * 2.0, MAX_BACKOFF)
                continue

            # If Binance explicitly returns 451 (legal block) -> treat as hard failure (fall back to CoinGecko)
            if r.status_code == 451:
                st.warning(f"Binance returned HTTP 451 (unavailable for legal reasons). Falling back to CoinGecko.")
                return r  # caller will handle fallback

            # Rate-limited -> honor Retry-After if present, else backoff and retry
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else min(backoff, MAX_BACKOFF)
                except Exception:
                    wait = min(backoff, MAX_BACKOFF)
                st.warning(f"Binance 429 rate limit. Sleeping {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                backoff = min(backoff * 2.0, MAX_BACKOFF)
                continue

            # Other 4xx (client) errors except 408 we treat as hard failures and fall back
            if 400 <= r.status_code < 500 and r.status_code not in (429, 408):
                st.warning(f"Binance returned client error {r.status_code}. Falling back to CoinGecko. Server msg: {r.text[:300]}")
                return r

            # 5xx or 3xx etc -> transient: wait and retry
            if r.status_code >= 500 or r.status_code == 0:
                st.warning(f"Binance server error {r.status_code}. Sleeping {min(backoff,MAX_BACKOFF):.1f}s then retrying (attempt {attempt}).")
                time.sleep(min(backoff, MAX_BACKOFF))
                backoff = min(backoff * 2.0, MAX_BACKOFF)
                continue

            # Success or other statuses
            return r

    # If no time range: request most recent 'limit' candles
    if start_time_ms is None:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = _binance_get_with_retries(BINANCE_KLINES, params, timeout=15)
        # If r is client error or 451 -> fallback to CoinGecko
        if r is None or r.status_code != 200:
            st.warning("Binance recent klines not available or returned non-200. Falling back to CoinGecko.")
            return fetch_klines_coingecko(symbol=symbol, interval=interval, start_time_ms=(now - 7*24*3600*1000), end_time_ms=now)

        try:
            data = r.json()
            df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["open_time","open","high","low","close","volume"]].reset_index(drop=True)
            df._source = "binance"
            return df
        except Exception as e:
            st.warning(f"Failed parsing Binance recent klines JSON: {e}. Falling back to CoinGecko.")
            return fetch_klines_coingecko(symbol=symbol, interval=interval, start_time_ms=(now - 7*24*3600*1000), end_time_ms=now)

    # Paginated historical fetch — each page uses the same retry-get helper
    all_rows = []
    fetch_start = int(start_time_ms)
    interval_map = {"1m":60_000,"3m":3*60_000,"5m":5*60_000,"15m":15*60_000,"30m":30*60_000,"1h":60*60*1000,"1d":24*60*60*1000}
    interval_ms = interval_map.get(interval, 60_000)
    MAX_LIMIT = 1000

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": MAX_LIMIT, "startTime": int(fetch_start), "endTime": int(end_time_ms)}
        r = _binance_get_with_retries(BINANCE_KLINES, params, timeout=20)

        # If Binance returned a non-200 that we decided is a hard failure, fall back
        if r is None or r.status_code != 200:
            st.warning("Binance historical klines failed or returned non-200. Falling back to CoinGecko.")
            return fetch_klines_coingecko(symbol=symbol, interval=interval, start_time_ms=start_time_ms, end_time_ms=end_time_ms)

        try:
            data = r.json()
        except Exception:
            st.warning("Failed to parse Binance klines JSON. Falling back to CoinGecko.")
            return fetch_klines_coingecko(symbol=symbol, interval=interval, start_time_ms=start_time_ms, end_time_ms=end_time_ms)

        if not data:
            break

        all_rows.extend(data)
        last_open_time = int(data[-1][0])
        next_start = last_open_time + interval_ms
        if next_start >= end_time_ms or next_start <= fetch_start:
            break
        fetch_start = next_start
        time.sleep(0.12)

    if not all_rows:
        df_empty = pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
        df_empty._source = "binance"
        return df_empty

    df = pd.DataFrame(all_rows, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open_time","open","high","low","close","volume"]].reset_index(drop=True)
    df._source = "binance"
    return df

def fetch_current_price(symbol="BTCUSDT"):
    """
    Try Binance indefinitely (capped backoff). Use CoinGecko only as fallback on hard failures.
    """
    session = _make_requests_session(retries=2, backoff_factor=0.2)
    attempt = 0
    backoff = 0.5
    MAX_BACKOFF = 8.0
    url = BINANCE_PRICE
    params = {"symbol": symbol}
    while True:
        attempt += 1
        try:
            r = session.get(url, params=params, timeout=5)
        except requests.RequestException as e:
            st.warning(f"Binance price network error (attempt {attempt}): {e}. Sleeping {min(backoff,MAX_BACKOFF):.1f}s then retrying.")
            time.sleep(min(backoff, MAX_BACKOFF))
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue

        if r.status_code == 451:
            st.warning("Binance price returned HTTP 451. Falling back to CoinGecko.")
            break

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else min(backoff, MAX_BACKOFF)
            except Exception:
                wait = min(backoff, MAX_BACKOFF)
            st.warning(f"Binance price 429. Sleeping {wait:.1f}s.")
            time.sleep(wait)
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue

        if 400 <= r.status_code < 500 and r.status_code not in (429, 408):
            st.warning(f"Binance price returned client error {r.status_code}. Falling back to CoinGecko.")
            break

        if r.status_code >= 500:
            st.warning(f"Binance price returned server error {r.status_code}. Sleeping {min(backoff,MAX_BACKOFF):.1f}s then retrying.")
            time.sleep(min(backoff, MAX_BACKOFF))
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue

        try:
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:
            st.warning("Failed parsing Binance price JSON; falling back to CoinGecko.")
            break

    # CoinGecko fallback
    try:
        r2 = requests.get("https://api.coingecko.com/api/v3/simple/price", params={"ids":"bitcoin","vs_currencies":"usd"}, timeout=5)
        r2.raise_for_status()
        return float(r2.json()["bitcoin"]["usd"])
    except Exception as e:
        st.error(f"Both Binance and CoinGecko failed to provide current price: {e}")
        raise

def fetch_binance_24h(symbol="BTCUSDT"):
    """
    Try Binance indefinitely (capped backoff) to get 24h stats. Use CoinGecko only as a fallback.
    """
    session = _make_requests_session(retries=2, backoff_factor=0.2)
    attempt = 0
    backoff = 0.5
    MAX_BACKOFF = 8.0
    url = BINANCE_24H
    params = {"symbol": symbol}
    while True:
        attempt += 1
        try:
            r = session.get(url, params=params, timeout=5)
        except requests.RequestException as e:
            st.warning(f"Binance 24h network error (attempt {attempt}): {e}. Sleeping {min(backoff,MAX_BACKOFF):.1f}s then retrying.")
            time.sleep(min(backoff, MAX_BACKOFF))
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue

        if r.status_code == 451:
            st.warning("Binance 24h returned HTTP 451. Falling back to CoinGecko.")
            break

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else min(backoff, MAX_BACKOFF)
            except Exception:
                wait = min(backoff, MAX_BACKOFF)
            st.warning(f"Binance 24h 429. Sleeping {wait:.1f}s.")
            time.sleep(wait)
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue

        if 400 <= r.status_code < 500 and r.status_code not in (429, 408):
            st.warning(f"Binance 24h returned client error {r.status_code}. Falling back to CoinGecko.")
            break

        if r.status_code >= 500:
            st.warning(f"Binance 24h returned server error {r.status_code}. Sleeping {min(backoff,MAX_BACKOFF):.1f}s then retrying.")
            time.sleep(min(backoff, MAX_BACKOFF))
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue

        try:
            r.raise_for_status()
            d = r.json()
            return {
                "price_change": float(d.get("priceChange", 0.0)),
                "price_change_percent": float(d.get("priceChangePercent", 0.0)),
                "high_price": float(d.get("highPrice", 0.0)),
                "low_price": float(d.get("lowPrice", 0.0)),
                "volume": float(d.get("volume", 0.0))
            }
        except Exception:
            st.warning("Failed parsing Binance 24h JSON; falling back to CoinGecko.")
            break

    # fallback to CoinGecko klines
    try:
        end = now_ms(); start = end - 2*24*3600*1000
        cg = fetch_klines_coingecko(start_time_ms=start, end_time_ms=end)
        if cg.empty:
            raise RuntimeError("CoinGecko fallback returned empty")
        first = float(cg["close"].iloc[0]); last = float(cg["close"].iloc[-1])
        return {
            "price_change": last - first,
            "price_change_percent": ((last-first)/first)*100.0 if first != 0 else 0.0,
            "high_price": float(cg["high"].max()),
            "low_price": float(cg["low"].min()),
            "volume": float(cg["volume"].sum())
        }
    except Exception as e:
        st.warning(f"Both Binance and CoinGecko failed for 24h stats: {e}")
        return {"price_change":0.0,"price_change_percent":0.0,"high_price":0.0,"low_price":0.0,"volume":0.0}

# ---------------------------
# indicators & feature builders
# ---------------------------
USE_PANDAS_TA = False
try:
    import pandas_ta as pta
    USE_PANDAS_TA = True
except Exception:
    USE_PANDAS_TA = False

def compute_indicators(df, close_col="Close", high_col="High", low_col="Low", vol_col="Volume"):
    d = df.copy().reset_index(drop=True)
    if close_col not in d.columns and "close" in d.columns: d[close_col]=d["close"]
    if high_col not in d.columns and "high" in d.columns: d[high_col]=d["high"]
    if low_col not in d.columns and "low" in d.columns: d[low_col]=d["low"]
    if vol_col not in d.columns and "volume" in d.columns: d[vol_col]=d["volume"]
    d["return_1"] = d[close_col].pct_change().fillna(0.0)
    d["log_return_1"] = np.log(d[close_col] / d[close_col].shift(1)).replace([np.inf, -np.inf], 0).fillna(0.0)
    d["sma_10"] = d[close_col].rolling(10, min_periods=1).mean()
    d["sma_50"] = d[close_col].rolling(50, min_periods=1).mean()
    d["ema_12"] = d[close_col].ewm(span=12, adjust=False).mean()
    d["ema_26"] = d[close_col].ewm(span=26, adjust=False).mean()
    d["vol_10"] = d["return_1"].rolling(10, min_periods=1).std()
    roll20 = d[close_col].rolling(20, min_periods=1)
    d["bb_mid"] = roll20.mean()
    d["bb_std"] = roll20.std().fillna(0)
    d["bb_upper"] = d["bb_mid"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_mid"] - 2 * d["bb_std"]
    def rsi(series, period=14):
        delta = series.diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/period, adjust=False).mean(); ma_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = ma_up / (ma_down + 1e-9); return 100 - (100 / (1 + rs))
    d["rsi_14"] = rsi(d[close_col])
    macd_line = d["ema_12"] - d["ema_26"]; macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    d["macd"] = macd_line; d["macd_signal"] = macd_signal; d["macd_hist"] = d["macd"] - d["macd_signal"]
    if USE_PANDAS_TA:
        try:
            ta_df = pta.utils.dropna(d[[close_col, high_col, low_col, vol_col]].copy(), how='all')
            ta_all = pta.all(ta_df, verbose=False)
            if "RSI_14" in ta_all.columns: d["rsi_14"]=ta_all["RSI_14"]
        except Exception:
            pass
    d = d.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return d

def build_features_from_df(df, n_lags=10, use_extra_cols=True):
    base = df.copy().reset_index(drop=True)
    if "Close" not in base.columns and "close" in base.columns: base["Close"] = base["close"]
    if "Close" not in base.columns: raise ValueError("Dataframe must contain a 'Close' column.")
    enriched = compute_indicators(base, close_col="Close", high_col="High" if "High" in base.columns else "high", low_col="Low" if "Low" in base.columns else "low", vol_col="Volume" if "Volume" in base.columns else "volume")
    price_series = enriched["Close"].astype(float).reset_index(drop=True)
    lag_df = pd.DataFrame({"price": price_series})
    for lag in range(1, n_lags+1): lag_df[f"lag_{lag}"] = price_series.shift(lag)
    lag_df = lag_df.dropna().reset_index(drop=True)
    enriched_aligned = enriched.iloc[n_lags:].reset_index(drop=True)
    feature_cols = [c for c in lag_df.columns if c != "price"]
    extras_keep = []
    candidate_extras = ["rsi_14","macd","macd_signal","macd_hist","bb_upper","bb_mid","bb_lower","sma_10","sma_50","vol_10","return_1","log_return_1","ema_12","ema_26"]
    for ex in candidate_extras:
        if ex in enriched_aligned.columns: extras_keep.append(ex)
    for col in ["Sentiment_Score","GDP_Growth","Inflation_Rate","RSI","MACD","Bollinger_Upper","Bollinger_Lower","Volume"]:
        if col in enriched_aligned.columns and col not in extras_keep: extras_keep.append(col)
    X = lag_df[feature_cols].reset_index(drop=True)
    if use_extra_cols and len(extras_keep)>0:
        X = pd.concat([X.reset_index(drop=True), enriched_aligned[extras_keep].reset_index(drop=True)], axis=1)
    y = lag_df["price"].values
    return X, y, X.columns.tolist(), enriched_aligned

# ---------------------------
# Train helpers: XGB CV + other models
# ---------------------------
def train_xgb_with_cv(X, y, n_splits=4, n_iter_search=20, random_state=42):
    if X.shape[0] < 3:
        raise ValueError("Not enough rows to train XGBoost.")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", eval_metric="mae", random_state=random_state, n_jobs= -1)
    param_dist = {"n_estimators":[50,100,200], "max_depth":[3,5,7], "learning_rate":[0.01,0.03,0.1], "subsample":[0.6,0.8,1.0], "colsample_bytree":[0.6,0.8,1.0]}
    rnd = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=n_iter_search, cv=tscv, scoring="neg_mean_absolute_error", random_state=random_state, n_jobs=-1)
    rnd.fit(X,y)
    best = rnd.best_estimator_
    maes=[]; rmses=[]
    for train_idx, test_idx in tscv.split(X):
        best.fit(X[train_idx], y[train_idx])
        p = best.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], p)); rmses.append(math.sqrt(mean_squared_error(y[test_idx], p)))
    return best, {"cv_mae_mean": float(np.mean(maes)), "cv_rmse_mean": float(np.mean(rmses))}

def train_other_model(model_name, X_train, y_train):
    if model_name == "RF":
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        return m
    if model_name == "LGB" and HAVE_LGB:
        m = lgb.LGBMRegressor(n_estimators=200)
        m.fit(X_train, y_train)
        return m
    if model_name == "CAT" and HAVE_CAT:
        m = CatBoostRegressor(verbose=0, iterations=200)
        m.fit(X_train, y_train)
        return m
    return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Real-time BTC Predictor — Ensemble of Models + Live Chart")

left, right = st.columns([2,1])
with left:
    uploaded = st.file_uploader("Upload CSV (optional). Must contain 'Close' column.", type=["csv","txt"])
    st.markdown("If no CSV, app fetches live BTC (Binance/Coingecko fallback).")
    interval = st.selectbox("Interval for training (live)", ["1m","5m","15m","1h"], index=1)
    lookback = st.number_input("History rows for training", min_value=200, max_value=5000, value=1000, step=50)
    n_lags = st.slider("Lag features (n)", 1, 50, 12)
    use_extras = st.checkbox("Use extra TA/macro columns", value=True)
    n_splits = st.slider("TimeSeriesSplit folds", 2, 8, 4)
    tune_iters = st.number_input("XGB tune iterations", min_value=4, max_value=200, value=20, step=2)
    train_btn = st.button("Train Models")
    st.markdown("----")
    st.markdown("Realtime / Live chart controls")
    live_interval = st.number_input("Live poll interval (s)", min_value=1, max_value=60, value=5)
    live_window = st.number_input("Live window length (points)", min_value=10, max_value=1000, value=200)
    refresh_btn = st.button("Refresh Now")
    start_live = st.button("Start Live")
    stop_live = st.button("Stop Live")

with right:
    st.markdown("Models active:")
    st.write("- XGBoost (tuned)")
    st.write("- RandomForest")
    if HAVE_LGB: st.write("- LightGBM")
    if HAVE_CAT: st.write("- CatBoost")
    st.markdown("---")
    st.markdown("Tip: use 'Refresh Now' to append a new tick to the real-time chart. Start Live runs a blocking loop (press Stop Live to stop).")

status = st.empty()
keybox = st.empty()
chart_ph = st.empty()
metrics_ph = st.empty()
feat_imp_ph = st.empty()
models_table_ph = st.empty()

# session_state initialization
if "models" not in st.session_state:
    st.session_state["models"] = {}   # dict name->model
if "model_metrics" not in st.session_state:
    st.session_state["model_metrics"] = {}
if "feat_names" not in st.session_state:
    st.session_state["feat_names"] = []
if "last_df_for_preds" not in st.session_state:
    st.session_state["last_df_for_preds"] = None
if "realtime" not in st.session_state:
    st.session_state["realtime"] = {"times": [], "prices": []}
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

# load CSV if uploaded
df = None
using_csv = False
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
            except Exception:
                pass
        if "Close" not in df.columns and "close" in df.columns:
            df["Close"] = df["close"]
        if "Close" not in df.columns:
            st.error("CSV must include 'Close' column.")
            df = None
        else:
            using_csv = True
            st.success(f"Loaded CSV with {len(df)} rows.")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None

# TRAIN: train multiple models (defensive)
if train_btn:
    try:
        status.info("Preparing data for training...")
        # load df_use (either CSV or live)
        if using_csv and df is not None:
            df_use = df.copy().reset_index(drop=True).iloc[-int(lookback):].reset_index(drop=True)
        else:
            status.info("Fetching historical series for BTCUSDT...")
            df_live = fetch_klines_resilient(symbol="BTCUSDT", interval=interval, limit=int(lookback))
            # detect source for debug
            if hasattr(df_live, "_source"):
                status.info(f"Data source: {df_live._source}")
            if df_live is None or df_live.empty:
                st.error("No historical data could be fetched from Binance or CoinGecko. Try increasing 'lookback', reduce interval, or try again later.")
                status.info("Aborting training due to lack of data.")
                st.stop()
            if "open_time" in df_live.columns:
                df_live = df_live.rename(columns={"open_time":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            df_use = df_live.reset_index(drop=True)

        # QUICK CHECK: ensure df_use has reasonable number of rows
        if df_use is None or df_use.empty or len(df_use) < (n_lags + 5):
            st.error(f"Not enough source rows ({0 if df_use is None else len(df_use)}) to create lag features (need at least n_lags + 5). Increase 'lookback' or upload more data.")
            st.stop()

        # Build features
        X, y, feat_names, enriched_aligned = build_features_from_df(df_use, n_lags=n_lags, use_extra_cols=use_extras)
        if X.shape[0] == 0:
            st.error("After creating lag features there are zero rows. Reduce n_lags or provide more data.")
            st.stop()

        # if X rows are fewer than n_splits+1, adjust n_splits down with a warning
        if X.shape[0] <= n_splits:
            old_n = n_splits
            n_splits = max(2, X.shape[0] - 1)
            st.warning(f"Reducing TimeSeriesSplit folds from {old_n} to {n_splits} because only {X.shape[0]} training rows are available.")

        # If dataset is small, reduce tune_iters
        if X.shape[0] < 100 and tune_iters > 20:
            old_iter = tune_iters
            tune_iters = max(4, int(tune_iters * (X.shape[0] / 100.0)))
            st.info(f"Reducing XGB tuning iterations from {old_iter} to {tune_iters} due to small dataset ({X.shape[0]} rows).")

        status.info(f"Features built: {X.shape[0]} rows x {X.shape[1]} cols")
        st.session_state["last_df_for_preds"] = df_use
        st.session_state["feat_names"] = feat_names

        # XGB: time-series CV tune
        status.info("Tuning XGBoost (this may take a while)...")
        xgb_model, xgb_metrics = train_xgb_with_cv(X.values, y, n_splits=n_splits, n_iter_search=int(tune_iters))
        st.session_state["models"]["XGB"] = xgb_model
        st.session_state["model_metrics"]["XGB"] = xgb_metrics

        # train RF on last 80% / test 20%
        status.info("Training RandomForest...")
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, shuffle=False)
        rf_model = train_other_model("RF", X_train, y_train)
        st.session_state["models"]["RF"] = rf_model
        st.session_state["model_metrics"]["RF"] = {"test_mae": float(mean_absolute_error(y_test, rf_model.predict(X_test)))}

        # optional LightGBM
        if HAVE_LGB:
            status.info("Training LightGBM...")
            lgb_model = train_other_model("LGB", X_train, y_train)
            st.session_state["models"]["LGB"] = lgb_model
            st.session_state["model_metrics"]["LGB"] = {"test_mae": float(mean_absolute_error(y_test, lgb_model.predict(X_test)))}

        # optional CatBoost
        if HAVE_CAT:
            status.info("Training CatBoost...")
            cat_model = train_other_model("CAT", X_train, y_train)
            st.session_state["models"]["CAT"] = cat_model
            st.session_state["model_metrics"]["CAT"] = {"test_mae": float(mean_absolute_error(y_test, cat_model.predict(X_test)))}

        status.success("Models trained and saved in session_state.")
        models_table_ph.dataframe(pd.DataFrame(st.session_state["model_metrics"]).T)

    except Exception as e:
        st.error(f"Training failed: {e}")

# Helper: craft input row for prediction from latest df
def craft_input_row(df_src, n_lags, feat_names):
    closes = df_src["Close"].astype(float).reset_index(drop=True)
    row_lags = {f"lag_{lag}": closes.iloc[-lag] for lag in range(1, n_lags+1)}
    enriched_aligned = compute_indicators(df_src, close_col="Close")
    enriched_aligned = enriched_aligned.iloc[n_lags:].reset_index(drop=True) if len(enriched_aligned) > n_lags else enriched_aligned
    extras_row = enriched_aligned.iloc[-1:].reset_index(drop=True) if not enriched_aligned.empty else pd.DataFrame()
    X_row = pd.DataFrame(columns=feat_names)
    for c in feat_names:
        if c.startswith("lag_"):
            X_row.at[0,c] = row_lags.get(c, np.nan)
        else:
            if not extras_row.empty and c in extras_row.columns:
                X_row.at[0,c] = extras_row.at[0,c]
            elif c in df_src.columns:
                X_row.at[0,c] = df_src.at[len(df_src)-1, c]
            else:
                X_row.at[0,c] = 0.0
    return X_row.fillna(method="ffill").fillna(0.0)

# REFRESH Now: append a tick to realtime buffer
def append_current_tick():
    try:
        p = fetch_current_price("BTCUSDT")
        t = datetime.utcnow()
        st.session_state["realtime"]["times"].append(t)
        st.session_state["realtime"]["prices"].append(float(p))
        if len(st.session_state["realtime"]["times"]) > live_window:
            excess = len(st.session_state["realtime"]["times"]) - live_window
            st.session_state["realtime"]["times"] = st.session_state["realtime"]["times"][excess:]
            st.session_state["realtime"]["prices"] = st.session_state["realtime"]["prices"][excess:]
        return t, p
    except Exception as e:
        st.warning(f"Tick fetch failed: {e}")
        return None, None

# UI buttons actions
if refresh_btn:
    t,p = append_current_tick()
    if t is not None:
        st.success(f"Fetched tick {p:.2f} at {t}")

if start_live:
    st.session_state["live_running"] = True

if stop_live:
    st.session_state["live_running"] = False

# Live loop (blocking) — not ideal for multi-user production but OK for prototype
if st.session_state["live_running"]:
    status.info("Live mode running — press Stop Live to stop.")
    try:
        while st.session_state["live_running"]:
            append_current_tick()
            time.sleep(live_interval)
    except KeyboardInterrupt:
        st.session_state["live_running"] = False
        status.info("Live stopped (KeyboardInterrupt).")

# PREDICT (single-shot)
if st.button("Predict (use trained models)"):
    if "models" not in st.session_state or not st.session_state["models"]:
        st.warning("Train models first.")
    elif st.session_state["last_df_for_preds"] is None:
        st.warning("No training data present; train or upload CSV first.")
    else:
        try:
            df_src = st.session_state["last_df_for_preds"].copy().reset_index(drop=True)
            feat_names = st.session_state["feat_names"]
            if len(df_src) < (n_lags + 1):
                st.error("Not enough rows to build lag features.")
            else:
                X_row = craft_input_row(df_src, n_lags, feat_names)
                preds = {}
                for name,model in st.session_state["models"].items():
                    try:
                        pv = float(model.predict(X_row.values)[0])
                    except Exception:
                        pv = float(model.predict(X_row)[0])
                    preds[name] = (pv, datetime.utcnow() + timedelta(seconds=5))

                # ensemble (simple average)
                ensemble_val = float(np.mean([v[0] for v in preds.values()]))
                preds["Ensemble"] = (ensemble_val, datetime.utcnow() + timedelta(seconds=5))

                # Key points
                if using_csv:
                    last_row = df_src.iloc[-1]
                    prev_row = df_src.iloc[-2] if len(df_src)>=2 else None
                    change=None; change_pct=None
                    try:
                        if prev_row is not None:
                            change = float(last_row["Close"]) - float(prev_row["Close"])
                            change_pct = (change / float(prev_row["Close"]))*100.0
                    except Exception:
                        pass
                    kp = {"date": last_row.get("Date",""), "ticker": last_row.get("Ticker",""), "24h_change": change, "24h_change_pct": change_pct, "high": last_row.get("High",None), "low": last_row.get("Low",None), "volume": last_row.get("Volume",None)}
                else:
                    b24 = fetch_binance_24h("BTCUSDT")
                    kp = {"date": datetime.utcnow(), "ticker": "BTCUSDT", "24h_change": b24["price_change"], "24h_change_pct": b24["price_change_percent"], "high": b24["high_price"], "low": b24["low_price"], "volume": b24["volume"]}

                keybox.markdown(f"""
                    <div style="background-color:#000000;color:#ffffff;padding:12px;border-radius:10px;border-left:5px solid #ffffff;margin-bottom:8px;">
                        <h4 style="margin-top:0;">📌 Key Points — {kp.get('ticker','')} (latest)</h4>
                        <ul style="font-size:14px;">
                            <li><b>Last date:</b> {kp.get('date')}</li>
                            <li><b>24h Change:</b> {kp.get('24h_change') if kp.get('24h_change') is not None else 'N/A'} USD ({kp.get('24h_change_pct'):.2f}% )</li>
                            <li><b>High:</b> {kp.get('high')}</li>
                            <li><b>Low:</b> {kp.get('low')}</li>
                            <li><b>Volume:</b> {kp.get('volume')}</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

                # Show model predictions table
                df_preds = pd.DataFrame([{ "model":k, "pred":v[0]} for k,v in preds.items()])
                metrics_ph.dataframe(df_preds.set_index("model"))

                # plot realtime data + model preds
                times = st.session_state["realtime"]["times"] if st.session_state["realtime"]["times"] else (list(pd.to_datetime(df_src["Date"].iloc[-live_window:])) if "Date" in df_src.columns else list(range(len(df_src))))
                prices = st.session_state["realtime"]["prices"] if st.session_state["realtime"]["prices"] else df_src["Close"].astype(float).tolist()[-len(times):]
                plot_series(chart_ph, times, prices, preds_dict=preds, title="Real-time price with model predictions")

                # feature importance for XGB if available
                if "XGB" in st.session_state["models"]:
                    try:
                        imp = st.session_state["models"]["XGB"].feature_importances_
                        plot_feature_importances(feat_imp_ph, feat_names, imp)
                    except Exception:
                        pass

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Always show small realtime chart (latest buffer)
if st.session_state["realtime"]["times"]:
    tlist = st.session_state["realtime"]["times"]
    plist = st.session_state["realtime"]["prices"]
    if len(tlist) > live_window:
        tlist = tlist[-live_window:]; plist = plist[-live_window:]
    plot_series(chart_ph, tlist, plist, title="Live price (buffer)")

st.markdown("---")
st.caption("Real-time chart + ensemble predictions. For production, consider converting the live loop into a non-blocking autorefresh with st_autorefresh or using Binance websockets for lower latency. This is educational — not financial advice.")
