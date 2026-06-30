"""Data fetcher — minute K-line (mootdx TCP) + fund flow (eastmoney HTTP)."""

import time
import random
from datetime import date as get_date
import pandas as pd
import requests
from mootdx.quotes import Quotes

# ── Stock pool: humanoid robot sector ──────────────────────
STOCK_POOL = [
    "688017",  # 绿的谐波 — 谐波减速器
    "301550",  # 斯菱股份 — 谐波减速器/无框力矩电机
    "002896",  # 中大力德 — 谐波减速器/滚珠丝杠
    "601100",  # 恒立液压 — 行星滚柱丝杠
    "603667",  # 五洲新春 — 行星滚柱丝杠
    "603009",  # 北特科技 — 行星滚柱丝杠
    "003021",  # 兆威机电 — 无框力矩电机/灵巧手
    "300124",  # 汇川技术 — 无框力矩电机
    "300007",  # 汉威科技 — 六维力传感器
    "688507",  # 索辰科技 — 六维力传感器
    "002765",  # 蓝黛科技 — 灵巧手
    "688623",  # 双元科技 — 滚珠丝杠
]

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# ── Eastmoney rate limiter ──────────────────────────────────
EM_SESSION = requests.Session()
EM_SESSION.headers.update({"User-Agent": UA})
EM_MIN_INTERVAL = 1.5
_em_last_call = [0.0]

def em_get(url, params=None, headers=None, timeout=15, **kwargs):
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return EM_SESSION.get(url, params=params, headers=headers, timeout=timeout, **kwargs)
    finally:
        _em_last_call[0] = time.time()


def fetch_minute_kline(code, date=None):
    """Fetch 1-min K-line from mootdx. Returns DataFrame with columns:
    time, open, high, low, close, vol, amount.
    category=7 means 1-minute bars, offset=240 covers full trading day (4h × 60min)."""
    if date is None:
        date = get_date.today().strftime("%Y-%m-%d")
    client = Quotes.factory(market="std")
    bars = client.bars(symbol=code, category=7, offset=240)
    if bars is None or len(bars) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df = df.rename(columns={
        "open": "open", "close": "close", "high": "high",
        "low": "low", "vol": "vol", "amount": "amount"
    })
    # mootdx returns no explicit time column for minute bars; we generate
    # 240 minutes starting from 09:30
    if "time" not in df.columns:
        base_times = pd.date_range(f"{date} 09:30", periods=120, freq="1min").union(
            pd.date_range(f"{date} 13:00", periods=120, freq="1min")
        )[: len(df)]
        df["time"] = base_times
    return df[["time", "open", "high", "low", "close", "vol", "amount"]]


def fetch_minute_fundflow(code, date=None):
    """Fetch minute-level fund flow from eastmoney push2.
    Returns DataFrame with columns: time, main_net, large_net, mid_net, small_net, super_net.
    Unit: yuan (元)."""
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    url = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
    params = {
        "secid": secid, "klt": 1,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=10)
        d = r.json()
    except Exception as e:
        print(f"[WARN] fund flow fetch failed for {code}: {e}")
        return pd.DataFrame()

    rows = []
    for line in d.get("data", {}).get("klines", []):
        parts = line.split(",")
        if len(parts) >= 6:
            rows.append({
                "time": parts[0],
                "main_net": float(parts[1]),
                "small_net": float(parts[2]),
                "mid_net": float(parts[3]),
                "large_net": float(parts[4]),
                "super_net": float(parts[5]),
            })
    return pd.DataFrame(rows)


def fetch_daily_fundflow(code):
    """Fetch daily fund flow (last 120 trading days) from eastmoney push2his.
    Returns DataFrame with columns: date, main_net, large_net, mid_net, small_net, super_net."""
    market_code = 1 if code.startswith("6") else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": f"{market_code}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "lmt": "120",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        d = r.json()
    except Exception as e:
        print(f"[WARN] daily fund flow fetch failed for {code}: {e}")
        return pd.DataFrame()

    rows = []
    for line in d.get("data", {}).get("klines", []):
        parts = line.split(",")
        if len(parts) >= 7:
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]) if parts[1] != "-" else 0,
                "small_net": float(parts[2]) if parts[2] != "-" else 0,
                "mid_net": float(parts[3]) if parts[3] != "-" else 0,
                "large_net": float(parts[4]) if parts[4] != "-" else 0,
                "super_net": float(parts[5]) if parts[5] != "-" else 0,
            })
    return pd.DataFrame(rows)


# ── Self-check ──────────────────────────────────────────────
if __name__ == "__main__":
    code = STOCK_POOL[0]  # 绿的谐波
    print(f"Testing fetch on {code}...")

    # K-line: mootdx may be unreachable (non-fatal)
    try:
        df_k = fetch_minute_kline(code)
        print(f"Minute K-line: {len(df_k)} rows")
        print(df_k.head(2))
    except Exception as e:
        print(f"[INFO] Minute K-line unavailable (mootdx server unreachable): {e}")

    # Fund flow: primary data source (eastmoney HTTP)
    df_f = fetch_minute_fundflow(code)
    print(f"Minute fund flow: {len(df_f)} rows")
    print(df_f.head(2))

    df_d = fetch_daily_fundflow(code)
    print(f"Daily fund flow: {len(df_d)} rows")
    print(df_d.head(2))
