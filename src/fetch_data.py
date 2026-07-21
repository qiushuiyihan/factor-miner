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
    except Exception as e:
        print(f"[WARN] fund flow fetch failed for {code}: {e}")
        return pd.DataFrame()


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
    except Exception as e:
        print(f"[WARN] daily fund flow fetch failed for {code}: {e}")
        return pd.DataFrame()


def fetch_daily_kline_baidu(code, start_time=""):
    """Fetch daily K-line (OHLCV + MA) from Baidu Finance API.

    Returns DataFrame with columns: date, open, close, high, low, volume, amount.
    ktype=1 means daily bars. No API key needed, no IP blocking risk.
    """
    url = "https://finance.pae.baidu.com/selfselect/getstockquotation"
    params = {
        "all": "1", "isIndex": "false", "isBk": "false", "isBlock": "false",
        "isFutures": "false", "isStock": "true", "newFormat": "1",
        "group": "quotation_kline_ab", "finClientType": "pc",
        "code": code, "start_time": start_time, "ktype": "1",
    }
    headers = {
        "User-Agent": UA,
        "Accept": "application/vnd.finance-web.v1+json",
        "Origin": "https://gushitong.baidu.com",
        "Referer": "https://gushitong.baidu.com/",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        d = r.json()
        result = d.get("Result", {})
        md = result.get("newMarketData", {})
        keys = md.get("keys", [])
        rows_str = md.get("marketData", "")

        if not keys or not rows_str:
            return pd.DataFrame()

        key_map = {k: i for i, k in enumerate(keys)}
        rows = []
        for line in rows_str.split(";"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < len(keys):
                continue
            rows.append({
                "date": parts[key_map.get("time", 0)],
                "open": float(parts[key_map.get("open", 1)]),
                "close": float(parts[key_map.get("close", 2)]),
                "high": float(parts[key_map.get("high", 3)]),
                "low": float(parts[key_map.get("low", 4)]),
                "volume": float(parts[key_map.get("volume", 5)]),
                "amount": float(parts[key_map.get("amount", 6)]),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[WARN] Baidu K-line failed for {code}: {e}")
        return pd.DataFrame()


def fetch_tick_data(code):
    """Fetch daily tick-by-tick transaction data from 迈锐API.

    Each tick: {d: date, t: time(HH:mm:ss), v: volume(shares), p: price, ts: direction}
    ts: 0=neutral, 1=buy, 2=sell. Returns list of dicts.
    Free licence required from https://www.mairuiapi.com/getlicence.
    """
    licence = "7DFC8EA6-8F7A-4765-B37D-356EA3F58829"
    url = f"http://api.mairuiapi.com/hsrl/zbjy/{code}/{licence}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"[WARN] Tick API returned {r.status_code} for {code}")
            return []
        data = r.json()
        if isinstance(data, list) and len(data) > 0 and "d" in data[0]:
            return data
        return []
    except Exception as e:
        print(f"[WARN] Tick fetch failed for {code}: {e}")
        return []


def fetch_minute_kline_eastmoney(code, klt=5, limit=2000):
    """Fetch historical minute K-line from eastmoney push2his.

    Args:
        code: 6-digit stock code
        klt: bar size — 5=5min, 15=15min, 30=30min, 60=60min
        limit: max bars to return (API caps at ~1533 for 5min)

    Returns DataFrame with columns: time, open, close, high, low, volume, amount.
    5-min historical data goes back ~32 trading days.
    """
    market_code = 1 if code.startswith("6") else 0
    secid = f"{market_code}.{code}"
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid, "klt": str(klt), "fqt": "1",
        "lmt": str(limit), "end": "20500101",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=20)
        d = r.json()
        rows = []
        for line in d.get("data", {}).get("klines", []):
            parts = line.split(",")
            if len(parts) >= 7:
                rows.append({
                    "time": parts[0],
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "volume": float(parts[5]),
                    "amount": float(parts[6]),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[WARN] Minute K-line failed for {code}: {e}")
        return pd.DataFrame()


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
