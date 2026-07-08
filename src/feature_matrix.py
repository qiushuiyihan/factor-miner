"""Feature matrix builder — daily + intraday fund flow features."""

import numpy as np
import pandas as pd
from fetch_data import (fetch_daily_fundflow, fetch_minute_fundflow,
                        fetch_daily_kline_baidu, fetch_tick_data,
                        fetch_minute_kline_eastmoney, STOCK_POOL)


def build_daily_matrix(codes=None, lookback_days=60):
    """Build a feature matrix from daily fund flow data.

    Each row = one stock-day. Columns include raw fund flow values,
    lagged values (1-5 days), rolling means (5/10/20), and derived features.

    Args:
        codes: stock code list, defaults to STOCK_POOL
        lookback_days: number of trading days to include

    Returns:
        pd.DataFrame indexed by (date, code) with all feature columns
        plus 'forward_return_1d' as the prediction target.
    """
    if codes is None:
        codes = STOCK_POOL

    all_frames = []
    all_prices = []
    for code in codes:
        df = fetch_daily_fundflow(code)
        if df.empty:
            print(f"[WARN] no fund flow data for {code}, skipping")
            continue
        df["code"] = code
        all_frames.append(df)

        # Fetch price K-line for forward return target
        px = fetch_daily_kline_baidu(code)
        if not px.empty:
            px["code"] = code
            all_prices.append(px)

    if not all_frames:
        raise RuntimeError("No data fetched for any stock")

    raw = pd.concat(all_frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["code", "date"]).reset_index(drop=True)

    # Merge price data
    if all_prices:
        prices = pd.concat(all_prices, ignore_index=True)
        prices["date"] = pd.to_datetime(prices["date"])
        raw = raw.merge(prices[["date", "code", "close", "volume", "amount"]],
                        on=["date", "code"], how="left", suffixes=("", "_px"))
        raw["volume_px"] = raw["volume"]
        raw["close_px"] = raw["close"]
        raw = raw.drop(columns=["volume", "close"])  # avoid ambiguity with fund flow cols
    else:
        raw["close_px"] = np.nan
        raw["volume_px"] = np.nan

    flow_cols = ["main_net", "large_net", "mid_net", "small_net", "super_net"]

    result = raw.copy()

    # Per-code lag features (1-5 days)
    for col in flow_cols:
        for lag in range(1, 6):
            result[f"{col}_lag{lag}"] = result.groupby("code")[col].shift(lag)

    # Per-code rolling aggregates
    for col in flow_cols:
        result[f"{col}_ma5"] = result.groupby("code")[col].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        result[f"{col}_ma10"] = result.groupby("code")[col].transform(
            lambda x: x.rolling(10, min_periods=5).mean()
        )
        result[f"{col}_ma20"] = result.groupby("code")[col].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        result[f"{col}_std5"] = result.groupby("code")[col].transform(
            lambda x: x.rolling(5, min_periods=3).std()
        )

    # Acceleration: 5-day slope minus 20-day slope
    for col in flow_cols:
        result[f"{col}_accel"] = (
            result[f"{col}"] - result[f"{col}_ma5"]
        ) - (
            result[f"{col}_ma5"] - result[f"{col}_ma20"]
        )

    # Cross-flow ratios
    result["large_main_ratio"] = result["large_net"] / (result["main_net"].abs() + 1)
    result["super_main_ratio"] = result["super_net"] / (result["main_net"].abs() + 1)
    result["retail_vs_main"] = result["small_net"] / (result["main_net"].abs() + 1)

    # Volume-derived features (from price data)
    if "volume_px" in result.columns and result["volume_px"].notna().any():
        result["vol_ma5"] = result.groupby("code")["volume_px"].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        result["vol_ratio"] = result["volume_px"] / (result["vol_ma5"] + 1)
        result["vol_ma20"] = result.groupby("code")["volume_px"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        result["vol_breakout"] = result["vol_ma5"] / (result["vol_ma20"] + 1)

    # Forward return: next day's close / today's close - 1 (the real alpha target)
    result["forward_return_1d"] = result.groupby("code")["close_px"].transform(
        lambda x: x.shift(-1) / x - 1
    )
    result["forward_return_5d"] = result.groupby("code")["close_px"].transform(
        lambda x: (x.shift(-5) / x - 1)
    )

    # Drop rows with NaN (first/last few days per stock)
    result = result.dropna()

    # Keep only the last lookback_days per stock
    result = result.groupby("code").tail(lookback_days).reset_index(drop=True)

    return result


def build_intraday_features(codes=None, minute_lookback=10):
    """Extract intraday flow pattern features from minute-level fund flow.

    For each stock-day, computes:
      - Session-level net flows (morning/afternoon/tail/open)
      - Intraday volatility and trend of main_net
      - Large/super order participation ratios
      - Morning→afternoon reversal signal

    Args:
        codes: stock list, defaults to STOCK_POOL
        minute_lookback: number of recent trading days to fetch minute data

    Returns:
        pd.DataFrame with columns: date, code, <intraday features>
    """
    if codes is None:
        codes = STOCK_POOL

    all_rows = []
    for code in codes:
        df = fetch_minute_fundflow(code)
        if df.empty:
            print(f"[WARN] no minute data for {code}, skipping intraday")
            continue

        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date
        df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60

        for day, group in df.groupby("date"):
            if len(group) < 30:  # skip partial days
                continue

            g = group.sort_values("time")
            morning = g[g["hour"] < 12.0]
            afternoon = g[g["hour"] >= 13.0]
            tail = g[g["hour"] >= 14.5]
            open30 = g[g["hour"] <= 10.0]

            # Session net flows
            morning_main = morning["main_net"].sum()
            afternoon_main = afternoon["main_net"].sum()
            tail_main = tail["main_net"].sum()
            open_main = open30["main_net"].sum()

            # Intraday volatility: std of main_net across minutes
            main_vol = g["main_net"].std() / (g["main_net"].abs().mean() + 1)

            # Intraday trend: slope of cumulative main_net (accumulation vs distribution)
            cumsum = g["main_net"].cumsum().values
            x = np.arange(len(cumsum))
            trend = np.polyfit(x, cumsum, 1)[0] if len(x) > 1 else 0
            trend_norm = trend / (abs(g["main_net"].mean()) + 1)

            # Reversal: afternoon vs morning sign
            reversal = 1 if (morning_main < 0 and afternoon_main > 0) else (
                -1 if (morning_main > 0 and afternoon_main < 0) else 0
            )

            # Order participation ratios
            large_ratio = g["large_net"].sum() / (abs(g["main_net"].sum()) + 1)
            super_ratio = g["super_net"].sum() / (abs(g["main_net"].sum()) + 1)
            retail_ratio = g["small_net"].sum() / (abs(g["main_net"].sum()) + 1)

            # Tail intensity: tail main_net as fraction of total
            tail_share = tail_main / (abs(g["main_net"].sum()) + 1)

            # Consecutive positive minutes
            signs = (g["main_net"].values > 0).astype(int)
            cons_pos = max(
                (len(list(g)) for _, g in __import__("itertools").groupby(signs) if _ == 1),
                default=0
            )

            all_rows.append({
                "date": pd.Timestamp(day),
                "code": code,
                "intra_morning_main": morning_main,
                "intra_afternoon_main": afternoon_main,
                "intra_tail_main": tail_main,
                "intra_open_main": open_main,
                "intra_main_vol": main_vol,
                "intra_main_trend": trend_norm,
                "intra_reversal": reversal,
                "intra_large_ratio": large_ratio,
                "intra_super_ratio": super_ratio,
                "intra_retail_ratio": retail_ratio,
                "intra_tail_share": tail_share,
                "intra_cons_pos_min": cons_pos,
            })

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    result = result.sort_values(["code", "date"]).reset_index(drop=True)
    return result


def build_enriched_matrix(codes=None, daily_lookback=60, intraday_lookback=10):
    """Merge daily features with intraday pattern features.

    When intraday data is available (e.g. today's snapshot), it's merged per-stock.
    Falls back to daily-only matrix when intraday data doesn't cover the date range
    (which is normal — minute data is typically only available for the current day).

    Args:
        codes: stock list
        daily_lookback: days of daily fund flow history
        intraday_lookback: passed to build_intraday_features

    Returns:
        pd.DataFrame — enriched if intraday data available, daily baseline otherwise
    """
    print(f"      Building daily matrix ({daily_lookback}d lookback)...")
    daily = build_daily_matrix(codes, lookback_days=daily_lookback)

    print(f"      Building intraday features...")
    intra = build_intraday_features(codes, minute_lookback=intraday_lookback)

    if intra.empty:
        print("      → No intraday data, using daily matrix only")
        return daily

    # Merge on (date, code)
    enriched = daily.merge(intra, on=["date", "code"], how="left")
    intra_cols = [c for c in intra.columns if c not in ("date", "code")]

    # Count rows with intraday data
    has_intra = enriched[intra_cols[0]].notna().sum() if intra_cols else 0
    if has_intra == 0:
        print(f"      → Intraday dates don't overlap with daily (got {len(intra)} intra rows), using daily only")
        return daily

    # Forward-fill missing intraday features within each stock
    enriched[intra_cols] = enriched.groupby("code")[intra_cols].ffill()

    print(f"      Enriched matrix: {enriched.shape[0]} rows × {enriched.shape[1]} cols ({has_intra} rows with intraday data)")
    return enriched


def build_tick_features(codes=None):
    """Extract features from tick-by-tick transaction data.

    Per stock, computes:
      - buy_sell_imbalance: (buy_vol - sell_vol) / total_vol
      - avg_trade_size: mean volume per tick
      - large_trade_ratio: fraction of trades with vol > 90th percentile
      - block_trade_count: number of >50k share trades
      - tick_arrival_rate: total ticks / trading minutes
      - vwap_deviation: (close - vwap) / vwap
      - buy_aggression: mean price of buy ticks / mean price of sell ticks

    Args:
        codes: stock list, defaults to STOCK_POOL

    Returns:
        pd.DataFrame with columns: date, code, <tick features>
    """
    if codes is None:
        codes = STOCK_POOL

    rows = []
    for code in codes:
        ticks = fetch_tick_data(code)
        if not ticks:
            print(f"[WARN] No tick data for {code}")
            continue

        df = pd.DataFrame(ticks)
        df["v"] = df["v"].astype(float)
        df["p"] = df["p"].astype(float)
        df["ts"] = df["ts"].astype(int)

        total_vol = df["v"].sum()
        buy_vol = df[df["ts"] == 1]["v"].sum()
        sell_vol = df[df["ts"] == 2]["v"].sum()

        # Buy/sell imbalance
        imbalance = (buy_vol - sell_vol) / (total_vol + 1)

        # Average trade size
        avg_size = df["v"].mean()

        # Large trade ratio (top 10% by volume)
        threshold = df["v"].quantile(0.9)
        large_ratio = (df["v"] > threshold).mean()

        # Block trades (>50k shares)
        block_count = (df["v"] > 50000).sum()

        # VWAP
        vwap = (df["p"] * df["v"]).sum() / (total_vol + 1)
        close_price = df["p"].iloc[-1]
        vwap_dev = (close_price - vwap) / (vwap + 1e-10)

        # Buy aggression: are buy orders at higher prices?
        buy_prices = df[df["ts"] == 1]["p"]
        sell_prices = df[df["ts"] == 2]["p"]
        buy_agg = buy_prices.mean() / (sell_prices.mean() + 1e-10) if len(buy_prices) and len(sell_prices) else 1.0

        # Tick arrival rate (ticks per minute)
        if "t" in df.columns:
            times = pd.to_datetime(df["d"] + " " + df["t"])
            trading_minutes = (times.max() - times.min()).total_seconds() / 60
            arrival_rate = len(df) / (trading_minutes + 1)
        else:
            arrival_rate = 0

        # Volume distribution skew (high vol at high prices = bullish)
        vol_weighted_price_skew = (
            (df["p"] - df["p"].mean()) * df["v"]
        ).sum() / (total_vol * df["p"].std() + 1e-10)

        date_str = df["d"].iloc[0] if "d" in df.columns else str(pd.Timestamp.now().date())

        rows.append({
            "date": pd.Timestamp(date_str),
            "code": code,
            "tick_imbalance": imbalance,
            "tick_avg_size": avg_size,
            "tick_large_ratio": large_ratio,
            "tick_block_count": block_count,
            "tick_vwap_dev": vwap_dev,
            "tick_buy_agg": buy_agg,
            "tick_arrival_rate": arrival_rate,
            "tick_vol_skew": vol_weighted_price_skew,
            "tick_total_vol": total_vol,
            "tick_close": close_price,
        })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["code", "date"]).reset_index(drop=True)


def build_price_matrix(codes=None, lookback_days=120):
    """Build a price-only feature matrix from Baidu K-line (no fund flow dependency).

    Falls back when eastmoney is unreachable. Uses 6 years of OHLCV data.
    Computes price-derived features + forward return target.
    """
    if codes is None:
        codes = STOCK_POOL

    all_frames = []
    for code in codes:
        px = fetch_daily_kline_baidu(code)
        if px.empty:
            print(f"[WARN] no price data for {code}, skipping")
            continue
        px["code"] = code
        all_frames.append(px)

    if not all_frames:
        raise RuntimeError("No price data fetched for any stock")

    raw = pd.concat(all_frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["code", "date"]).reset_index(drop=True)

    result = raw.copy()

    # ── Price features ──
    # Returns
    result["return_1d"] = result.groupby("code")["close"].pct_change()
    result["return_1d_lag1"] = result.groupby("code")["return_1d"].shift(1)

    # Amplitude
    result["amplitude"] = (result["high"] - result["low"]) / (result["close"] + 1e-10)

    # Price position within day
    result["price_position"] = (result["close"] - result["low"]) / (result["high"] - result["low"] + 1e-10)

    # Gap
    result["gap"] = result.groupby("code")["open"].transform(
        lambda x: x / x.shift(1) - 1
    )

    # Lagged closes
    for lag in range(1, 6):
        result[f"close_lag{lag}"] = result.groupby("code")["close"].shift(lag)

    # Moving averages
    for w in [5, 10, 20, 60]:
        result[f"close_ma{w}"] = result.groupby("code")["close"].transform(
            lambda x: x.rolling(w, min_periods=max(3, w//2)).mean()
        )

    # MA deviations
    for w in [5, 10, 20]:
        result[f"close_dev{w}"] = result["close"] / (result[f"close_ma{w}"] + 1e-10) - 1

    # Close std
    result["close_std20"] = result.groupby("code")["close"].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )

    # ── Volume features ──
    result["vol_ma5"] = result.groupby("code")["volume"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    result["vol_ma20"] = result.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    result["vol_ratio"] = result["volume"] / (result["vol_ma5"] + 1)
    result["vol_breakout"] = result["vol_ma5"] / (result["vol_ma20"] + 1)
    result["amount_ma5"] = result.groupby("code")["amount"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    result["amount_ratio"] = result["amount"] / (result["amount_ma5"] + 1)

    # Volume-Price interaction
    result["vp_corr10"] = result.groupby("code").apply(
        lambda g: g["volume"].rolling(10).corr(g["close"])
    ).reset_index(level=0, drop=True)

    # ── Forward target ──
    result["forward_return_1d"] = result.groupby("code")["close"].transform(
        lambda x: x.shift(-1) / x - 1
    )
    result["forward_return_5d"] = result.groupby("code")["close"].transform(
        lambda x: x.shift(-5) / x - 1
    )

    result = result.dropna()
    result = result.groupby("code").tail(lookback_days).reset_index(drop=True)
    return result


# ── Self-check ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Daily matrix ===")
    mat = build_daily_matrix(["688017", "002896"], lookback_days=20)
    print(f"Shape: {mat.shape}")
    print(f"Date range: {mat['date'].min()} ~ {mat['date'].max()}")

    print("\n=== Intraday features ===")
    intra = build_intraday_features(["688017", "002896"])
    print(f"Shape: {intra.shape}")
    if not intra.empty:
        print(f"Columns: {list(intra.columns)}")
        print(intra.head(2))

    print("\n=== Enriched matrix ===")
    enriched = build_enriched_matrix(["688017", "002896"], daily_lookback=20, intraday_lookback=5)
    print(f"Shape: {enriched.shape}")
    if not enriched.empty:
        print(f"Total columns: {len(enriched.columns)}")
