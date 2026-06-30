"""Feature matrix builder — align daily fund flow into ML-ready matrix."""

import numpy as np
import pandas as pd
from fetch_data import fetch_daily_fundflow, STOCK_POOL


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
    for code in codes:
        df = fetch_daily_fundflow(code)
        if df.empty:
            print(f"[WARN] no daily data for {code}, skipping")
            continue
        df["code"] = code
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No data fetched for any stock")

    raw = pd.concat(all_frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["code", "date"]).reset_index(drop=True)

    # Forward return: next day's main_net as proxy for price movement signal
    # (actual forward return requires price data; fund flow delta is the alpha target)
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

    # Forward return: main_net next day as target (normalized by absolute mean)
    result["forward_main_net_1d"] = result.groupby("code")["main_net"].shift(-1)
    result["forward_main_net_5d"] = result.groupby("code")["main_net"].transform(
        lambda x: x.shift(-5).rolling(5, min_periods=1).mean()
    )

    # Drop rows with NaN (first/last few days per stock)
    result = result.dropna()

    # Keep only the last lookback_days per stock
    result = result.groupby("code").tail(lookback_days).reset_index(drop=True)

    return result


# ── Self-check ──────────────────────────────────────────────
if __name__ == "__main__":
    mat = build_daily_matrix(["688017", "002896"], lookback_days=30)
    print(f"Matrix shape: {mat.shape}")
    print(f"Columns ({len(mat.columns)}): {list(mat.columns)[:15]}...")
    print(f"Date range: {mat['date'].min()} ~ {mat['date'].max()}")
    print(f"Stocks: {mat['code'].unique()}")
    print(mat.head())
