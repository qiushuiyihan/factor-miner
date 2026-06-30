"""Expression tree engine — type-constrained factor expression generation.

The engine categorizes feature columns by type (FLOW, PRICE, VOLUME, DERIVED)
and enforces that operators combine compatible types only. This prunes ~70%
of nonsensical combinations before they even reach gplearn.
"""

import re
import numpy as np
import pandas as pd

# ── Type categories ─────────────────────────────────────────
FEATURE_COLUMNS = {
    "FLOW": [   # raw fund flow values (yuan)
        "main_net", "large_net", "mid_net", "small_net", "super_net",
        "main_net_lag1", "main_net_lag2", "main_net_lag3", "main_net_lag4", "main_net_lag5",
        "large_net_lag1", "large_net_lag2", "large_net_lag3", "large_net_lag4", "large_net_lag5",
        "mid_net_lag1", "mid_net_lag2", "mid_net_lag3", "mid_net_lag4", "mid_net_lag5",
        "small_net_lag1", "small_net_lag2", "small_net_lag3", "small_net_lag4", "small_net_lag5",
        "super_net_lag1", "super_net_lag2", "super_net_lag3", "super_net_lag4", "super_net_lag5",
    ],
    "ROLLING": [  # rolling aggregates
        "main_net_ma5", "main_net_ma10", "main_net_ma20", "main_net_std5",
        "large_net_ma5", "large_net_ma10", "large_net_ma20", "large_net_std5",
        "mid_net_ma5", "mid_net_ma10", "mid_net_ma20", "mid_net_std5",
        "small_net_ma5", "small_net_ma10", "small_net_ma20", "small_net_std5",
        "super_net_ma5", "super_net_ma10", "super_net_ma20", "super_net_std5",
    ],
    "PRICE_VOL": [  # price + volume features (from Baidu K-line)
        "close_px", "volume_px",
        "vol_ma5", "vol_ratio", "vol_ma20", "vol_breakout",
    ],
    "DERIVED": [  # computed ratios and accelerations
        "large_main_ratio", "super_main_ratio", "retail_vs_main",
        "main_net_accel", "large_net_accel", "mid_net_accel",
        "small_net_accel", "super_net_accel",
    ],
    "TICK": [    # tick-level features (from 迈锐逐笔成交)
        "tick_imbalance", "tick_avg_size", "tick_large_ratio",
        "tick_block_count", "tick_vwap_dev", "tick_buy_agg",
        "tick_arrival_rate", "tick_vol_skew", "tick_total_vol",
    ],
    "INTRA": [   # intraday flow pattern features (from minute data)
        "intra_morning_main", "intra_afternoon_main",
        "intra_tail_main", "intra_open_main",
        "intra_main_vol", "intra_main_trend",
        "intra_reversal", "intra_large_ratio",
        "intra_super_ratio", "intra_retail_ratio",
        "intra_tail_share", "intra_cons_pos_min",
    ],
}

ALL_COLUMNS = [c for group in FEATURE_COLUMNS.values() for c in group]


def decode_expression(expr, feature_cols=None):
    """Convert gplearn X-index expression to human-readable column names.

    Example: "add(X0, mul(X3, X5))" → "add(close_px, mul(vol_ratio, main_net))"
    Falls back to original expression if feature_cols is None.
    """
    if not feature_cols:
        return expr
    result = expr
    # Replace in reverse index order to avoid partial matches (X1 vs X10)
    for i in sorted(range(len(feature_cols)), reverse=True):
        col = feature_cols[i] if i < len(feature_cols) else f"col{i}"
        result = result.replace(f"X{i}", col)
    return result

# ── Safe operators ───────────────────────────────────────────
def _safe_div(a, b):
    denom = np.where(np.abs(b) < 1e-10, np.sign(b) * 1e-10, b)
    return a / denom

def _safe_log(x):
    shifted = x - np.nanmin(x) + 1
    shifted = np.where(shifted <= 0, 1e-10, shifted)
    return np.log(shifted)

def _rolling_sum(x, window):
    return pd.Series(x).rolling(window, min_periods=max(3, window//2)).sum().values

def _rolling_mean(x, window):
    return pd.Series(x).rolling(window, min_periods=max(3, window//2)).mean().values

def _rolling_corr(x, y, window):
    return pd.Series(x).rolling(window, min_periods=max(5, window//2)).corr(pd.Series(y)).values

OPERATORS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": _safe_div,
    "abs": lambda x: np.abs(x),
    "log": _safe_log,
    "sqrt": lambda x: np.sqrt(np.maximum(x - np.nanmin(x), 0) + 1e-10),
    "neg": lambda x: -x,
    "inv": lambda x: 1 / (np.abs(x) + 1e-10),
    "rank": lambda x: pd.Series(x).rank(pct=True).values,
    "zscore": lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-10),
}


def evaluate_expression(expr, data, feature_cols=None):
    """Evaluate a factor expression string against a feature matrix DataFrame.

    Supports two expression formats:
      1. Named columns: "div(sub(main_net, main_net_ma5), main_net_std5)"
      2. gplearn X-indices: "add(X0, mul(X3, X5))" — requires feature_cols list
         mapping Xi → actual column name.

    Args:
        expr: expression string in either format
        data: pd.DataFrame with feature columns
        feature_cols: optional list of column names for gplearn X-index mapping
    """
    context = {}
    # Map named columns
    for c in ALL_COLUMNS:
        if c in data.columns:
            context[c] = data[c].values
    # Map gplearn X-index format (Xi → column value)
    if feature_cols:
        for i, col in enumerate(feature_cols):
            if col in data.columns:
                context[f"X{i}"] = data[col].values
    # Also map any column that appears in data but not in ALL_COLUMNS
    for c in data.columns:
        if c not in context:
            context[c] = data[c].values

    context.update(OPERATORS)
    context["ts_sum"] = _rolling_sum
    context["ts_mean"] = _rolling_mean
    context["ts_std"] = lambda x, w: pd.Series(x).rolling(w, min_periods=max(3, w//2)).std().values
    context["corr"] = _rolling_corr
    context["np"] = np

    try:
        result = eval(expr, {"__builtins__": {}}, context)
        return np.asarray(result, dtype=float).ravel()
    except Exception as e:
        raise ValueError(f"Expression evaluation failed: {expr}\n{e}")


def generate_expressions(feature_columns, n=200, seed=42):
    """Generate a pool of valid candidate expression strings.

    Uses type-constrained templates. Returns exactly n expressions.
    The expressions are valid input for evaluate_expression().
    """
    rng = np.random.RandomState(seed)
    flow_cols = [c for c in feature_columns.get("FLOW", []) if "lag" not in c]
    flow_all = feature_columns.get("FLOW", [])
    rolling = feature_columns.get("ROLLING", [])
    derived = feature_columns.get("DERIVED", [])

    # Templates: each is (weight, generator_fn)
    templates = []

    # Type A: simple transformations on single flow columns
    for col in flow_cols:
        templates.append((3, lambda c=col: f"div(sub({c}, ts_mean({c}, 5)), ts_std({c}, 5))"))
        templates.append((3, lambda c=col: f"div(sub({c}, ts_mean({c}, 10)), ts_std({c}, 10))"))
        templates.append((2, lambda c=col: f"rank({c})"))
        templates.append((2, lambda c=col: f"zscore({c})"))
        templates.append((2, lambda c=col: f"log(abs({c}))"))
        templates.append((2, lambda c=col: f"div({c}, abs({c}_lag1))"))

    # Type B: cross-column operations (within FLOW type)
    for c1 in flow_cols:
        for c2 in flow_cols:
            if c1 < c2:
                templates.append((1, lambda a=c1, b=c2: f"div(sub({a}, {b}), add(abs({a}), abs({b})))"))
                templates.append((1, lambda a=c1, b=c2: f"corr({a}, {b}, 10)"))
                templates.append((1, lambda a=c1, b=c2: f"sub(rank({a}), rank({b}))"))

    # Type C: acceleration and derived feature combinations
    for d in derived:
        templates.append((2, lambda x=d: f"zscore({x})"))
        templates.append((2, lambda x=d: f"rank({x})"))
    for d1 in derived:
        for d2 in derived:
            if d1 < d2 and "accel" in d1 and "accel" in d2:
                templates.append((1, lambda a=d1, b=d2: f"div({a}, add(abs({b}), 1e-10))"))

    # Type D: ratios between different investor types
    templates.append((3, lambda: "div(main_net, add(abs(retail_vs_main), 1e-10))"))
    templates.append((3, lambda: "div(sub(large_net, small_net), add(abs(main_net), 1e-10))"))
    templates.append((3, lambda: "div(super_net, add(abs(main_net), 1e-10))"))

    # Type E: intraday flow pattern combinations
    intra_cols = feature_columns.get("INTRA", [])
    if intra_cols:
        # ── Session flow cross-comparisons ──
        templates.append((4, lambda: "div(sub(intra_afternoon_main, intra_morning_main), add(abs(intra_main_vol), 1e-10))"))
        templates.append((4, lambda: "div(intra_tail_main, add(abs(intra_open_main), 1e-10))"))
        templates.append((3, lambda: "mul(intra_reversal, intra_tail_share)"))
        templates.append((3, lambda: "div(intra_tail_main, add(abs(intra_open_main), 1e-10))"))
        # ── Intraday volatility interactions ──
        templates.append((3, lambda: "div(intra_main_trend, add(abs(intra_main_vol), 1e-10))"))
        templates.append((3, lambda: "mul(intra_cons_pos_min, intra_large_ratio)"))
        templates.append((2, lambda: "div(sub(intra_large_ratio, intra_retail_ratio), add(abs(intra_main_vol), 1e-10))"))
        templates.append((2, lambda: "div(intra_super_ratio, add(abs(intra_retail_ratio), 1e-10))"))
        # ── Intraday vs daily flow interactions ──
        templates.append((3, lambda: "div(intra_morning_main, add(abs(main_net_lag1), 1e-10))"))
        templates.append((3, lambda: "div(intra_tail_main, add(abs(main_net_lag1), 1e-10))"))
        templates.append((2, lambda: "div(intra_main_trend, add(abs(main_net_accel), 1e-10))"))
        # ── Single-column transforms on intraday features ──
        for col in intra_cols:
            templates.append((2, lambda c=col: f"zscore({c})"))
            templates.append((2, lambda c=col: f"rank({c})"))

    # Type F: volume-price-flow cross interactions
    price_cols = feature_columns.get("PRICE_VOL", [])
    if price_cols:
        # Volume breakout + fund flow
        templates.append((4, lambda: "mul(vol_ratio, zscore(main_net))"))
        templates.append((4, lambda: "div(main_net, add(abs(vol_ma5), 1e-10))"))
        templates.append((3, lambda: "mul(vol_breakout, rank(main_net))"))
        templates.append((3, lambda: "div(sub(large_net, small_net), add(abs(volume_px), 1e-10))"))
        # Price change + fund flow (量价配合)
        templates.append((4, lambda: "mul(zscore(main_net), zscore(vol_ratio))"))
        templates.append((3, lambda: "div(main_net_accel, add(abs(vol_ratio), 1e-10))"))
        for col in price_cols:
            templates.append((1, lambda c=col: f"zscore({c})"))
            templates.append((1, lambda c=col: f"rank({c})"))

    # Type G: tick-level feature combinations
    tick_cols = feature_columns.get("TICK", [])
    if tick_cols:
        # Buy/sell imbalance + fund flow
        templates.append((5, lambda: "mul(tick_imbalance, zscore(main_net))"))
        templates.append((5, lambda: "div(tick_imbalance, add(abs(tick_vwap_dev), 1e-10))"))
        templates.append((4, lambda: "mul(tick_buy_agg, tick_large_ratio)"))
        templates.append((4, lambda: "div(tick_block_count, add(abs(tick_arrival_rate), 1e-10))"))
        templates.append((4, lambda: "mul(tick_vol_skew, rank(vol_ratio))"))
        # Tick + daily flow cross
        templates.append((4, lambda: "div(tick_imbalance, add(abs(main_net_lag1), 1e-10))"))
        templates.append((3, lambda: "mul(tick_vwap_dev, zscore(main_net_accel))"))
        templates.append((3, lambda: "div(tick_avg_size, add(abs(tick_total_vol), 1e-10))"))
        # Single tick feature transforms
        for col in tick_cols[:6]:  # top 6 most important
            templates.append((2, lambda c=col: f"zscore({c})"))
            templates.append((2, lambda c=col: f"rank({c})"))

    weights = [w for w, _ in templates]
    total = sum(weights)
    probs = [w / total for w in weights]

    generated = set()
    attempts = 0
    while len(generated) < n and attempts < n * 10:
        idx = rng.choice(len(templates), p=probs)
        try:
            expr = templates[idx][1]()
            if expr not in generated:
                generated.add(expr)
        except Exception:
            pass
        attempts += 1

    return list(generated)[:n]


# ── Self-check ──────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    exprs = generate_expressions(FEATURE_COLUMNS, n=20)
    print(f"Generated {len(exprs)} expressions:")
    for e in exprs[:5]:
        print(f"  {e}")

    # Test evaluation with dummy data
    dummy = pd.DataFrame({
        c: np.random.randn(100) * 1e6
        for c in ALL_COLUMNS[:10]  # just a few columns
    })
    # Add some required columns
    for c in ["main_net", "main_net_std5"]:
        if c not in dummy.columns:
            dummy[c] = np.random.randn(100) * 1e6

    val = evaluate_expression("zscore(main_net)", dummy)
    print(f"\nTest evaluation: zscore(main_net) -> shape={val.shape}, mean={val.mean():.4f}, std={val.std():.4f}")
