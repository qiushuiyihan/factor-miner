"""Validator — 3-layer IC validation + correlation-based deduplication."""

import sys
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from expression_tree import evaluate_expression  # noqa: E402
from genetic_miner import spearman_ic         # noqa: E402


def three_layer_validate(expr, data, target_col="forward_return_1d", n_windows=3,
                         feature_cols=None):
    """3-layer validation for a factor expression.

    L1: In-sample Rank IC on full dataset
    L2: Rolling window IC on n_windows non-overlapping segments
    L3: IC direction consistency (all windows same sign)
        AND mean IC > 0.03
        AND stability (max_ic - min_ic) < 0.08

    Args:
        expr: expression string (named-column or gplearn X-index format)
        data: feature matrix DataFrame
        target_col: prediction target column name
        n_windows: number of rolling validation windows
        feature_cols: list of column names for gplearn X-index mapping

    Returns dict with all metrics and pass/fail verdict.
    """
    try:
        factor_vals = evaluate_expression(expr, data, feature_cols=feature_cols)
    except Exception as e:
        return {"passed": False, "error": str(e), "ic_in_sample": 0,
                "ic_windows": [], "ic_mean": 0, "ic_stability": 1.0}

    if data[target_col].dtype == object:
        target = data[target_col].astype(float).values
    else:
        target = data[target_col].values

    # L1: In-sample IC
    ic_full = spearman_ic(factor_vals, target)

    # L2: Rolling windows (non-overlapping, equal-sized)
    n = len(factor_vals)
    window_size = n // n_windows
    if window_size < 10:
        # Not enough data for windows — relax to L1 only
        passed = abs(ic_full) > 0.02
        return {
            "passed": passed, "ic_in_sample": round(ic_full, 4),
            "ic_windows": [], "ic_mean": 0, "ic_stability": 1.0,
            "note": "insufficient data for rolling windows, L1 only"
        }

    ic_windows = []
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n)
        ic_w = spearman_ic(factor_vals[start:end], target[start:end])
        ic_windows.append(ic_w)

    ic_mean = np.mean(ic_windows)
    ic_stability = max(ic_windows) - min(ic_windows)

    # L3: Consistency checks
    all_same_sign = all(w >= 0 for w in ic_windows) or all(w <= 0 for w in ic_windows)
    mean_ok = abs(ic_mean) > 0.03
    stable_ok = ic_stability < 0.08

    passed = all_same_sign and mean_ok and stable_ok

    return {
        "passed": passed,
        "ic_in_sample": round(ic_full, 4),
        "ic_windows": [round(w, 4) for w in ic_windows],
        "ic_mean": round(ic_mean, 4),
        "ic_stability": round(ic_stability, 4),
        "all_same_sign": all_same_sign,
        "mean_ok": mean_ok,
        "stable_ok": stable_ok,
    }


def deduplicate(factors, data, max_corr=0.7):
    """Remove redundant factors via hierarchical clustering on inter-factor IC.

    Args:
        factors: list of [{expression, ic, ...}]
        data: feature matrix DataFrame
        max_corr: maximum allowed correlation between factors (distance = 1 - |IC|)

    Returns:
        deduplicated list, highest IC factor retained per cluster
    """
    if len(factors) <= 1:
        return factors

    # Compute factor values for each
    factor_vals_list = []
    valid_indices = []
    for i, f in enumerate(factors):
        try:
            vals = evaluate_expression(f["expression"], data,
                                       feature_cols=f.get("feature_cols"))
            factor_vals_list.append(vals)
            valid_indices.append(i)
        except Exception:
            continue

    if len(factor_vals_list) <= 1:
        return [factors[i] for i in valid_indices]

    # Inter-factor correlation matrix
    m = len(factor_vals_list)
    dist_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            ic = abs(spearman_ic(factor_vals_list[i], factor_vals_list[j]))
            dist = 1.0 - ic
            dist_matrix[i][j] = dist_matrix[j][i] = dist

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    threshold = 1.0 - max_corr
    clusters = fcluster(Z, t=threshold, criterion="distance")

    # Keep best IC per cluster
    best_per_cluster = {}
    for idx, cluster_id in enumerate(clusters):
        fi = valid_indices[idx]
        ic_abs = abs(factors[fi]["ic"])
        if cluster_id not in best_per_cluster or ic_abs > abs(best_per_cluster[cluster_id]["ic"]):
            best_per_cluster[cluster_id] = factors[fi]

    result = sorted(best_per_cluster.values(), key=lambda f: abs(f["ic"]), reverse=True)
    removed = len(factors) - len(result)
    if removed > 0:
        print(f"[validator] Dedup removed {removed} redundant factors, retained {len(result)}")
    return result


# ── Self-check ──────────────────────────────────────────────
if __name__ == "__main__":
    from feature_matrix import build_daily_matrix
    from expression_tree import generate_expressions, FEATURE_COLUMNS

    mat = build_daily_matrix(["688017", "002896"], lookback_days=30)
    exprs = generate_expressions(FEATURE_COLUMNS, n=30)

    results = []
    for e in exprs[:15]:
        v = three_layer_validate(e, mat, "forward_return_1d")
        if v["ic_in_sample"] != 0:
            results.append({"expression": e, "ic": v["ic_in_sample"]})
        if v["passed"]:
            print(f"PASS: IC={v['ic_in_sample']:.4f} windows={v['ic_windows']} | {e[:60]}")

    print(f"\nTested {len(exprs[:15])} expressions, {len([r for r in results if abs(r['ic']) > 0.02])} with IC > 0.02")

    if results:
        deduped = deduplicate(results, mat)
        print(f"After dedup: {len(deduped)} factors retained")
