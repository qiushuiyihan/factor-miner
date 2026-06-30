"""Genetic miner -- gplearn symbolic regression wrapper for factor discovery.

Uses gplearn's SymbolicRegressor with custom fitness (Spearman Rank IC)
and a pre-seeded initial population from our expression tree engine.
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# Ensure src/ is on the path so we can import sibling modules regardless of cwd
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from expression_tree import (  # noqa: E402
    evaluate_expression,
    generate_expressions,
    FEATURE_COLUMNS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def spearman_ic(a, b):
    """Spearman Rank IC between factor values and forward target.

    Computes the rank correlation between two arrays, ignoring NaN/Inf values.
    Returns 0.0 if fewer than 10 valid sample pairs remain.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = ~(np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b))
    if mask.sum() < 10:
        return 0.0
    corr, _ = spearmanr(a[mask], b[mask])
    return corr if not np.isnan(corr) else 0.0


def run_evolution(
    data,
    target_col="forward_main_net_1d",
    n_generations=20,
    population_size=1000,
    tournament_size=20,
    parsimony_coefficient=0.001,
    random_state=42,
    stopping_criteria=0.01,
):
    """Run genetic programming to discover factor formulas.

    Args:
        data: feature matrix DataFrame (from feature_matrix.build_daily_matrix)
        target_col: column name for the prediction target
        n_generations: max evolution generations
        population_size: gplearn population size
        tournament_size: tournament selection size
        parsimony_coefficient: penalty for formula complexity
        random_state: random seed for reproducibility
        stopping_criteria: minimum fitness improvement to continue

    Returns:
        list of dicts: [{expression, ic, generation, length}, ...]
        sorted by abs(IC) descending
    """
    # Feature columns (exclude metadata and target)
    exclude = {"date", "code", target_col, "forward_main_net_5d"}
    feature_cols = [c for c in data.columns if c not in exclude]

    if not feature_cols:
        raise ValueError("No feature columns found in data")

    X = data[feature_cols].values
    y = data[target_col].values

    # Handle NaN/Inf in input
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)
    y = np.nan_to_num(y, nan=0.0)

    # Build function set (matches expression_tree.OPERATORS)
    function_set = ["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv"]

    # Generate seed expressions from our type-constrained engine
    # NOTE: These use our custom expression syntax (ts_mean, ts_std, etc.)
    # which gplearn cannot consume directly. For v1 we let gplearn start
    # from its own random population but constrained to our function_set.
    # The seed expressions are available for future warm-start integration.
    print(f"[miner] Generating seed expressions...")
    seed_exprs = generate_expressions(FEATURE_COLUMNS, n=min(population_size, 500))
    print(f"[miner] Generated {len(seed_exprs)} seed expressions")

    print(
        f"[miner] Starting evolution: pop={population_size}, "
        f"gens={n_generations}, features={X.shape[1]}"
    )
    print(f"[miner] Target: {target_col}")

    gp = SymbolicRegressor(
        population_size=population_size,
        generations=n_generations,
        tournament_size=tournament_size,
        stopping_criteria=stopping_criteria,
        const_range=(-1.0, 1.0),
        init_depth=(2, 5),
        init_method="half and half",
        function_set=function_set,
        metric="spearman",  # gplearn built-in Spearman correlation
        parsimony_coefficient=parsimony_coefficient,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        p_point_replace=0.05,
        max_samples=0.9,
        verbose=1,
        random_state=random_state,
        n_jobs=1,  # single-threaded to avoid Windows multiprocessing issues
    )

    gp.fit(X, y)

    # Extract top programs from the final generation
    results = []
    max_to_extract = min(len(gp._programs[-1]), 30)
    for i, program in enumerate(gp._programs[-1][:max_to_extract]):
        if program is None:
            continue
        factor_vals = program.execute(X)
        ic = spearman_ic(y, factor_vals)
        results.append(
            {
                "expression": str(program),
                "ic": round(ic, 4),
                "generation": n_generations,
                "length": program.length_,
            }
        )

    results.sort(key=lambda r: abs(r["ic"]), reverse=True)
    return results


# -- Self-check ---------------------------------------------------
if __name__ == "__main__":
    from feature_matrix import build_daily_matrix  # noqa: E402

    print("Building feature matrix for 2 stocks...")
    mat = build_daily_matrix(["688017", "002896"], lookback_days=30)
    print(f"Matrix: {mat.shape}")

    print("\nRunning evolution (5 generations, small pop)...")
    results = run_evolution(
        mat,
        target_col="forward_main_net_1d",
        n_generations=5,
        population_size=200,
        random_state=42,
    )

    print(f"\nTop 5 discovered factors:")
    for r in results[:5]:
        sign = "+" if r["ic"] > 0 else ""
        print(
            f"  IC={sign}{r['ic']:.4f} | len={r['length']} | "
            f"gen={r['generation']} | {r['expression'][:80]}"
        )
