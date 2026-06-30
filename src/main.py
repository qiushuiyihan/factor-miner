"""Factor Miner — automated factor discovery pipeline.

Usage:
    D:/conda/envs_dirs/1/python.exe src/main.py

Runs the full pipeline and writes a report to output/{YYYYMMDD}/.
"""

import sys
import os
from datetime import date as get_date
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_data import STOCK_POOL
from feature_matrix import build_daily_matrix
from expression_tree import generate_expressions, FEATURE_COLUMNS, evaluate_expression
from genetic_miner import run_evolution, spearman_ic
from validator import three_layer_validate, deduplicate
from llm_layer import search_methodology, interpret_factors, suggest_next_round
from report import generate_report


def main():
    today = get_date.today().strftime("%Y-%m-%d")
    output_dir = Path(__file__).resolve().parent.parent / "output" / today
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = STOCK_POOL
    print(f"{'='*60}")
    print(f"Factor Miner — {today}")
    print(f"Stock pool: {len(codes)} stocks")
    print(f"{'='*60}\n")

    # ── Step 1: Data ────────────────────────────────────────
    print("[1/6] Fetching daily fund flow data...")
    try:
        data = build_daily_matrix(codes, lookback_days=60)
        print(f"      Feature matrix: {data.shape[0]} rows × {data.shape[1]} cols")
        print(f"      Stocks with data: {data['code'].nunique()}/{len(codes)}")
    except Exception as e:
        print(f"[FATAL] Data fetch failed: {e}")
        return 1

    if data.shape[0] < 100:
        print(f"[FATAL] Insufficient data: {data.shape[0]} rows. Need at least 100.")
        return 1

    # ── Step 2: LLM methodology search ──
    print("\n[2/6] Searching for factor mining methodologies (LLM)...")
    methodology = search_methodology()
    print(f"      {'Got methodology notes' if not methodology.startswith('[LLM ERROR]') else 'LLM unavailable'}")

    # ── Step 3: Mine ────────────────────────────────────────
    print("\n[3/6] Running genetic programming miner...")
    results = run_evolution(
        data,
        target_col="forward_main_net_1d",
        n_generations=20,
        population_size=1000,
        random_state=42,
    )
    print(f"      Evolution complete. Top IC: {results[0]['ic']:.4f}" if results else "      No results")

    if not results:
        print("[FATAL] Evolution returned no results")
        return 1

    # ── Step 4: Validate ────────────────────────────────────
    print("\n[4/6] Running 3-layer validation...")
    factors = []
    validated = []
    for r in results[:50]:  # validate top 50
        v = three_layer_validate(r["expression"], data, "forward_main_net_1d")
        if v.get("ic_in_sample", 0) != 0:
            factors.append(r)
            validated.append(v)

    passed_count = sum(1 for v in validated if v.get("passed"))
    print(f"      {passed_count}/{len(validated)} passed 3-layer validation")

    # ── Step 5: Dedup ───────────────────────────────────────
    print("\n[5/6] Deduplicating factors...")
    if passed_count > 0:
        passed_factors = [f for f, v in zip(factors, validated) if v.get("passed")]
        passed_vals = [v for f, v in zip(factors, validated) if v.get("passed")]
        deduped = deduplicate(passed_factors, data)
        deduped_vals = []
        for d in deduped:
            # Find matching validation
            for f, v in zip(passed_factors, passed_vals):
                if f["expression"] == d["expression"]:
                    deduped_vals.append(v)
                    break
            else:
                deduped_vals.append({"passed": True})
        print(f"      {len(deduped)} factors retained after dedup")
    else:
        deduped = []
        deduped_vals = []
        print("      No factors passed validation, nothing to dedup")

    # ── Step 6: LLM interpretation + report ─────────────────
    print("\n[6/6] LLM interpretation & report generation...")

    llm_interpretation = ""
    suggestions = ""
    if deduped:
        llm_interpretation = interpret_factors(deduped, deduped_vals)
        suggestions = suggest_next_round(deduped, methodology)
    elif factors:
        # Use top factors even if they didn't pass validation
        llm_interpretation = interpret_factors(factors[:5], validated[:5])
        suggestions = suggest_next_round(factors[:5], methodology)

    # Always include all factors (passed + failed) in report for transparency
    report_path = generate_report(
        factors, validated,
        llm_interpretation, methodology, suggestions,
        str(output_dir)
    )

    print(f"\n{'='*60}")
    print(f"DONE. Report: {report_path}")
    print(f"Passed factors: {len(deduped)}")
    print(f"Next: review report, run again tomorrow for iteration")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
