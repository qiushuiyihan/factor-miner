"""Report generator — markdown report + JSON factor pool."""

import json
import os
from datetime import date as get_date
from pathlib import Path


def generate_report(factors, validations, llm_interpretation,
                    methodology, suggestions, output_dir):
    """Generate daily factor mining report.

    Args:
        factors: list of [{expression, ic, generation, length}]
        validations: list of [{passed, ic_in_sample, ic_windows, ...}]
        llm_interpretation: str from LLM factor interpretation
        methodology: str from LLM methodology search
        suggestions: str from LLM next-round suggestions
        output_dir: path to output/{YYYYMMDD}/

    Returns:
        str: path to generated report.md
    """
    today = get_date.today().strftime("%Y-%m-%d")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    passed = [(f, v) for f, v in zip(factors, validations) if v.get("passed")]
    failed = [(f, v) for f, v in zip(factors, validations) if not v.get("passed")]

    # ── Build markdown ──
    lines = []
    lines.append(f"# 因子挖掘日报 — {today}")
    lines.append("")
    lines.append("## 概况")
    lines.append("")
    lines.append(f"- 候选因子评估数：{len(factors)}")
    lines.append(f"- 通过三层验证：{len(passed)} 个")
    lines.append(f"- 未通过：{len(failed)} 个")
    lines.append(f"- 累计因子池：见 `found_factors.json`")
    lines.append("")

    if passed:
        lines.append("## ✅ 新发现因子")
        lines.append("")
        for i, (f, v) in enumerate(passed):
            lines.append(f"### ⠀#{i+1} | IC={f['ic']:.4f}")
            lines.append("")
            lines.append(f"- **表达式**: `{f['expression']}`")
            lines.append(f"- **复杂度**: {f.get('length', '?')} 节点")
            lines.append(f"- **样本内IC**: {v.get('ic_in_sample', '?')}")
            lines.append(f"- **滚动窗口IC**: {v.get('ic_windows', [])}")
            lines.append(f"- **IC均值**: {v.get('ic_mean', '?')}")
            lines.append(f"- **稳定性**: {v.get('ic_stability', '?')}")
            lines.append("")

    if passed and llm_interpretation and not llm_interpretation.startswith("[LLM ERROR]"):
        lines.append("## 🧠 LLM 因子解读")
        lines.append("")
        lines.append(llm_interpretation)
        lines.append("")

    if methodology and not methodology.startswith("[LLM ERROR]"):
        lines.append("## 📚 方法论参考")
        lines.append("")
        lines.append(methodology)
        lines.append("")

    if suggestions and not suggestions.startswith("[LLM ERROR]"):
        lines.append("## 🔄 下一轮建议")
        lines.append("")
        lines.append(suggestions)
        lines.append("")

    if failed:
        lines.append("## ❌ 未通过验证")
        lines.append("")
        lines.append(f"共 {len(failed)} 个因子未通过三层验证。")
        lines.append("")
        lines.append("| 表达式 | IC | 原因 |")
        lines.append("|--------|-----|------|")
        for f, v in failed[:10]:
            reason_parts = []
            if not v.get("all_same_sign", True):
                reason_parts.append("IC方向不一致")
            if not v.get("mean_ok", True):
                reason_parts.append("均值IC不达标")
            if not v.get("stable_ok", True):
                reason_parts.append("IC不够稳定")
            reason = ", ".join(reason_parts) if reason_parts else v.get("error", "未知")
            lines.append(f"| `{f['expression'][:60]}` | {f['ic']:.4f} | {reason} |")
        lines.append("")

    report_path = out / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ── Save factor pool JSON ──
    pool = []
    for f, v in passed:
        pool.append({
            "expression": f["expression"],
            "ic": f["ic"],
            "ic_windows": v.get("ic_windows", []),
            "ic_mean": v.get("ic_mean", 0),
            "ic_stability": v.get("ic_stability", 1),
            "discovery_date": today,
            "length": f.get("length", 0),
        })

    pool_path = out / "found_factors.json"
    pool_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[report] Wrote {report_path} ({len(passed)} factors retained)")
    return str(report_path)


# ── Self-check ──────────────────────────────────────────────
if __name__ == "__main__":
    dummy_factors = [
        {"expression": "div(sub(main_net, main_net_ma5), main_net_std5)", "ic": 0.045, "length": 6},
        {"expression": "rank(super_main_ratio)", "ic": 0.012, "length": 3},
    ]
    dummy_vals = [
        {"passed": True, "ic_in_sample": 0.045, "ic_windows": [0.04, 0.05, 0.04], "ic_mean": 0.043, "ic_stability": 0.01},
        {"passed": False, "ic_in_sample": 0.012, "ic_windows": [0.01, -0.005, 0.02], "ic_mean": 0.008, "ic_stability": 0.03, "all_same_sign": False, "mean_ok": False, "stable_ok": True},
    ]

    path = generate_report(
        dummy_factors, dummy_vals,
        llm_interpretation="样本解读：(dummy)",
        methodology="样本方法论：(dummy)",
        suggestions="样本建议：(dummy)",
        output_dir="C:/Users/33480/Desktop/claude-workspace/factor-miner/output/test"
    )
    print(f"Report: {path}")
