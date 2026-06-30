"""LLM layer — DeepSeek integration for methodology search, factor interpretation,
and iteration guidance."""

import os
from pathlib import Path
from openai import OpenAI


def _get_client():
    """Initialize OpenAI-compatible client pointed at DeepSeek."""
    env_path = Path.home() / ".vibe-trading" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY=") or line.startswith("VIBE_TRADING_DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault("DEEPSEEK_API_KEY", key)
                    break

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    return OpenAI(api_key=api_key, base_url=base_url), "deepseek-v4-pro"


def _chat(system_prompt, user_prompt, temperature=0.3, max_tokens=2000):
    """Send a single-turn chat to DeepSeek and return the response text."""
    client, model = _get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM ERROR] {e}"


def search_methodology():
    """Pre-mining: search for factor mining methodologies.

    Queries DeepSeek (using its training data) for structured methodology notes
    covering WorldQuant flow-factor alpha patterns, A-share-specific flow
    anomalies, gplearn symbolic regression tips, and practical operator
    constraints that can feed into expression template generation.

    Returns:
        str: Structured methodology notes in Chinese.
    """
    system = """你是一个量化因子研究员。你的任务是总结当前业界最有效的资金流因子挖掘方法论。

请搜索并总结（直接从你的训练数据中提取）：
1. WorldQuant alpha 中与资金流相关的 alpha 构造逻辑
2. 分钟级资金流（主力/散户/大单）的有效组合方式
3. A股特有的资金流模式（跌停板效应、北向联动、尾盘异动等）
4. gplearn 符号回归在因子挖掘中的应用经验

每条方法论请包含：
- 核心思想（一句话）
- 公式逻辑（用自然语言描述，比如"主力连续净流入天数与未来收益正相关"）
- 期望的 IC 范围（A股实际情况，不吹牛）
- 可以翻译成什么样的算子约束

用中文回答，简洁但具体。不要泛泛而谈，每条都要有可操作的公式逻辑。"""

    return _chat(system, "开始搜索资金流因子方法论。请给出具体的、可操作的发现。")


def interpret_factors(factors, validation_results):
    """Post-mining: interpret discovered factors with economic logic.

    Args:
        factors: list of dicts, each with at least {expression, ic, ...}.
        validation_results: list of dicts, each with at least
            {passed, ic_windows, ic_mean, ic_stability, ...}.  Must have
            the same length as *factors*.

    Returns:
        str: LLM interpretation text (Chinese).
    """
    factor_text = ""
    n = min(len(factors), len(validation_results))
    for i in range(n):
        f = factors[i]
        v = validation_results[i]
        factor_text += f"""
因子 #{i+1}:
  表达式: {f['expression']}
  样本内IC: {f['ic']}
  滚动窗口IC: {v.get('ic_windows', [])}
  均值IC: {v.get('ic_mean', 'N/A')}
  稳定性: {v.get('ic_stability', 'N/A')}
  通过: {'是' if v.get('passed') else '否'}
"""

    system = """你是一个量化因子分析师。你的任务是对机器发现的因子进行经济学解读。

对于每个因子，分析：
1. 它的经济学逻辑——为什么这个公式可能捕捉到真实的市场行为？
2. 是市场微观结构效应还是真正的 alpha？
3. 可能存在的风险：数据泄露、幸存者偏差、过拟合、市场体制切换？
4. 是否值得保留？给出你的判断和理由。

用中文回答。对每个因子逐一分析，不要跳过。结构清晰但不要写论文。"""

    return _chat(system, f"以下是本轮挖掘出的因子池，请逐一解读：\n{factor_text}")


def suggest_next_round(current_factors, methodology=""):
    """Post-interpretation: suggest parameter / operator changes for the next
    round of genetic mining.

    Args:
        current_factors: list of dicts, each with at least {ic, expression, ...}.
        methodology: optional methodology notes from search_methodology().

    Returns:
        str: LLM suggestions text (Chinese).
    """
    factor_summary = "\n".join([
        f"- IC={f['ic']:.4f} | {f['expression'][:100]}"
        for f in current_factors[:10]
    ])

    system = """你是一个量化因子挖掘的迭代教练。基于本轮结果，建议下一轮改进方向。

请给出具体建议：
1. 应该新增什么算子或变换？为什么？
2. 哪些参数范围需要调整（窗口大小、进化代数、种群规模）？
3. 是否有特定的数据维度被忽略了（比如时间分段、截面归一化）？
4. 建议下一轮优先探索的方向。

用中文，每条建议必须有理由，不要空泛。"""

    return _chat(system, f"""本轮方法论:
{methodology[:500] if methodology else '无'}

本轮发现的有效因子:
{factor_summary if factor_summary else '未发现通过验证的因子'}

请给出下一轮的具体改进建议。""")


# ── Self-check ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Methodology Search ===")
    method = search_methodology()
    print(method[:500])
    print("\n=== Factor Interpretation (dummy) ===")
    dummy_factors = [
        {"expression": "div(sub(main_net, main_net_ma5), main_net_std5)", "ic": 0.045}
    ]
    dummy_val = [
        {
            "passed": True,
            "ic_windows": [0.04, 0.05, 0.04],
            "ic_mean": 0.043,
            "ic_stability": 0.01,
        }
    ]
    interp = interpret_factors(dummy_factors, dummy_val)
    print(interp[:500])
    print("\n=== Next-Round Suggestions (dummy) ===")
    sug = suggest_next_round(dummy_factors, methodology="")
    print(sug[:500])
    print("\nDone.")
