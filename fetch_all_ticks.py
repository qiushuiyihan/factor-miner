#!/usr/bin/env python3
"""Fetch all A-share tick data from 迈锐 API and upload to cloud.

Usage:
    python fetch_all_ticks.py              # all stocks
    python fetch_all_ticks.py --limit 50   # first 50 stocks (for testing)

Output:
    output/tick-data-{YYYYMMDD}/
    ├── 日报_{YYYYMMDD}.xlsx
    └── parquet/*.parquet
"""

import os
import sys
import json
import time
import argparse
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# ── Config ───────────────────────────────────────────────────
LICENCE = "7DFC8EA6-8F7A-4765-B37D-356EA3F58829"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
MAX_WORKERS = 8
RETRY = 3
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# ── Stock list ───────────────────────────────────────────────
STOCK_LIST_URL = (
    "https://push2.eastmoney.com/api/qt/clist/get"
    "?pn=1&pz=6000&po=1&np=1&fltt=2&invt=2&fid=f3"
    "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
    "&fields=f12,f14"
)
STOCK_LIST_FILE = Path(__file__).resolve().parent / "data" / "stock_list.json"


def get_stock_list(force_refresh=False):
    """Get full A-share stock code list from cached file. Refresh from API on demand."""
    STOCK_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)

    if force_refresh:
        # Try multiple sources
        stocks = _fetch_from_akshare() or _fetch_from_eastmoney()
        if stocks:
            with open(STOCK_LIST_FILE, "w") as f:
                json.dump(stocks, f, ensure_ascii=False)
            print(f"  Refreshed: {len(stocks)} stocks from API")
            return stocks

    if STOCK_LIST_FILE.exists():
        with open(STOCK_LIST_FILE) as f:
            stocks = json.load(f)
        print(f"  Loaded {len(stocks)} stocks from cache")
        return stocks

    # First run, no cache — must fetch
    stocks = _fetch_from_akshare() or _fetch_from_eastmoney()
    if stocks:
        with open(STOCK_LIST_FILE, "w") as f:
            json.dump(stocks, f, ensure_ascii=False)
        print(f"  Fetched {len(stocks)} stocks from API")
        return stocks
    raise RuntimeError("Cannot get stock list — no cache and all APIs failed")


def _fetch_from_akshare():
    """Try akshare for stock list."""
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        return [{"code": row["code"], "name": row["name"]} for _, row in df.iterrows()]
    except Exception as e:
        print(f"  [WARN] akshare failed: {e}")
        return None


def _fetch_from_eastmoney():
    """Try eastmoney for stock list."""
    try:
        r = requests.get(STOCK_LIST_URL, headers={"User-Agent": UA}, timeout=15)
        data = r.json()["data"]["diff"]
        return [
            {"code": d["f12"], "name": d.get("f14", "")}
            for d in data
            if d["f12"].isdigit() and len(d["f12"]) == 6
        ]
    except Exception as e:
        print(f"  [WARN] eastmoney stock list failed: {e}")
        return None


def fetch_one_tick(code):
    """Fetch tick data for one stock. Returns list of dicts or None."""
    url = f"http://api.mairuiapi.com/hsrl/zbjy/{code}/{LICENCE}"
    for attempt in range(RETRY):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0 and "d" in data[0]:
                    return {"code": code, "ticks": data, "n": len(data)}
            return None
        except Exception as e:
            if attempt < RETRY - 1:
                time.sleep(1)
            else:
                return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int,
                        default=int(os.environ.get("TICK_FETCH_LIMIT", "0")),
                        help="Limit N stocks (0=all)")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("TICK_FETCH_WORKERS", str(MAX_WORKERS))))
    parser.add_argument("--refresh-stocks", action="store_true")
    args = parser.parse_args()

    today_str = date.today().strftime("%Y%m%d")
    out_dir = OUTPUT_ROOT / f"tick-data-{today_str}"
    pq_dir = out_dir / "parquet"
    pq_dir.mkdir(parents=True, exist_ok=True)

    # ── Get stock list ──
    stocks = get_stock_list(force_refresh=args.refresh_stocks)
    if args.limit:
        stocks = stocks[: args.limit]

    print(f"{'='*60}")
    print(f"A-Share Tick Fetcher — {today_str}")
    print(f"Stocks: {len(stocks)}  Workers: {args.workers}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    # ── Fetch in parallel ──
    t0 = time.time()
    done = 0
    failed = 0

    summary_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one_tick, s["code"]): s for s in stocks
        }
        for f in as_completed(futures):
            stock = futures[f]
            try:
                result = f.result()
            except Exception as e:
                failed += 1
                done += 1
                continue

            if result is None:
                failed += 1
            else:
                # Save parquet
                df = pd.DataFrame(result["ticks"])
                df.to_parquet(pq_dir / f'{stock["code"]}.parquet', index=False)

                buys = (df["ts"].astype(int) == 1).sum()
                sells = (df["ts"].astype(int) == 2).sum()
                avg_price = df["p"].astype(float).mean()
                summary_rows.append({
                    "代码": stock["code"],
                    "名称": stock["name"],
                    "逐笔数": result["n"],
                    "买入笔": buys,
                    "卖出笔": sells,
                    "净买卖比": round((buys - sells) / (buys + sells + 1), 4),
                    "均价": round(avg_price, 2),
                })

            done += 1
            if done % 100 == 0 or done == len(stocks):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(stocks) - done) if done else 0
                print(f"  [{done}/{len(stocks)}] OK={done-failed} Fail={failed} "
                      f"Elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    # ── Summary Excel ──
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary = summary.sort_values("逐笔数", ascending=False)
        summary.to_excel(out_dir / f"日报_{today_str}.xlsx", index=False)
        print(f"\nDone. {len(summary_rows)} stocks / {done} total / {elapsed:.0f}s")
        print(f"  Summary: {out_dir / f'日报_{today_str}.xlsx'}")
        print(f"  Parquet: {pq_dir}/ ({len(summary_rows)} files)")
    else:
        print(f"\n[FATAL] No data fetched. All {done} stocks failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
