"""Download long candle history (and perpetual funding rates) into cache/ for research backtests.

  python scripts/fetch_history.py [--days 1100]

Candles go through backtest.history() (cache/candles/); funding rates of the Binance USDT-M
perpetuals go to cache/funding/<ASSET>.json as [[ms, rate], ...].
"""
import argparse
import json
import sys
import time

import ccxt

from hermes_trading import backtest, config
from hermes_trading.storage import atomic_write, load_yaml


def funding(asset: str, days: float) -> int:
    client = ccxt.binanceusdm({"enableRateLimit": True})
    symbol = asset + ":USDT"
    since, rows = int((time.time() - days * 86400) * 1000), []
    while True:
        page = client.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not page:
            break
        rows += [[int(r["timestamp"]), float(r["fundingRate"])] for r in page]
        since = page[-1]["timestamp"] + 1
        if len(page) < 1000:
            break
    path = config.ROOT / "cache" / "funding" / f"{config.asset_slug(asset)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(rows))
    return len(rows)


def main() -> None:
    config.load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=1100)
    args = ap.parse_args()
    goal = load_yaml(config.GOAL_FILE)
    for asset in config.goal_assets(goal):
        tfs = ("1h", "4h", "1d")
        for tf in tfs:
            try:
                h = backtest.history(asset, tf, args.days)
                print(asset, tf, len(h["t"]), h.get("source"), file=sys.stderr, flush=True)
            except Exception as e:
                print(asset, tf, "FAILED", type(e).__name__, str(e)[:100], file=sys.stderr, flush=True)
        if not config.is_stock(asset):
            try:
                print(asset, "funding", funding(asset, args.days), file=sys.stderr, flush=True)
            except Exception as e:
                print(asset, "funding FAILED", type(e).__name__, str(e)[:100], file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
