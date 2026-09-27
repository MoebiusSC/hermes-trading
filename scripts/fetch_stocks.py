"""Download split- and dividend-adjusted stock bars (Alpaca SIP feed) for scripts/research_stocks.py.

  railway run uv run python scripts/fetch_stocks.py            (needs ALPACA_API_KEY / ALPACA_API_SECRET)

Daily bars from 2015 for every stock; 15-minute bars from mid-2021 for the 2021 universe, from which
1h and 4h bars are built (Alpaca pages intraday history in ~2-week slices: 15m alone is ~150 requests
per stock). Cached in cache/stocks/<SYM>_<tf>.json as {t, open, high, low, close}. The free plan
serves SIP history up to 15 minutes ago and 200 requests a minute.
"""
import datetime as dt
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "cache" / "stocks"
UNIVERSE_2016 = ["AAPL", "GOOGL", "MSFT", "XOM", "AMZN", "META", "BRK.B", "JNJ", "GE", "WFC", "T", "JPM", "PG", "WMT",
                 "CVX", "VZ", "PFE", "KO", "V", "ORCL", "HD", "MRK", "PM", "INTC", "CMCSA"]
UNIVERSE_2021 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "NVDA", "JPM", "JNJ", "V", "UNH", "WMT", "HD",
                 "PG", "MA", "BAC", "DIS", "ADBE", "PYPL", "NFLX", "CRM", "XOM", "CMCSA", "PFE", "AMD"]
REFERENCE = ["SPY", "QQQ"]
ETFS = ["GLD", "TLT", "USO", "VOO", "SECZ"]  # the ETFs the bot trades besides SPY and QQQ
TFS = {"1d": ("1Day", "2015-01-01"), "15m": ("15Min", "2021-06-01")}
RESAMPLED = {"1h": 3600000, "4h": 4 * 3600000}
_lock, _sent = threading.Lock(), []


def _throttle(per_min: int = 180) -> None:
    while True:
        with _lock:
            now = time.time()
            _sent[:] = [t for t in _sent if now - t < 60]
            if len(_sent) < per_min:
                _sent.append(now)
                return
        time.sleep(0.5)


def fetch(symbol: str, tf: str) -> dict:
    timeframe, start = TFS[tf]
    end = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)).isoformat()
    headers = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET"]}
    rows, token = [], None
    while True:
        params = {"timeframe": timeframe, "start": start + "T00:00:00Z", "end": end, "limit": 10000, "feed": "sip",
                  "adjustment": "all"}
        if token:
            params["page_token"] = token
        for attempt in range(5):
            _throttle()
            r = httpx.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars", params=params, headers=headers, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        data = r.json()
        rows += [[int(dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp() * 1000), b["o"], b["h"], b["l"], b["c"]]
                 for b in data.get("bars") or []]
        token = data.get("next_page_token")
        if not token:
            break
    return {"t": [r[0] for r in rows], "open": [r[1] for r in rows], "high": [r[2] for r in rows],
            "low": [r[3] for r in rows], "close": [r[4] for r in rows]}


def resample(d: dict, ms: int) -> dict:
    out = {k: [] for k in ("t", "open", "high", "low", "close")}
    for i, t in enumerate(d["t"]):
        start = t - t % ms
        if out["t"] and out["t"][-1] == start:
            out["high"][-1] = max(out["high"][-1], d["high"][i])
            out["low"][-1] = min(out["low"][-1], d["low"][i])
            out["close"][-1] = d["close"][i]
        else:
            for k, v in (("t", start), ("open", d["open"][i]), ("high", d["high"][i]), ("low", d["low"][i]), ("close", d["close"][i])):
                out[k].append(v)
    return out


def job(symbol: str, tf: str) -> None:
    path = OUT / f"{symbol}_{tf}.json"
    if not path.exists():
        data = fetch(symbol, tf)
        path.write_text(json.dumps(data))
        first = dt.datetime.fromtimestamp(data["t"][0] / 1000, dt.timezone.utc).date() if data["t"] else None
        print(symbol, tf, len(data["t"]), first, file=sys.stderr, flush=True)
    if tf == "15m":
        data = json.loads(path.read_text())
        for name, ms in RESAMPLED.items():
            (OUT / f"{symbol}_{name}.json").write_text(json.dumps(resample(data, ms)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, "1d") for s in sorted(set(UNIVERSE_2016 + UNIVERSE_2021 + REFERENCE + ETFS))]
    jobs += [(s, "15m") for s in sorted(set(UNIVERSE_2021 + REFERENCE + ETFS))]
    with ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(job, s, tf) for s, tf in jobs]:
            f.result()


if __name__ == "__main__":
    main()
