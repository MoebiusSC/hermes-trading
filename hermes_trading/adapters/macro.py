"""Macro backdrop via yfinance (free): dollar index, VIX, S&P 500 daily closes.
The same for every asset, so it is fetched once per refresh and shared (yfinance isn't thread-safe)."""
from __future__ import annotations

import asyncio

import yfinance as yf

from . import SCHEMA_VERSION, cached

TICKERS = {"dxy": "DX-Y.NYB", "vix": "^VIX", "spx": "^GSPC"}


def _last_closes() -> dict:
    values = {}
    for name, ticker in TICKERS.items():
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        values[name] = float(hist["Close"].iloc[-1]) if not hist.empty else None
    return values


@cached(900)
async def _shared() -> dict:
    values = await asyncio.to_thread(_last_closes)
    if all(v is None for v in values.values()):
        raise RuntimeError("yfinance returned no macro data")
    return {"schema_version": SCHEMA_VERSION, "source": "yfinance", "values": values}


async def fetch(asset: str) -> dict:
    return await _shared()
