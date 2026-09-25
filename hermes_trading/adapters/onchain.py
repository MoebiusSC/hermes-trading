"""On-chain context. Free: mempool.space fee pressure (BTC only). GLASSNODE_API_KEY: active addresses."""
from __future__ import annotations

import httpx

from ..config import env
from . import SCHEMA_VERSION, cached


@cached(300)
async def fetch(asset: str) -> dict:
    base = asset.split("/")[0].upper()
    key = env("GLASSNODE_API_KEY")
    async with httpx.AsyncClient(timeout=10) as client:
        if key:
            r = await client.get(
                "https://api.glassnode.com/v1/metrics/addresses/active_count",
                params={"a": base, "i": "24h", "api_key": key},
            )
            r.raise_for_status()
            rows = r.json()
            return {
                "schema_version": SCHEMA_VERSION,
                "source": "glassnode",
                "metrics": {"active_addresses": rows[-1]["v"] if rows else None},
            }
        if base != "BTC":
            return {"schema_version": SCHEMA_VERSION, "source": "none", "metrics": {}}
        r = await client.get("https://mempool.space/api/v1/fees/recommended")
        r.raise_for_status()
        fees = r.json()
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "mempool.space",
            "metrics": {
                "fastest_fee_sat_vb": fees.get("fastestFee"),
                "hour_fee_sat_vb": fees.get("hourFee"),
            },
        }
