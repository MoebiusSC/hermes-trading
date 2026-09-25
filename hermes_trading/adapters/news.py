"""Sentiment. Free: alternative.me Fear & Greed index. NEWS_API_KEY adds newsapi.org headlines."""
from __future__ import annotations

import httpx

from ..config import env
from . import SCHEMA_VERSION, cached


@cached(300)
async def fetch(asset: str) -> dict:
    base = asset.split("/")[0].upper()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get("https://api.alternative.me/fng/", params={"limit": 1})
        r.raise_for_status()
        fng = r.json()["data"][0]
        out = {
            "schema_version": SCHEMA_VERSION,
            "source": "alternative.me",
            "fear_greed": int(fng["value"]),
            "fear_greed_label": fng["value_classification"],
            "headlines": [],
        }
        key = env("NEWS_API_KEY")
        if key:
            r = await client.get(
                "https://newsapi.org/v2/everything",
                params={"q": base, "pageSize": 5, "sortBy": "publishedAt", "language": "en"},
                headers={"X-Api-Key": key},
            )
            r.raise_for_status()
            out["headlines"] = [a["title"] for a in r.json().get("articles", [])]
            out["source"] += "+newsapi"
    return out
