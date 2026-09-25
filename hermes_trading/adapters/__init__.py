"""Data adapters. Each module exposes `async def fetch(asset) -> dict` carrying `schema_version`."""
from __future__ import annotations

import asyncio
import functools
import time

SCHEMA_VERSION = "1"


class SchemaError(RuntimeError):
    """An adapter returned data in a shape the loop doesn't understand. Halts the loop."""


def check_schema(name: str, payload: dict, required: tuple[str, ...]) -> None:
    if not isinstance(payload, dict):
        raise SchemaError(f"{name}: expected dict, got {type(payload).__name__}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"{name}: schema_version {payload.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    missing = [k for k in required if k not in payload]
    if missing:
        raise SchemaError(f"{name}: missing fields {missing}")


def cached(ttl_s: float):
    """Cache an async fetch per-argument for ttl_s seconds (slow-moving data, rate limits).
    Concurrent callers with the same arguments share one in-flight fetch."""

    def deco(fn):
        store: dict = {}
        locks: dict = {}

        @functools.wraps(fn)
        async def wrapper(*args):
            lock = locks.setdefault(args, asyncio.Lock())
            async with lock:
                hit = store.get(args)
                if hit and time.monotonic() - hit[0] < ttl_s:
                    return hit[1]
                value = await fn(*args)
                store[args] = (time.monotonic(), value)
                return value

        return wrapper

    return deco
