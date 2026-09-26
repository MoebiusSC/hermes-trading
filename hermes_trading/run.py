"""Entrypoint: `python -m hermes_trading.run [--asset BTC/USDT] [--once]`."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys

from . import config, state_server
from .adapters import SchemaError
from .loop import Worker
from .storage import load_yaml


def _run(coro) -> None:
    # ccxt/aiohttp are happier on the selector loop than Windows' default proactor loop
    if sys.platform == "win32" and sys.version_info >= (3, 12):
        asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
        return
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(coro)


def seed_state() -> None:
    """Copy any missing state files from the image's seed dir (never overwrites)."""
    if not config.STATE_SEED.is_dir():
        return
    for src in config.STATE_SEED.rglob("*"):
        dest = config.STATE / src.relative_to(config.STATE_SEED)
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def _legacy_asset(default: str) -> str:
    """Which asset the single-asset files belong to, read from its own trade log if possible."""
    trades = config.STATE / "trades.jsonl"
    if trades.exists():
        for line in trades.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return json.loads(line).get("asset", default)
    return default


def migrate_legacy_layout(default_asset: str) -> None:
    """The single-asset version kept its files at the state root; move them to assets/<asset>/."""
    legacy = [config.STATE / name for name in config.LEGACY_FILES if (config.STATE / name).exists()]
    legacy_history = config.STATE / "history"
    if not legacy and not legacy_history.is_dir():
        return
    dest = config.asset_paths(_legacy_asset(default_asset))
    if dest.strategy.exists():
        print(f"Legacy files at {config.STATE} left untouched: {dest.root} already has a strategy.")
        return
    dest.history.mkdir(parents=True, exist_ok=True)
    for src in legacy:
        shutil.move(src, dest.root / src.name)
    if legacy_history.is_dir():
        for f in legacy_history.iterdir():
            shutil.move(f, dest.history / f.name)
        legacy_history.rmdir()
    print(f"Migrated single-asset state into {dest.root}")


def main(argv: list[str] | None = None) -> None:
    config.load_env()
    parser = argparse.ArgumentParser(description="hermes-trading paper worker")
    parser.add_argument("--asset", help="trade only this ccxt symbol instead of goal.yaml's assets")
    parser.add_argument("--once", action="store_true", help="run a single tick and exit")
    args = parser.parse_args(argv)

    mode = config.env("HERMES_TRADING_MODE", "paper").lower()
    if mode != "paper":
        accepted = config.env("HERMES_TRADING_I_ACCEPT_RISK", "false").lower() == "true"
        if mode == "live" and accepted:
            sys.exit("Live execution is not implemented in this build. Set HERMES_TRADING_MODE=paper.")
        sys.exit(f"Refusing to start: HERMES_TRADING_MODE={mode!r}. Only 'paper' is supported.")

    seed_state()
    goal = load_yaml(config.GOAL_FILE)
    assets = [args.asset] if args.asset else config.goal_assets(goal)
    migrate_legacy_layout(assets[0])
    state_server.start()
    try:
        worker = Worker(assets, goal)
        state_server.attach(worker)
        _run(worker.run(once=args.once))
    except KeyboardInterrupt:
        pass
    except SchemaError as e:
        sys.exit(f"HALTED — adapter schema mismatch: {e}")


if __name__ == "__main__":
    main()
