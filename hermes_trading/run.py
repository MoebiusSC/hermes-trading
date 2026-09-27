"""Entrypoint: `python -m hermes_trading.run [--asset BTC/USDT] [--once]`."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys

from . import config, reflect, state_server
from .adapters import SchemaError
from .loop import Worker
from .storage import dump_yaml, load_yaml


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


def migrate_strategies(goal: dict, assets: list[str]) -> None:
    """Apply goal.yaml `strategy_defaults` to every strategy that predates them, once, as a
    recorded change (mode "migration"): the new fields plus any listed value the reflection
    hasn't already moved. The template for new assets gets them too."""
    defaults = goal.get("strategy_defaults") or {}
    if not defaults:
        return
    marker = next(iter(defaults))  # a strategy that has the first default key was already migrated
    note = str(goal.get("strategy_defaults_note") or "Nuevos valores por defecto elegidos por backtest.")
    for asset in assets:
        paths = config.asset_paths(asset)
        if config.sleeve(asset) or not paths.strategy.exists():
            continue
        strategy = load_yaml(paths.strategy)
        try:
            reflect.get_path(strategy, marker)
            continue
        except KeyError:
            pass
        changes = {k: v for k, v in defaults.items() if not (config.is_stock(asset) and k == "entry.direction")}
        records = reflect.apply_manual(paths, changes, config.is_stock(asset), mode="migration", rationale=note)
        print(f"Migrated {asset}: " + ", ".join(f"{r['variable']} {r['old_value']} → {r['new_value']}" for r in records), flush=True)
    template = load_yaml(config.STRATEGY_TEMPLATE)
    for key, value in defaults.items():
        reflect._set_path_creating(template, key, value)
    dump_yaml(config.STRATEGY_TEMPLATE, template)


def apply_migrations(goal: dict, assets: list[str]) -> None:
    """goal.yaml `strategy_migrations`: named, one-off changes for the assets of the listed kinds,
    each recorded as a "migration" change. A strategy lists the migrations it received, so a
    migration runs once per asset even across restarts. A migration applies to the pairs' main
    strategies, or with `sleeve: <name>` to that sleeve's strategies only. A crypto migration of main
    strategies also writes the crypto template, so pairs added later start from it."""
    for migration in goal.get("strategy_migrations") or []:
        mid, changes = str(migration["id"]), dict(migration["changes"])
        kinds = set(migration.get("kinds") or ("crypto", "stock"))
        target_sleeve = migration.get("sleeve")
        note = str(migration.get("note") or "Cambio de estrategia elegido por backtest.")
        for asset in assets:
            stock = config.is_stock(asset)
            paths = config.asset_paths(asset)
            if ("stock" if stock else "crypto") not in kinds or not paths.strategy.exists():
                continue
            if config.sleeve(asset) != target_sleeve:
                continue
            if mid in (load_yaml(paths.strategy).get("migrations") or []):
                continue
            records = reflect.apply_manual(paths, changes, stock, mode="migration", rationale=note)
            strategy = load_yaml(paths.strategy)
            strategy["migrations"] = [*(strategy.get("migrations") or []), mid]
            dump_yaml(paths.strategy, strategy)
            print(f"Migration {mid} → {asset}: " + (", ".join(f"{r['variable']} {r['old_value']} → {r['new_value']}" for r in records) or "nothing to change"), flush=True)
        if "crypto" in kinds and not target_sleeve:
            template = load_yaml(config.STRATEGY_TEMPLATE)
            for key, value in changes.items():
                reflect._set_path_creating(template, key, value)
            template["migrations"] = [*(template.get("migrations") or []), mid]
            dump_yaml(config.STRATEGY_TEMPLATE_CRYPTO, template)


def ensure_sleeves(goal: dict) -> dict:
    """goal.yaml `sleeves`: extra strategies run next to each pair's main one, each in its own
    account ("BTC/USDT@momentum", same market). Adds any missing sleeve to goal.yaml's assets and
    starts its strategy from the template plus the sleeve's settings (recorded as a migration).
    Returns the goal re-read with the sleeves listed."""
    sleeves = goal.get("sleeves") or {}
    if not sleeves:
        return goal
    for name, spec in sleeves.items():
        kinds = set(spec.get("kinds") or ("crypto",))
        changes = dict(spec.get("strategy") or {})
        note = str(spec.get("note") or f"Sub-cuenta '{name}' creada.")
        for base in config.goal_assets(load_yaml(config.GOAL_FILE)):
            if config.sleeve(base) or ("stock" if config.is_stock(base) else "crypto") not in kinds:
                continue
            sid = f"{base}{config.SLEEVE_SEP}{name}"
            if sid not in config.goal_assets(load_yaml(config.GOAL_FILE)):
                config.add_goal_asset(config.GOAL_FILE, sid)
            paths = config.asset_paths(sid)
            if paths.strategy.exists():
                continue
            paths.history.mkdir(parents=True, exist_ok=True)
            dump_yaml(paths.strategy, load_yaml(config.STRATEGY_TEMPLATE))
            reflect.apply_manual(paths, changes, config.is_stock(base), mode="migration", rationale=note)
            print(f"Sleeve {sid} created", flush=True)
    return load_yaml(config.GOAL_FILE)


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
    migrate_strategies(goal, config.goal_assets(goal))
    apply_migrations(goal, config.goal_assets(goal))
    goal = ensure_sleeves(goal)
    apply_migrations(goal, config.goal_assets(goal))  # sleeve migrations, for sleeves just created too
    if not args.asset:
        assets = config.goal_assets(goal)
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
