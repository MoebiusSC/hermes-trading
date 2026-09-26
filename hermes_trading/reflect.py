"""Reflection cycle: per asset, look at closed trades and change exactly ONE strategy variable.

  python -m hermes_trading.reflect --fallback   deterministic rule (no Hermes needed)
  python -m hermes_trading.reflect --hermes     ask the `hermes` CLI for a hypothesis

Runs every asset in goal.yaml (or just --asset X). Each asset has its own strategy, trades and
cadence. Add --force to reflect before `reflection_every` new trades have closed.
The Hermes command is DEFAULT_HERMES_CMD + `--query-file <prompt>`; override the base with HERMES_CMD.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config
from .score import metrics, score
from .storage import append_jsonl, dump_yaml, load_yaml, read_jsonl

# The only variables a reflection may touch, with hard bounds.
TUNABLE = {
    "entry.threshold": (5.0, 95.0),
    "stop_loss_pct": (0.2, 10.0),
    "position_size_r": (0.1, 2.0),
    "take_profit_r": (0.5, 10.0),
}
TUNABLE_DEFAULTS = {"take_profit_r": 2.0}
# Largest move allowed per cycle, so one reflection nudges a variable instead of replacing it.
MAX_STEP = {
    "entry.threshold": 5.0,
    "stop_loss_pct": 0.5,
    "position_size_r": 0.25,
    "take_profit_r": 0.5,
}
HERMES_TRADE_WINDOW = 25
DEFAULT_HERMES_CMD = (
    "hermes chat -Q --oneshot -t todo --ignore-rules --source tool --max-turns 3 --run-budget 300"
)


def get_path(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d[part]
    return d


def set_path(d: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    for part in parents:
        d = d[part]
    d[leaf] = value


def new_trades_since_last_reflection(trades: list[dict], hypotheses: list[dict]) -> int:
    if not hypotheses:
        return len(trades)
    last = hypotheses[-1]["ts"]
    return sum(1 for t in trades if t["closed_at"] > last)


def fallback_hypothesis(strategy: dict, goal: dict, m: dict) -> dict | None:
    """Drawdown breach is checked first (risk before return); only one rule ever fires."""
    if m["max_drawdown"] > float(goal["max_drawdown"]):
        old = float(strategy["stop_loss_pct"])
        return {
            "variable": "stop_loss_pct",
            "new_value": round(old - 0.2, 2),
            "rationale": f"Drawdown {m['max_drawdown']:.2%} exceeded max {goal['max_drawdown']:.2%}; "
            "tighten stop by 0.2.",
            "predicted_direction": "up",
        }
    if m["realised_return"] < float(goal["target_return_30d"]):
        step = 2 if strategy["entry"]["direction"] == "long" else -2
        return {
            "variable": "entry.threshold",
            "new_value": strategy["entry"]["threshold"] + step,
            "rationale": f"Return {m['realised_return']:.2%} below target {goal['target_return_30d']:.2%}; "
            "loosen entry threshold by 2 to take more trades.",
            "predicted_direction": "up",
        }
    return None


def build_prompt(asset: str, strategy: dict, goal: dict, trades: list[dict], m: dict, s: float) -> str:
    trade_lines = "\n".join(json.dumps(t) for t in trades)
    market = (
        "a US-listed ETF/stock, long only, traded in regular market hours via an Alpaca paper account"
        if config.is_stock(asset)
        else "a crypto spot pair traded 24/7"
    )
    return f"""You are tuning a paper-trading strategy for {asset} ({market}; 1-minute candles). Propose exactly ONE change.

Goal (goal.yaml):
{json.dumps(goal, indent=2)}

Current strategy (strategy.yaml):
{json.dumps(strategy, indent=2)}

Metrics over all closed trades: {json.dumps(m)}
Composite score in [-1, 1]: {s:.3f}

Last {len(trades)} closed trades (JSON lines):
{trade_lines}

You may change only one of these variables, within these [min, max] bounds:
{json.dumps(TUNABLE)}
Maximum change from the current value in one cycle (larger proposals are clamped):
{json.dumps(MAX_STEP)}

Reply with a single JSON object and nothing else:
{{"variable": "<one of the names above>", "new_value": <number>, "rationale": "<one or two sentences>", "predicted_direction": "up or down (expected score change)"}}
"""


def parse_hypothesis(text: str) -> dict:
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.S)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "variable" in obj and "new_value" in obj:
            return obj
    raise ValueError("no hypothesis JSON found in Hermes output")


def hermes_hypothesis(
    asset: str, strategy: dict, goal: dict, trades: list[dict], m: dict, s: float
) -> dict:
    # One-shot, text-only: no terminal/file/code tools, bounded turns and wall-clock.
    cmd = shlex.split(config.env("HERMES_CMD", DEFAULT_HERMES_CMD))
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(f"`{cmd[0]}` not found on PATH — install Hermes or use --fallback.")
    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = Path(tmp) / "prompt.txt"
        prompt_file.write_text(build_prompt(asset, strategy, goal, trades, m, s), encoding="utf-8")
        proc = subprocess.run(
            [exe, *cmd[1:], "--query-file", str(prompt_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=900,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"hermes exited with {proc.returncode}: {proc.stderr.strip()[-500:]}")
    hyp = parse_hypothesis(proc.stdout)
    hyp.setdefault("rationale", "")
    hyp.setdefault("predicted_direction", "up")
    return hyp


def apply(
    paths: config.AssetPaths, strategy: dict, hyp: dict, mode: str, m: dict, s: float
) -> dict | None:
    variable = hyp["variable"]
    if variable not in TUNABLE:
        raise ValueError(f"{variable!r} is not tunable; allowed: {sorted(TUNABLE)}")
    lo, hi = TUNABLE[variable]
    try:
        old = get_path(strategy, variable)
    except KeyError:
        old = TUNABLE_DEFAULTS[variable]
    requested = float(hyp["new_value"])
    step = MAX_STEP[variable]
    new = min(max(requested, float(old) - step, lo), float(old) + step, hi)
    new = round(new, 4)
    if isinstance(old, int) and new.is_integer():
        new = int(new)
    if new == old:
        return None

    prior_version = str(strategy["version"])
    dump_yaml(paths.history / f"v{int(prior_version):04d}.yaml", strategy)

    updated = copy.deepcopy(strategy)
    set_path(updated, variable, new)
    updated["version"] = f"{int(prior_version) + 1:02d}"
    dump_yaml(paths.strategy, updated)

    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asset": paths.asset,
        "mode": mode,
        "from_version": prior_version,
        "to_version": updated["version"],
        "variable": variable,
        "old_value": old,
        "new_value": new,
        "requested_value": hyp["new_value"],
        "clamped": new != requested,
        "rationale": hyp.get("rationale", ""),
        "predicted_direction": hyp.get("predicted_direction"),
        "metrics_before": m,
        "score_before": s,
    }
    append_jsonl(paths.hypotheses, record)
    return record


def apply_manual(paths: config.AssetPaths, changes: dict, stock: bool) -> list[dict]:
    """Settings changed by hand from the dashboard. Same bounds, versioning and hypothesis log as a
    reflection, but no per-cycle step limit and several variables at once. The log entry also
    restarts the reflection cadence, so the new settings get `reflection_every` trades first."""
    strategy = load_yaml(paths.strategy)
    updated = copy.deepcopy(strategy)
    diffs = []
    for variable, requested in changes.items():
        if variable == "entry.direction":
            new = str(requested)
            if new not in ("long", "short"):
                raise ValueError("la dirección debe ser long o short")
            if stock and new != "long":
                raise ValueError("las acciones y los ETFs solo operan en long")
        elif variable in TUNABLE:
            lo, hi = TUNABLE[variable]
            try:
                new = round(float(requested), 4)
            except (TypeError, ValueError):
                raise ValueError(f"{variable} debe ser un número") from None
            if not lo <= new <= hi:
                raise ValueError(f"{variable} debe estar entre {lo:g} y {hi:g}")
        else:
            raise ValueError(f"{variable!r} no se puede cambiar; permitidos: entry.direction, {', '.join(sorted(TUNABLE))}")
        try:
            old = get_path(strategy, variable)
        except KeyError:
            old = TUNABLE_DEFAULTS[variable]
        if isinstance(old, int) and isinstance(new, float) and new.is_integer():
            new = int(new)
        if new != old:
            set_path(updated, variable, new)
            diffs.append((variable, old, new))
    if not diffs:
        return []

    prior_version = str(strategy["version"])
    dump_yaml(paths.history / f"v{int(prior_version):04d}.yaml", strategy)
    updated["version"] = f"{int(prior_version) + 1:02d}"
    dump_yaml(paths.strategy, updated)
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    records = []
    for variable, old, new in diffs:
        record = {
            "ts": ts, "asset": paths.asset, "mode": "manual",
            "from_version": prior_version, "to_version": updated["version"],
            "variable": variable, "old_value": old, "new_value": new,
            "requested_value": new, "clamped": False,
            "rationale": "Cambiado a mano desde el dashboard.", "predicted_direction": None,
        }
        append_jsonl(paths.hypotheses, record)
        records.append(record)
    return records


def reflect_asset(asset: str, goal: dict, hermes: bool, force: bool) -> str:
    """One reflection cycle for one asset; returns a one-line report."""
    paths = config.asset_paths(asset)
    if not paths.strategy.exists():
        return "no state yet — worker hasn't started this asset."
    strategy = load_yaml(paths.strategy)
    trades = read_jsonl(paths.trades)
    hypotheses = read_jsonl(paths.hypotheses)

    if not trades:
        return "no closed trades yet — nothing to reflect on."
    pending = new_trades_since_last_reflection(trades, hypotheses)
    if pending < int(goal["reflection_every"]) and not force:
        return f"{pending}/{goal['reflection_every']} new closed trades since last reflection — waiting."

    m = metrics(trades)
    s = score(trades, goal)
    if hermes:
        hyp = hermes_hypothesis(asset, strategy, goal, trades[-HERMES_TRADE_WINDOW:], m, s)
    else:
        hyp = fallback_hypothesis(strategy, goal, m)
    if hyp is None:
        return f"targets met (score {s}) — no change this cycle."
    record = apply(paths, strategy, hyp, "hermes" if hermes else "fallback", m, s)
    if record is None:
        return f"{hyp['variable']} is already at {hyp['new_value']} or its bound — no change."
    clamp_note = f" [requested {record['requested_value']}, clamped to max step]" if record["clamped"] else ""
    return (
        f"v{record['from_version']} → v{record['to_version']}: {record['variable']} "
        f"{record['old_value']} → {record['new_value']}{clamp_note}  ({record['rationale']})"
    )


def main(argv: list[str] | None = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(description="Run one reflection cycle per asset.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--fallback", action="store_true", help="deterministic rule")
    which.add_argument("--hermes", action="store_true", help="ask Hermes for the hypothesis")
    parser.add_argument("--force", action="store_true", help="ignore reflection_every cadence")
    parser.add_argument("--asset", help="reflect on only this asset (default: all in goal.yaml)")
    args = parser.parse_args(argv)

    goal = load_yaml(config.GOAL_FILE)
    assets = [args.asset] if args.asset else config.goal_assets(goal)
    failed = False
    for asset in assets:
        try:
            report = reflect_asset(asset, goal, args.hermes, args.force)
        except Exception as e:  # one asset's failure must not block the others
            failed = True
            report = f"FAILED — {type(e).__name__}: {e}"
        print(f"{asset}: {report}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
