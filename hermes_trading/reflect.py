"""Reflection cycle: per asset, look at closed trades and change exactly ONE strategy variable.

  python -m hermes_trading.reflect --fallback   deterministic rule (no Hermes needed)
  python -m hermes_trading.reflect --hermes     ask the `hermes` CLI for a hypothesis
  python -m hermes_trading.reflect --llm        ask an OpenAI-compatible API directly (LLM_API_KEY)

Runs every asset in goal.yaml (or just --asset X). Each asset has its own strategy, trades and
cadence. Add --force to reflect before `reflection_every` new trades have closed.

Each cycle, per asset that is due:
  1. if the last automatic change has been measured and made things worse, it is reverted;
  2. otherwise a hypothesis is asked for (Hermes, the LLM, or the fallback rule) and checked with a
     walk-forward backtest (backtest.py): it is applied only if it doesn't lower the out-of-sample
     score and improves the whole-period score. A rejected hypothesis is logged, not applied.
The Hermes command is DEFAULT_HERMES_CMD + `--query-file <prompt>`; override the base with HERMES_CMD.
--llm sends the same prompt to LLM_BASE_URL's /chat/completions with LLM_MODEL (default: Gemini's
free tier). It is what the Railway worker uses: the `hermes` CLI only signs in interactively,
which a server can't do.
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
import time
from pathlib import Path
from typing import NamedTuple

import httpx

from . import config
from . import strategy as rules
from .score import metrics, score
from .storage import append_jsonl, dump_yaml, load_yaml, read_jsonl

# The only variables a reflection may touch, with hard bounds.
TUNABLE = {
    "entry.threshold": (5.0, 95.0),
    "exit_rsi": (55.0, 90.0),
    "stop_loss_pct": (0.2, 10.0),
    "stop_atr_mult": (0.0, 6.0),
    "position_size_r": (0.1, 2.0),
    "take_profit_r": (0.0, 10.0),  # 0 = no target
    "max_hold_min": (0.0, 2880.0),
    "entry.fast": (5.0, 100.0),
    "entry.slow": (20.0, 400.0),
    "entry.lookback": (5.0, 365.0),
    "entry.ma": (10.0, 400.0),
    "entry.target_vol": (0.0, 3.0),
}
# Values for fields an older strategy.yaml doesn't have (the original behaviour; see strategy.py)
TUNABLE_DEFAULTS = {"take_profit_r": 2.0, "exit_rsi": 70.0, "stop_atr_mult": 0.0, "max_hold_min": 0.0, "position_pct": 0.0,
                    "entry.fast": 50.0, "entry.slow": 200.0, "entry.lookback": 60.0, "entry.ma": 100.0,
                    "entry.target_vol": 0.0}
# Largest move allowed per cycle, so one reflection nudges a variable instead of replacing it.
MAX_STEP = {
    "entry.threshold": 5.0,
    "exit_rsi": 5.0,
    "stop_loss_pct": 0.5,
    "stop_atr_mult": 0.5,
    "position_size_r": 0.25,
    "take_profit_r": 0.5,
    "max_hold_min": 120.0,
    "entry.fast": 5.0,
    "entry.slow": 20.0,
    "entry.lookback": 10.0,
    "entry.ma": 10.0,
    "entry.target_vol": 0.1,
}
# Settings only changed by hand (dashboard) or by a migration, never by a reflection
# Numbers only changed by hand: how much of the account a position uses is the owner's call, not the AI's
MANUAL_NUMBERS = {"position_pct": (0.0, 100.0)}
CHOICES = {"entry.indicator": rules.INDICATORS,  # "hold" too, by hand; a reflection proposes AUTO_INDICATORS only "entry.direction": rules.DIRECTIONS,
           "entry.timeframe": rules.ENTRY_TIMEFRAMES, "trend_filter": rules.TREND_FILTERS}
AUTO_MODES = ("hermes", "llm", "fallback", "revert")  # changes the reflection made (not manual/migration)
REVERT_MARGIN = 0.05  # revert when the measured score fell by more than this
# Backtest validation of a hypothesis. With the 15m strategy, 90 days leaves 4-16 out-of-sample trades
# per asset (median ~10); fewer than MIN_OOS_TRADES is too thin to trust either way.
VALIDATION_DAYS = 90
# Slower signals trade less, so they need a longer window for the same out-of-sample evidence
VALIDATION_DAYS_BY_TF = {"1m": 30, "5m": 60, "15m": 90, "1h": 365, "4h": 730, "1d": 1095}


def validation_days(strategy: dict) -> int:
    return VALIDATION_DAYS_BY_TF.get(rules.params(strategy)["timeframe"], VALIDATION_DAYS)
MIN_OOS_TRADES = 6
DUPLICATE_LOOKBACK = 10  # don't re-test a value already tried in the asset's last N hypotheses
HERMES_TRADE_WINDOW = 25
DEFAULT_HERMES_CMD = (
    "hermes chat -Q --oneshot -t todo --ignore-rules --source tool --max-turns 3 --run-budget 300"
)
# Gemini's OpenAI-compatible endpoint; any other (OpenRouter, Groq, ...) works via LLM_BASE_URL
DEFAULT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_LLM_MODEL = "gemini-3.8-flash"
LLM_RETRIES = 4
MODES = ("hermes", "llm", "fallback")


def get_path(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d[part]
    return d


def _set_path_creating(d: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    for part in parents:
        d = d.setdefault(part, {})
    d[leaf] = value


def set_path(d: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    for part in parents:
        d = d[part]
    d[leaf] = value


def new_trades_since_last_reflection(trades: list[dict], hypotheses: list[dict]) -> int:
    """Count evidence since the last *valid* reflection decision.

    Technical/validation failures are logged with consumes_cadence=False, so they do not throw away
    the accumulated closed trades. Older records have no flag and therefore keep the historical
    behaviour (they consume the cadence).
    """
    cadence = [h for h in hypotheses if h.get("consumes_cadence", True)]
    if not cadence:
        return len(trades)
    last = cadence[-1]["ts"]
    return sum(1 for t in trades if t["closed_at"] > last)


def fallback_hypothesis(strategy: dict, goal: dict, m: dict) -> dict | None:
    """Drawdown breach is checked first (risk before return); only one rule ever fires."""
    if rules.params(strategy)["indicator"] in rules.STATE_INDICATORS:
        # trend following: only react to risk, with a wider stop that is hit less by noise
        if m["max_drawdown"] > float(goal["max_drawdown"]):
            old = float(strategy.get("stop_atr_mult", 3) or 3)
            return {"variable": "stop_atr_mult", "new_value": round(old + 0.5, 2), "predicted_direction": "up",
                    "rationale": f"Drawdown {m['max_drawdown']:.2%} exceeded max {goal['max_drawdown']:.2%}; "
                    "widen the ATR stop so noise stops the trend less often."}
        return None
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


def build_prompt(asset: str, strategy: dict, goal: dict, trades: list[dict], m: dict, s: float, bt: dict | None = None) -> str:
    trade_lines = "\n".join(json.dumps(t) for t in trades)
    market = (
        "a US-listed ETF/stock, long only, traded in regular market hours via an Alpaca paper account"
        if config.is_stock(asset)
        else "a crypto spot pair traded 24/7"
    )
    stock = config.is_stock(asset)
    cost_text = "; ".join(
        f"{side}s fee {f * 100:.3f}%, slippage {sl * 100:.3f}%" + (" plus perpetual funding" if rules.pays_funding(goal, stock, side) else "")
        for side, (f, sl) in ((side, rules.costs(goal, stock, side)) for side in ("long", "short")))
    backtest_text = "not available"
    if bt:
        backtest_text = json.dumps({k: bt[k] for k in ("from", "to", "buy_hold_pct", "all", "in_sample", "out_of_sample")})
    return f"""You are tuning a paper-trading strategy for {asset} ({market}). Propose exactly ONE change.

How the strategy trades (strategy.yaml fields):
- entry.indicator "rsi" (mean reversion): RSI(14) on entry.timeframe candles; long enters when RSI < entry.threshold
  (short: RSI > threshold), only when trend_filter (off/1h/4h EMA50) agrees; exits when RSI reaches exit_rsi
  (short: 100 - exit_rsi).
- entry.indicator "ema_cross" (trend following): long while EMA(entry.fast) > EMA(entry.slow) on entry.timeframe
  candles, short while below (entry.direction both/long/short); exits and flips at each cross. entry.threshold and
  exit_rsi don't apply; tune entry.fast, entry.slow, stop_atr_mult or take_profit_r (0 = no target) instead.
- entry.indicator "tsmom" (time-series momentum): long while the close is above the close entry.lookback candles
  ago, short while below; "ma_regime": long while the close is above its entry.ma-candle simple moving average.
  Both exit/flip when that changes; entry.target_vol > 0 scales the size down when recent volatility is higher.
- exits: stop, target (take_profit_r x stop distance), the signal above, or after max_hold_min minutes (0 = no
  limit). Stop distance = stop_atr_mult x ATR(14) when stop_atr_mult > 0, otherwise stop_loss_pct % of the price,
  and never closer than a few times the round-trip cost.
- size: position_size_r % of the account is lost if the stop is hit, unless position_pct > 0 (set by the owner, not
  tunable): then every position uses position_pct % of the account and position_size_r has no effect.
- costs per side of a trade: {cost_text} (already included in all P&L below).

Walk-forward backtest of the CURRENT strategy on recent history (in sample = first 70%, out of sample = last 30%):
{backtest_text}
Your change will be backtested the same way and applied only if it holds out of sample.

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
    asset: str, strategy: dict, goal: dict, trades: list[dict], m: dict, s: float, bt: dict | None = None
) -> dict:
    # One-shot, text-only: no terminal/file/code tools, bounded turns and wall-clock.
    cmd = shlex.split(config.env("HERMES_CMD", DEFAULT_HERMES_CMD))
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(f"`{cmd[0]}` not found on PATH — install Hermes or use --fallback.")
    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = Path(tmp) / "prompt.txt"
        prompt_file.write_text(build_prompt(asset, strategy, goal, trades, m, s, bt), encoding="utf-8")
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


def llm_hypothesis(
    asset: str, strategy: dict, goal: dict, trades: list[dict], m: dict, s: float, bt: dict | None = None
) -> dict:
    key = config.env("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set — create a free key at aistudio.google.com or use --fallback.")
    url = config.env("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/") + "/chat/completions"
    body = {
        "model": config.env("LLM_MODEL", DEFAULT_LLM_MODEL),
        "messages": [{"role": "user", "content": build_prompt(asset, strategy, goal, trades, m, s, bt)}],
    }
    # Free tiers allow a few requests per minute: wait out 429s (and gateway blips) and retry;
    # anything else fails the asset.
    for attempt in range(LLM_RETRIES):
        r = httpx.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=180)
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == LLM_RETRIES - 1:
            break
        try:
            wait = float(r.headers.get("retry-after", ""))
        except ValueError:
            wait = 15 * 2**attempt
        time.sleep(min(wait, 120))
    if r.status_code != 200:
        raise RuntimeError(f"LLM API returned {r.status_code}: {r.text.strip()[-300:]}")
    hyp = parse_hypothesis(r.json()["choices"][0]["message"]["content"] or "")
    hyp.setdefault("rationale", "")
    hyp.setdefault("predicted_direction", "up")
    return hyp


def apply(
    paths: config.AssetPaths, strategy: dict, hyp: dict, mode: str, m: dict, s: float,
    backtest: dict | None = None, unclamped: bool = False,
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
    step = float("inf") if unclamped else MAX_STEP[variable]  # a revert goes straight back
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
        "consumes_cadence": True,
    }
    if backtest:
        record["backtest"] = backtest
    append_jsonl(paths.hypotheses, record)
    return record


def apply_manual(paths: config.AssetPaths, changes: dict, stock: bool, mode: str = "manual",
                 rationale: str = "Cambiado a mano desde el dashboard.") -> list[dict]:
    """Settings changed by hand from the dashboard. Same bounds, versioning and hypothesis log as a
    reflection, but no per-cycle step limit and several variables at once. The log entry also
    restarts the reflection cadence, so the new settings get `reflection_every` trades first."""
    strategy = load_yaml(paths.strategy)
    updated = copy.deepcopy(strategy)
    diffs = []
    for variable, requested in changes.items():
        if variable in CHOICES:
            new = str(requested)
            if new not in CHOICES[variable]:
                raise ValueError(f"{variable} debe ser uno de: {', '.join(CHOICES[variable])}")
            if variable == "entry.direction" and stock and new != "long":
                raise ValueError("las acciones y los ETFs solo operan en long")
        elif variable in TUNABLE or variable in MANUAL_NUMBERS:
            lo, hi = TUNABLE.get(variable) or MANUAL_NUMBERS[variable]
            try:
                new = round(float(requested), 4)
            except (TypeError, ValueError):
                raise ValueError(f"{variable} debe ser un número") from None
            if not lo <= new <= hi:
                raise ValueError(f"{variable} debe estar entre {lo:g} y {hi:g}")
        else:
            raise ValueError(f"{variable!r} no se puede cambiar; permitidos: {', '.join(sorted(CHOICES) + sorted(TUNABLE) + sorted(MANUAL_NUMBERS))}")
        try:
            old = get_path(strategy, variable)
        except KeyError:
            old = TUNABLE_DEFAULTS[variable] if variable in TUNABLE_DEFAULTS else {"entry.timeframe": "1m", "trend_filter": "off"}.get(variable)
        if isinstance(old, int) and isinstance(new, float) and new.is_integer():
            new = int(new)
        if new != old:
            _set_path_creating(updated, variable, new)
            diffs.append((variable, old, new))
    if not diffs:
        return []
    try:  # the combination must be valid too (e.g. entry.fast below entry.slow), or the worker would fail every tick
        rules.params(updated)
    except ValueError as e:
        raise ValueError(f"combinación no válida: {e}") from None

    prior_version = str(strategy["version"])
    dump_yaml(paths.history / f"v{int(prior_version):04d}.yaml", strategy)
    updated["version"] = f"{int(prior_version) + 1:02d}"
    dump_yaml(paths.strategy, updated)
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    records = []
    for variable, old, new in diffs:
        record = {
            "ts": ts, "asset": paths.asset, "mode": mode,
            "from_version": prior_version, "to_version": updated["version"],
            "variable": variable, "old_value": old, "new_value": new,
            "requested_value": new, "clamped": False,
            "rationale": rationale, "predicted_direction": None,
            "consumes_cadence": True,
        }
        append_jsonl(paths.hypotheses, record)
        records.append(record)
    return records


def evaluate_changes(trades: list[dict], hypotheses: list[dict], goal: dict) -> list[dict]:
    """Did each strategy change help? Compares the score of the `reflection_every` trades closed
    before it with the ones opened under it (until the next change). Small samples: a hint only."""
    n = int(goal["reflection_every"])
    applied = [h for h in hypotheses if not h.get("rejected") and not h.get("no_change")]
    out = []
    for h in hypotheses:
        if h.get("no_change"):
            out.append({**h, "evaluation": {"status": "observed"}})
            continue
        if h.get("rejected"):
            out.append({**h, "evaluation": {"status": "rejected"}})
            continue
        end = next((x["ts"] for x in applied if x["ts"] > h["ts"]), None)
        before = [t for t in trades if t["closed_at"] <= h["ts"]][-n:]
        after = [t for t in trades if t["opened_at"] >= h["ts"] and (end is None or t["opened_at"] < end)][:n]
        ev: dict = {"n_before": len(before), "n_after": len(after), "needed": n}
        if before and (len(after) >= n or (end is not None and len(after) >= 2)):
            sb, sa = score(before, goal), score(after, goal)
            ev.update(status="done", score_before=sb, score_after=sa, improved=sa > sb,
                      avg_before=sum(float(t["pnl_pct"]) for t in before) / len(before),
                      avg_after=sum(float(t["pnl_pct"]) for t in after) / len(after))
            if h.get("predicted_direction") in ("up", "down"):
                ev["matched"] = (sa > sb) == (h["predicted_direction"] == "up")
        else:
            ev["status"] = "pending" if end is None else "insufficient"
        out.append({**h, "evaluation": ev})
    return out


class Proposal(NamedTuple):
    paths: config.AssetPaths
    strategy: dict  # the strategy the hypothesis was built from
    hyp: dict
    m: dict
    s: float
    mode: str = ""           # overrides the cycle's mode (e.g. "revert" / "observe")
    backtest: dict | None = None
    rejected: bool = False
    consumes_cadence: bool = True


def _bt_brief(r: dict) -> dict:
    return {"all_score": r["all"]["score"], "all_return_pct": r["all"]["return_pct"], "all_n": r["all"]["n"],
            "oos_score": r["out_of_sample"]["score"], "oos_return_pct": r["out_of_sample"]["return_pct"], "oos_n": r["out_of_sample"]["n"]}


def _recent_duplicate(hypotheses: list[dict], variable: str, value: float) -> bool:
    """Avoid repeatedly testing the same bounded parameter value in automatic reflection."""
    for h in hypotheses[-DUPLICATE_LOOKBACK:]:
        if h.get("variable") != variable:
            continue
        try:
            if abs(float(h.get("new_value")) - float(value)) < 1e-9:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _revert_candidate(trades: list[dict], hypotheses: list[dict], goal: dict) -> dict | None:
    """The last applied automatic change, if it has been measured and made things clearly worse."""
    applied = [h for h in evaluate_changes(trades, hypotheses, goal)
               if not h.get("rejected") and not h.get("no_change")]
    if not applied:
        return None
    last = applied[-1]
    ev = last["evaluation"]
    if last.get("mode") not in AUTO_MODES or last.get("mode") == "revert" or ev.get("status") != "done":
        return None
    if ev["score_after"] < ev["score_before"] - REVERT_MARGIN:
        return last
    return None


def propose(asset: str, goal: dict, mode: str, force: bool, validate: bool = True) -> Proposal | str:
    """Read an asset's state and ask for one change. Returns a one-line report when there's
    nothing to apply. Writes nothing, so it can run while the worker trades."""
    paths = config.asset_paths(asset)
    if not paths.strategy.exists():
        return "no state yet — worker hasn't started this asset."
    strategy = load_yaml(paths.strategy)
    trades = read_jsonl(paths.trades)
    hypotheses = read_jsonl(paths.hypotheses)

    if not trades:
        return "no closed trades yet — nothing to reflect on."
    if str(strategy.get("entry", {}).get("indicator")) == "hold":
        return "buy and hold — the owner's choice, not tuned by reflection."
    pending = new_trades_since_last_reflection(trades, hypotheses)
    if pending < int(goal["reflection_every"]) and not force:
        return f"{pending}/{goal['reflection_every']} new closed trades since last reflection — waiting."

    m = metrics(trades)
    s = score(trades, goal)
    bad = _revert_candidate(trades, hypotheses, goal)
    if bad:
        ev = bad["evaluation"]
        hyp = {"variable": bad["variable"], "new_value": bad["old_value"], "predicted_direction": "up",
               "rationale": f"Revert v{bad['from_version']} → v{bad['to_version']}: score fell from {ev['score_before']:.2f} "
                            f"to {ev['score_after']:.2f} over the {ev['n_after']} trades under it."}
        return Proposal(paths, strategy, hyp, m, s, mode="revert")

    from . import backtest  # network + history; imported here so the dashboard doesn't pay for it

    baseline = None
    if validate:
        try:
            baseline = backtest.run(asset, strategy, goal, validation_days(strategy))
        except Exception as e:  # no history (new listing, data outage): decide without it
            baseline = {"error": f"{type(e).__name__}: {e}"[:200]}
    bt_ok = baseline if baseline and "error" not in baseline else None
    if mode == "hermes":
        hyp = hermes_hypothesis(asset, strategy, goal, trades[-HERMES_TRADE_WINDOW:], m, s, bt_ok)
    elif mode == "llm":
        hyp = llm_hypothesis(asset, strategy, goal, trades[-HERMES_TRADE_WINDOW:], m, s, bt_ok)
    else:
        hyp = fallback_hypothesis(strategy, goal, m)
    if hyp is None:
        return Proposal(
            paths, strategy, {"rationale": f"targets met (score {s}) — no change needed."}, m, s,
            mode="observe",
        )
    if hyp["variable"] == "entry.indicator" and hyp["new_value"] not in rules.AUTO_INDICATORS:
        return Proposal(paths, strategy, hyp, m, s, rejected=True, consumes_cadence=False,
                        backtest={"verdict": "rejected", "reason": "invalid", "error": f"{hyp['new_value']} is set by hand only"})
    if _recent_duplicate(hypotheses, hyp["variable"], _bounded(strategy, hyp)):
        # A duplicate is a model-quality issue, not new evidence: log it without consuming the 10 trades.
        return Proposal(paths, strategy, hyp, m, s, backtest={"verdict": "rejected", "reason": "duplicate"},
                        rejected=True, consumes_cadence=False)
    if not validate:
        return Proposal(paths, strategy, hyp, m, s)
    if not bt_ok:
        # Never apply an AI proposal without its backtest gate. Keep the accumulated live evidence.
        return Proposal(paths, strategy, hyp, m, s,
                        backtest={"verdict": "unavailable", "error": (baseline or {}).get("error")},
                        rejected=True, consumes_cadence=False)

    candidate_strategy = copy.deepcopy(strategy)
    _set_path_creating(candidate_strategy, hyp["variable"], _bounded(strategy, hyp))
    try:
        rules.params(candidate_strategy)
    except ValueError as e:  # e.g. entry.fast above entry.slow: never apply it unvalidated
        return Proposal(paths, strategy, hyp, m, s,
                        backtest={"verdict": "rejected", "reason": "invalid", "error": str(e)[:200]},
                        rejected=True, consumes_cadence=False)
    try:
        candidate = backtest.run(asset, candidate_strategy, goal, validation_days(strategy))
    except Exception as e:
        return Proposal(paths, strategy, hyp, m, s,
                        backtest={"verdict": "unavailable", "error": f"{type(e).__name__}: {e}"[:200]},
                        rejected=True, consumes_cadence=False)
    b, c = _bt_brief(bt_ok), _bt_brief(candidate)
    if c["oos_n"] < MIN_OOS_TRADES:
        verdict = {"verdict": "rejected", "reason": "insufficient_oos_trades",
                   "minimum_oos_trades": MIN_OOS_TRADES, "baseline": b, "candidate": c,
                   "period": {"from": bt_ok["from"], "to": bt_ok["to"], "split": bt_ok["split"]}}
        return Proposal(paths, strategy, hyp, m, s, backtest=verdict, rejected=True, consumes_cadence=False)
    better_all = (c["all_score"], c["all_return_pct"]) > (b["all_score"], b["all_return_pct"])
    accepted = c["oos_score"] >= b["oos_score"] and better_all
    verdict = {"verdict": "accepted" if accepted else "rejected", "baseline": b, "candidate": c,
               "period": {"from": bt_ok["from"], "to": bt_ok["to"], "split": bt_ok["split"]}}
    return Proposal(paths, strategy, hyp, m, s, backtest=verdict, rejected=not accepted)


def _bounded(strategy: dict, hyp: dict):
    """The value apply() would set: within bounds and one MAX_STEP of the current value."""
    variable = hyp["variable"]
    if variable not in TUNABLE:
        raise ValueError(f"{variable!r} is not tunable; allowed: {sorted(TUNABLE)}")
    lo, hi = TUNABLE[variable]
    try:
        old = float(get_path(strategy, variable))
    except KeyError:
        old = TUNABLE_DEFAULTS[variable]
    step = MAX_STEP[variable]
    return round(min(max(float(hyp["new_value"]), old - step, lo), old + step, hi), 4)


def reject(p: Proposal, mode: str) -> dict:
    """Log a refused hypothesis without touching the strategy.

    Only evidence-based rejections consume the 10-trade cadence. Technical failures, insufficient
    OOS history, invalid proposals and duplicates remain visible in the log but keep the evidence.
    """
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asset": p.paths.asset, "mode": mode, "rejected": True,
        "consumes_cadence": p.consumes_cadence,
        "from_version": str(p.strategy["version"]), "to_version": str(p.strategy["version"]),
        "variable": p.hyp["variable"], "old_value": _current(p.strategy, p.hyp["variable"]),
        "new_value": _bounded(p.strategy, p.hyp), "requested_value": p.hyp["new_value"], "clamped": False,
        "rationale": p.hyp.get("rationale", ""), "predicted_direction": p.hyp.get("predicted_direction"),
        "backtest": p.backtest, "metrics_before": p.m, "score_before": p.s,
    }
    append_jsonl(p.paths.hypotheses, record)
    return record


def _current(strategy: dict, variable: str):
    try:
        return get_path(strategy, variable)
    except KeyError:
        return TUNABLE_DEFAULTS.get(variable)


def observe(p: Proposal, mode: str) -> dict:
    """Record a valid reflection that concluded no parameter change was needed."""
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asset": p.paths.asset,
        "mode": mode,
        "no_change": True,
        "consumes_cadence": True,
        "from_version": str(p.strategy["version"]),
        "to_version": str(p.strategy["version"]),
        "rationale": p.hyp.get("rationale", "no change needed"),
        "metrics_before": p.m,
        "score_before": p.s,
    }
    append_jsonl(p.paths.hypotheses, record)
    return record


def apply_proposal(p: Proposal, mode: str) -> str:
    mode = p.mode or mode
    if mode == "observe":
        r = observe(p, mode)
        return r["rationale"]
    if p.rejected:
        r = reject(p, mode)
        verdict = (p.backtest or {}).get("verdict")
        if "candidate" not in (p.backtest or {}):
            return f"rejected: {r['variable']} {r['old_value']} → {r['new_value']} ({(p.backtest or {}).get('reason', verdict or 'validation failed')})"
        c, b = p.backtest["candidate"], p.backtest["baseline"]
        return (f"rejected by backtest: {r['variable']} {r['old_value']} → {r['new_value']} "
                f"(OOS trades={c['oos_n']}, score {b['oos_score']:+.2f} → {c['oos_score']:+.2f}, "
                f"whole period {b['all_score']:+.2f} → {c['all_score']:+.2f})"
                + (f" — fewer than {MIN_OOS_TRADES} out-of-sample trades" if p.backtest.get("reason") == "insufficient_oos_trades" else ""))

    record = apply(p.paths, p.strategy, p.hyp, mode, p.m, p.s, backtest=p.backtest, unclamped=mode == "revert")
    if record is None:
        return f"{p.hyp['variable']} is already at {p.hyp['new_value']} or its bound — no change."
    clamp_note = f" [requested {record['requested_value']}, clamped to max step]" if record["clamped"] else ""
    return (
        f"v{record['from_version']} → v{record['to_version']}: {record['variable']} "
        f"{record['old_value']} → {record['new_value']}{clamp_note}  ({record['rationale']})"
    )


def reflect_asset(asset: str, goal: dict, mode: str, force: bool) -> str:
    """One reflection cycle for one asset; returns a one-line report."""
    p = propose(asset, goal, mode, force)
    return p if isinstance(p, str) else apply_proposal(p, mode)


def main(argv: list[str] | None = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(description="Run one reflection cycle per asset.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--fallback", action="store_true", help="deterministic rule")
    which.add_argument("--hermes", action="store_true", help="ask Hermes for the hypothesis")
    which.add_argument("--llm", action="store_true", help="ask an OpenAI-compatible API (LLM_API_KEY)")
    parser.add_argument("--force", action="store_true", help="ignore reflection_every cadence")
    parser.add_argument("--asset", help="reflect on only this asset (default: all in goal.yaml)")
    args = parser.parse_args(argv)

    goal = load_yaml(config.GOAL_FILE)
    assets = [args.asset] if args.asset else config.goal_assets(goal)
    mode = next(m for m in MODES if getattr(args, m))
    failed = False
    for asset in assets:
        try:
            report = reflect_asset(asset, goal, mode, args.force)
        except Exception as e:  # one asset's failure must not block the others
            failed = True
            report = f"FAILED — {type(e).__name__}: {e}"
        print(f"{asset}: {report}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
