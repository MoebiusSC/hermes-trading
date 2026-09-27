"""Monthly momentum rotation across large US stocks (goal.yaml `rotation:`), next to the per-asset books.

On the first trading day of each month (`after_open_min` minutes after the open) the `universe` is
ranked by its 12-1 month return: the close `skip_days` trading days ago over the close
`lookback_days` ago, on split- and dividend-adjusted daily bars. The account holds the `top` best,
equal weight: positions that dropped out are sold and new picks bought with 1/top of the account's
value each (within its cash). Kept positions are not resized.

Why a portfolio rule and not one more strategy per stock: scripts/research_stocks.py. Per-stock
trend and momentum signals kept a Sharpe close to holding the stocks but made about half their
return, and a stock's fit in one half of the history didn't predict the other half. Ranking the
stocks against each other did: 20.7%/yr over five years (top 25 US stocks of 2021) and 19.6%/yr
over eight (top 25 of 2016), against 12.5% and 16.6% for holding them all, drawdowns ~35%.

One virtual account in state/rotation/: paper_account.json (cash, equity = start + closed P&L,
positions, picks, last_rebalance, pending buys), trades.jsonl (each closed position) and
rankings.jsonl (every ranking and its picks). Orders are real Alpaca paper orders; buys are capped
by Alpaca's buying power, which the ETF accounts share, and wait in `pending` until there is cash.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import time
import uuid
from pathlib import Path

from . import config
from . import strategy as rules
from .adapters import alpaca
from .adapters.alpaca import AlpacaError
from .storage import append_jsonl, write_json

DEFAULTS = {"enabled": True, "top": 5, "capital": 5000.0, "lookback_days": 252, "skip_days": 21, "after_open_min": 30}
MIN_ORDER_USD = 1.0  # Alpaca's fractional-order minimum
BUYING_POWER_USE = 0.98  # headroom for the price moving between the quote and the fill
RANK_CONCURRENCY = 5


def settings(goal: dict) -> dict | None:
    """goal.yaml `rotation:` with defaults, or None when there is none or it's disabled."""
    spec = goal.get("rotation")
    if not spec or not spec.get("enabled", True):
        return None
    s = {**DEFAULTS, **spec, "universe": [str(x).upper() for x in spec.get("universe") or []]}
    s["top"] = int(s["top"])
    if s["top"] < 1 or len(s["universe"]) < s["top"]:
        raise ValueError(f"rotation needs at least top={s['top']} stocks in its universe")
    if not 0 < int(s["skip_days"]) < int(s["lookback_days"]):
        raise ValueError("rotation: skip_days must be between 0 and lookback_days")
    return s


def momentum(closes: list[float], lookback: int, skip: int) -> float | None:
    """12-1 month return from daily closes (oldest first): close `skip` days ago over `lookback` ago."""
    if len(closes) < lookback + 1 or closes[-1 - lookback] <= 0:
        return None
    return closes[-1 - skip] / closes[-1 - lookback] - 1


def pick(ranking: list[tuple[str, float]], top: int) -> list[str]:
    return [sym for sym, _ in sorted(ranking, key=lambda r: -r[1])[:top]]


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class RotationBook:
    asset = "rotation"

    def __init__(self, spec: dict, root: Path | None = None) -> None:
        self.spec = spec
        self.root = root or config.ROTATION_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.paper_path = self.root / "paper_account.json"
        self.trades_path = self.root / "trades.jsonl"
        self.rankings_path = self.root / "rankings.jsonl"
        self.paper = self._load()
        self.lock = asyncio.Lock()
        self.exclude: set[str] = set()  # symbols a per-asset book trades: never bought here
        self.session: dict = {}

    # --- account -------------------------------------------------------------

    def _load(self) -> dict:
        if self.paper_path.exists():
            return json.loads(self.paper_path.read_text(encoding="utf-8"))
        capital = float(self.spec["capital"])
        return {"start_equity": capital, "cash": capital, "equity": capital, "positions": {}, "picks": [],
                "pending": [], "last_rebalance": None}

    def _save(self) -> None:
        write_json(self.paper_path, self.paper)

    @property
    def start_equity(self) -> float:
        return float(self.paper["start_equity"])

    def _price(self, pos: dict) -> float:
        return float(pos.get("last_price") or pos["entry_price"])

    def invested(self) -> float:
        return sum(p["qty"] * self._price(p) for p in self.paper["positions"].values())

    def unrealized(self) -> float:
        return sum(p["qty"] * (self._price(p) - p["entry_price"]) for p in self.paper["positions"].values())

    def value(self) -> float:
        return float(self.paper["cash"]) + self.invested()

    def summary(self) -> dict:
        return {"start_equity": self.start_equity, "equity": round(self.paper["equity"], 4), "cash": round(self.paper["cash"], 4),
                "value": round(self.value(), 4), "positions": len(self.paper["positions"]), "picks": self.paper.get("picks") or [],
                "pending": self.paper.get("pending") or [], "last_rebalance": self.paper.get("last_rebalance")}

    # --- one tick ------------------------------------------------------------

    async def tick(self) -> dict:
        async with self.lock:
            try:
                return await self._tick()
            except Exception as e:  # the ETFs and crypto keep trading whatever happens here
                return {**self.summary(), "decision": "error", "error": f"{type(e).__name__}: {e}"[:300]}

    async def _tick(self) -> dict:
        broker = alpaca.client()
        self.session = await broker.session()
        if not self.session["is_open"]:
            opens = self.session["next_open"][:16].replace("T", " ")
            return {**self.summary(), "market": "closed", "decision": f"market closed · opens {opens} ET"}
        await self._mark(await broker.positions())
        minutes = self.session.get("minutes_since_open")
        if minutes is None or minutes < float(self.spec["after_open_min"]):
            return {**self.summary(), "decision": f"waiting · trades from {self.spec['after_open_min']} min after the open"}
        month = self.session["date"][:7]
        if self.paper.get("last_rebalance") != month:
            decision = await self._rebalance(month)
        elif self.paper.get("pending"):
            decision = await self._buy_pending()
        else:
            decision = f"holding {', '.join(self.paper['positions']) or 'nothing'} until next month"
        return {**self.summary(), "decision": decision}

    async def _mark(self, held: dict[str, dict]) -> None:
        """Last prices from Alpaca; a position gone from Alpaca (sold by hand, account reset) is
        recorded as closed at its last price."""
        for sym, pos in list(self.paper["positions"].items()):
            row = held.get(sym)
            if row is None:
                await self._record_close(sym, self._price(pos), "missing_at_broker")
                continue
            pos["last_price"] = float(row.get("current_price") or pos["entry_price"])
        self._save()

    # --- rebalance -----------------------------------------------------------

    async def _rank(self) -> tuple[list[tuple[str, float]], list[str]]:
        broker = alpaca.client()
        lookback, skip = int(self.spec["lookback_days"]), int(self.spec["skip_days"])
        gate = asyncio.Semaphore(RANK_CONCURRENCY)
        failed: list[str] = []

        async def one(sym: str) -> tuple[str, float] | None:
            async with gate:
                try:
                    candles = rules.closed(await broker.ohlcv(sym, "1d", lookback + 5), "1d")
                except Exception as e:
                    failed.append(f"{sym}: {type(e).__name__}")
                    return None
            score = momentum(candles["close"], lookback, skip)
            if score is None:
                failed.append(f"{sym}: only {len(candles['close'])} days")
                return None
            return sym, score

        universe = [s for s in self.spec["universe"] if s not in self.exclude]
        rows = [r for r in await asyncio.gather(*(one(s) for s in universe)) if r]
        return sorted(rows, key=lambda r: -r[1]), failed

    async def _rebalance(self, month: str) -> str:
        ranking, failed = await self._rank()
        top = int(self.spec["top"])
        if len(ranking) < top:
            return f"skip: only {len(ranking)} stocks ranked ({'; '.join(failed)[:200]})"
        picks = pick(ranking, top)
        append_jsonl(self.rankings_path, {"ts": _utcnow(), "month": month, "picks": picks, "failed": failed,
                                          "ranking": [[s, round(m, 4)] for s, m in ranking]})
        notes = []
        for sym in [s for s in self.paper["positions"] if s not in picks]:
            notes.append(await self._sell(sym, "rotation_exit"))
        unsold = [s for s in self.paper["positions"] if s not in picks]  # a refused sell stays and is retried next month
        self.paper.update(picks=picks, last_rebalance=month,
                          pending=[s for s in picks if s not in self.paper["positions"]])
        self._save()
        notes.append(await self._buy_pending())
        if unsold:
            notes.append(f"could not sell {', '.join(unsold)}")
        return f"rebalanced ({month}): picks {', '.join(picks)}; " + "; ".join(n for n in notes if n)

    async def _quote(self, sym: str) -> float:
        bars = await alpaca.client().bars(sym, limit=1)
        if not bars:
            raise RuntimeError(f"no price for {sym}")
        return float(bars[-1][1])

    async def _buy_pending(self) -> str:
        pending = list(self.paper.get("pending") or [])
        if not pending:
            return ""
        broker = alpaca.client()
        account = await broker.account()
        power = float(account.get("buying_power") or 0) * BUYING_POWER_USE
        target = self.value() / int(self.spec["top"])
        bought, left, notes = [], [], []
        for sym in pending:
            amount = min(target, float(self.paper["cash"]), power)
            if amount < MIN_ORDER_USD:
                left.append(sym)
                continue
            try:
                last = await self._quote(sym)
                qty = math.floor(amount / last * 1e6) / 1e6
                client_id = f"hermes-rot-{sym}-{uuid.uuid4().hex[:10]}"
                order = await broker.wait_filled((await broker.market_order(sym, qty, "buy", client_id))["id"])
            except (AlpacaError, RuntimeError) as e:
                left.append(sym)
                notes.append(f"{sym} not bought: {getattr(e, 'message', e)}"[:120])
                continue
            fill, filled = float(order["filled_avg_price"]), float(order["filled_qty"])
            self.paper["positions"][sym] = {"qty": filled, "entry_price": fill, "last_price": fill, "opened_at": _utcnow(),
                                            "opened_ms": int(time.time() * 1000), "cost": round(filled * fill, 4),
                                            "short_of_target": round(max(0.0, target - filled * fill), 2),
                                            "broker": {"entry_order_id": order["id"], "client_order_id": client_id}}
            self.paper["cash"] = float(self.paper["cash"]) - filled * fill
            power -= filled * fill
            bought.append(f"{sym} ${filled * fill:,.0f}")
        self.paper["pending"] = left
        self._save()
        text = ("bought " + ", ".join(bought)) if bought else ""
        if left:
            text += ("; " if text else "") + f"waiting for cash to buy {', '.join(left)} (Alpaca buying power ${power / BUYING_POWER_USE:,.0f})"
        return "; ".join([text, *notes]) if notes else text

    async def _sell(self, sym: str, reason: str) -> str:
        pos = self.paper["positions"][sym]
        broker = alpaca.client()
        try:
            order = await broker.wait_filled(
                (await broker.market_order(sym, pos["qty"], "sell", f"hermes-rot-{sym}-{uuid.uuid4().hex[:10]}"))["id"])
        except AlpacaError as e:
            return f"{sym} not sold: {e.message}"[:120]
        await self._record_close(sym, float(order["filled_avg_price"]), reason, exit_order_id=order["id"])
        return f"sold {sym}"

    async def _record_close(self, sym: str, exit_price: float, reason: str, **extra) -> None:
        pos = self.paper["positions"].pop(sym)
        gross = pos["qty"] * (exit_price - pos["entry_price"])
        before = float(self.paper["equity"])
        self.paper["equity"] = before + gross
        self.paper["cash"] = float(self.paper["cash"]) + pos["qty"] * exit_price
        trade = {**{k: v for k, v in pos.items() if k != "last_price"}, "asset": f"{sym}{config.SLEEVE_SEP}rotation", "symbol": sym,
                 "direction": "long", "mode": "paper", "closed_at": _utcnow(), "exit_price": exit_price, "exit_reason": reason,
                 "gross_pnl": round(gross, 4), "fees": 0.0, "pnl": round(gross, 4), "pnl_pct": gross / before if before else 0.0,
                 "equity_after": round(self.paper["equity"], 4), **extra}
        append_jsonl(self.trades_path, trade)
        self._save()
