"""24/7 reliability loop: every minute, for each asset, pull data, evaluate that asset's
strategy.yaml, paper trade, log. One heartbeat covers all assets.

With HERMES_REFLECT=llm|hermes|fallback the worker also runs the reflection cycle itself every
HERMES_REFLECT_EVERY_S seconds (default 1800), writing straight to its own state — the same job
the local scheduled task did through remote.py, without the pull/push round trip.

Crypto pairs (BTC/USDT) are simulated fills at the last price. Stocks/ETFs (SPY) are real
orders in the Alpaca *paper* account, traded only during market hours."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import time
import uuid

import aiofiles
import numpy as np
import pandas as pd
from rich.console import Console

from . import config, reflect
from . import strategy as rules
from .adapters import SchemaError, alpaca, check_schema, macro, news, onchain, price, stocks
from .adapters.alpaca import AlpacaError
from .storage import append_jsonl, load_yaml, write_json

console = Console()

CRYPTO_ADAPTERS = {"price": price.fetch, "onchain": onchain.fetch, "news": news.fetch, "macro": macro.fetch}
STOCK_ADAPTERS = {"price": stocks.fetch, "macro": macro.fetch}
REQUIRED = {
    "price": ("source", "closes", "last"),
    "onchain": ("source", "metrics"),
    "news": ("source", "fear_greed"),
    "macro": ("source", "values"),
}
RETRIES = 3
BREAKER_THRESHOLD = 5  # consecutive failed ticks before an adapter is benched
BREAKER_COOLDOWN_S = 300
TICK_S = 60
REFLECT_EVERY_S = 1800
SNAPSHOT_EVERY_S = 300
START_EQUITY = config.CRYPTO_START_EQUITY  # per crypto asset
RSI_PERIOD = 14
# Portfolio limits (goal.yaml `risk:` overrides): crypto pairs move together, so many open
# positions are close to one big bet.
DEFAULT_RISK = {"max_open_positions": 6, "max_open_crypto": 4, "max_exposure_pct": 60}
STOCK_WARMUP_MIN = 15  # no stock entries until RSI is built from today's bars
MIN_ORDER_USD = 1.0  # Alpaca's fractional-order minimum


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def rsi(closes: list[float], period: int = RSI_PERIOD) -> float:
    delta = pd.Series(closes, dtype=float).diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = float(gain.iloc[-1]), float(loss.iloc[-1])
    if last_loss == 0:
        return 50.0 if last_gain == 0 else 100.0
    value = 100 - 100 / (1 + last_gain / last_loss)
    return float(np.round(value, 4))


class Breaker:
    def __init__(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def record(self, ok: bool) -> None:
        if ok:
            self.failures = 0
            return
        self.failures += 1
        if self.failures >= BREAKER_THRESHOLD:
            self.open_until = time.monotonic() + BREAKER_COOLDOWN_S
            self.failures = 0


async def fetch_with_retry(name: str, fn, asset: str) -> dict:
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        try:
            payload = await fn(asset)
        except Exception as e:
            last_exc = e
            if attempt < RETRIES - 1:
                await asyncio.sleep(2**attempt)
            continue
        check_schema(name, payload, REQUIRED[name])  # SchemaError propagates and halts the loop
        return payload
    raise last_exc


class AssetBook:
    """One crypto asset: strategy, simulated paper account, open position, adapter breakers."""

    ADAPTERS = CRYPTO_ADAPTERS

    OHLCV = staticmethod(price.ohlcv)

    def __init__(self, asset: str, start_equity: float = START_EQUITY, costs: tuple[float, float] = (0.0, 0.0)) -> None:
        self.asset = asset
        self.start_equity = start_equity
        self.fee, self.slippage = costs  # fractions per side, applied to simulated fills
        self.min_stop_frac = 0.0  # closest a stop may be (set by make_book from goal.yaml costs)
        self.paths = config.ensure_asset_state(asset)
        self.breakers = {name: Breaker() for name in self.ADAPTERS}
        self.paper = self._load_paper()
        self.lock = asyncio.Lock()  # a tick and a manual buy/sell never interleave on one asset
        self.worker: Worker | None = None  # set by Worker; checks portfolio risk limits before entries

    # --- paper account -------------------------------------------------------

    def _load_paper(self) -> dict:
        if self.paths.paper.exists():
            return json.loads(self.paths.paper.read_text(encoding="utf-8"))
        return {"equity": self.start_equity, "position": None}

    def _save_paper(self) -> None:
        write_json(self.paths.paper, self.paper)

    # --- data ----------------------------------------------------------------

    async def gather(self) -> tuple[dict, dict]:
        results: dict = {}
        status: dict = {}

        async def one(name: str, fn) -> None:
            breaker = self.breakers[name]
            if breaker.is_open():
                status[name] = "circuit_open"
                return
            try:
                results[name] = await fetch_with_retry(name, fn, self.asset)
            except SchemaError:
                raise
            except Exception as e:
                breaker.record(False)
                status[name] = f"error: {type(e).__name__}: {e}"[:200]
                return
            breaker.record(True)
            status[name] = f"ok ({results[name]['source']})"

        await asyncio.gather(*(one(name, fn) for name, fn in self.ADAPTERS.items()))
        return results, status

    # --- strategy rules (shared by crypto and stocks) ------------------------

    @staticmethod
    def _context(data: dict) -> dict:
        return {
            "fear_greed": data.get("news", {}).get("fear_greed"),
            "vix": data.get("macro", {}).get("values", {}).get("vix"),
            "dxy": data.get("macro", {}).get("values", {}).get("dxy"),
            "onchain": data.get("onchain", {}).get("metrics"),
        }

    async def signals(self, strategy: dict) -> dict:
        """RSI, ATR, the EMA cross and the trend filter on the strategy's timeframe, from closed candles."""
        p = rules.params(strategy)
        candles = rules.closed(await self.OHLCV(self.asset, p["timeframe"], rules.bars_needed(p) + 1), p["timeframe"])
        if len(candles["close"]) < rules.RSI_PERIOD + 2:
            raise RuntimeError(f"only {len(candles['close'])} closed {p['timeframe']} candles")
        rsi_now = float(rules.rsi_series(candles["close"])[-1])
        atr_now = float(rules.atr_series(candles["high"], candles["low"], candles["close"])[-1])
        trend = None
        if p["trend_filter"] != "off":
            tc = rules.closed(await self.OHLCV(self.asset, p["trend_filter"], rules.TREND_BARS), p["trend_filter"])
            trend = rules.trend_ok(p["direction"], tc["close"])
        state = target = None
        scale = 1.0
        if p["indicator"] in rules.STATE_INDICATORS:
            state = rules.signal_state(p, candles["close"])
            if state is None:
                raise RuntimeError(f"only {len(candles['close'])} closed {p['timeframe']} candles for {p['indicator']}")
            target = rules.target_direction(p, state)
            if p["target_vol"] > 0:
                scale = rules.vol_scale(p, rules.realized_vol_series(candles["close"], p["timeframe"])[-1])
        return {"p": p, "rsi": round(rsi_now, 4), "atr": None if np.isnan(atr_now) else atr_now, "trend": trend,
                "state": state, "target": target, "vol_scale": scale}

    def _position(self, strategy: dict, fill: float, qty: float, sig: dict, data: dict, side: str | None = None, **extra) -> dict:
        p = sig["p"]
        side = side or p["direction"]
        dist = rules.stop_distance(p, fill, sig.get("atr"), self.min_stop_frac)
        stop, target = rules.levels(p, side, fill, dist)
        return {
            "id": uuid.uuid4().hex[:12],
            "asset": self.asset,
            "direction": side,
            "opened_at": utcnow(),
            "opened_ms": int(time.time() * 1000),
            "entry_price": fill,
            "qty": qty,
            "stop": stop,
            "target": target if math.isfinite(target) else None,  # None: no target (JSON has no infinity)
            "strategy_version": str(strategy["version"]),
            "rsi_at_entry": round(sig["rsi"], 2),
            "rsi_timeframe": p["timeframe"],
            "atr_at_entry": sig.get("atr"),
            "trend_ok": sig.get("trend"),
            "ema_state": sig.get("state"),
            "fees": round(qty * fill * self.fee, 6),  # entry side; the exit side is added on close
            "context": self._context(data),
            **extra,
        }

    def _entry_size(self, sig: dict, price_now: float) -> float:
        dist = rules.stop_distance(sig["p"], price_now, sig.get("atr"), self.min_stop_frac)
        return rules.size(sig["p"], self.paper["equity"], price_now, dist, sig.get("vol_scale", 1.0))

    async def _record_close(self, exit_price: float, reason: str, rsi_value: float | None, **extra) -> None:
        pos = self.paper["position"]
        sign = 1 if pos["direction"] == "long" else -1
        gross = sign * pos["qty"] * (exit_price - pos["entry_price"])
        fees = pos.get("fees", 0.0) + pos["qty"] * exit_price * self.fee
        pnl = gross - fees
        equity_before = self.paper["equity"]
        self.paper["equity"] = equity_before + pnl
        trade = {
            **pos,
            "mode": "paper",
            "closed_at": utcnow(),
            "exit_price": exit_price,
            "exit_reason": reason,
            "rsi_at_exit": round(rsi_value, 2) if rsi_value is not None else None,
            "gross_pnl": round(gross, 4),
            "fees": round(fees, 4),
            "pnl": round(pnl, 4),
            "pnl_pct": pnl / equity_before,
            "equity_after": round(self.paper["equity"], 4),
            **extra,
        }
        async with aiofiles.open(self.paths.trades, "a", encoding="utf-8") as f:
            await f.write(json.dumps(trade) + "\n")
        self.paper["position"] = None
        self._save_paper()

    def _risk_block(self, notional: float) -> str | None:
        """A reason to skip this entry because of portfolio limits; reserves the slot otherwise."""
        return self.worker.risk_check(self, notional) if self.worker else None

    def _release(self) -> None:
        if self.worker:
            self.worker.release(self.asset)

    # --- crypto execution: simulated at the last price, with fees and slippage --------

    def _entry_side(self, sig: dict) -> tuple[str | None, str]:
        """The side to open now, or None with the reason. State signals follow the state (not
        re-entering a side just stopped out of until it changes); rsi enters on its threshold."""
        p = sig["p"]
        if p["indicator"] in rules.STATE_INDICATORS:
            target, blocked = sig.get("target"), self.paper.get("blocked")
            if blocked and target != blocked:
                self.paper.pop("blocked", None)
                blocked = None
            if target is None:
                return None, "no signal"
            if target == blocked:
                return None, f"no signal (stopped out of {target}, waiting for the signal to change)"
            return target, ""
        if not rules.entry_fires(p, sig["rsi"], sig["trend"]):
            if p["trend_filter"] != "off" and sig["trend"] is not True and rules.entry_fires({**p, "trend_filter": "off"}, sig["rsi"], None):
                return None, "no signal (against the trend)"
            return None, "no signal"
        return p["direction"], ""

    async def _close_if_due(self, p: dict, last: float, sig: dict) -> str | None:
        pos = self.paper["position"]
        reason = rules.exit_reason(pos, p, last, sig["rsi"], time.time() * 1000, sig.get("target"))
        if not reason:
            return None
        side = "sell" if pos["direction"] == "long" else "buy"
        await self._record_close(rules.fill(last, side, self.slippage), reason, sig["rsi"])
        if reason == "stop_loss":
            self.paper["blocked"] = pos["direction"]
            self._save_paper()
        return f"closed {pos['direction']} ({reason})"

    async def decide(self, strategy: dict, last: float, sig: dict, data: dict) -> str:
        p = sig["p"]
        pos = self.paper["position"]
        closed = None
        if pos:
            closed = await self._close_if_due(p, last, sig)
            if not closed:
                return f"holding {pos['direction']}"
            if p["indicator"] not in rules.STATE_INDICATORS:
                return closed
        side, why = self._entry_side(sig)
        if not side:
            return closed or why
        fill_price = rules.fill(last, "buy" if side == "long" else "sell", self.slippage)
        qty = self._entry_size(sig, fill_price)
        blocked = self._risk_block(qty * fill_price)
        if blocked:
            return f"{closed}; skip: {blocked}" if closed else f"skip: {blocked}"
        try:
            self.paper["position"] = self._position(strategy, fill_price, qty, sig, data, side)
            self._save_paper()
        finally:
            self._release()
        return f"{closed}; opened {side}" if closed else f"opened {side}"

    # --- one tick ------------------------------------------------------------

    async def tick(self) -> dict:
        """Returns this asset's heartbeat section. Only SchemaError escapes."""
        async with self.lock:
            return await self._tick()

    async def _tick(self) -> dict:
        summary: dict = {"start_equity": self.start_equity}
        try:
            strategy = load_yaml(self.paths.strategy)  # re-read so reflections apply without restart
            summary["strategy_version"] = str(strategy.get("version"))
            gate = await self.gate()
            if gate:
                summary.update(gate)
            else:
                data, status = await self.gather()
                summary["adapters"] = status
                price_data = data.get("price")
                if not price_data:
                    summary["decision"] = "skip: no price data"
                else:
                    sig = await self.signals(strategy)
                    summary.update(
                        last_price=price_data["last"],
                        price_source=price_data["source"],
                        rsi=sig["rsi"],
                        rsi_timeframe=sig["p"]["timeframe"],
                        indicator=sig["p"]["indicator"],
                        ema_state=sig.get("state"),
                        atr=sig["atr"],
                        trend=sig["trend"],
                        decision=await self.decide(strategy, price_data["last"], sig, data),
                    )
        except SchemaError:
            raise
        except Exception as e:
            summary["decision"] = "error"
            summary["error"] = f"{type(e).__name__}: {e}"[:300]
        summary["equity"] = round(self.paper["equity"], 4)
        summary["open_position"] = self.paper["position"] is not None
        return summary

    async def gate(self) -> dict | None:
        """Return a heartbeat section to skip this tick (e.g. market closed), or None to trade."""
        return None

    # --- manual actions from the dashboard -----------------------------------

    async def _quote(self) -> tuple[float, float]:
        """Last price and RSI, for a manual action between ticks."""
        data = await fetch_with_retry("price", self.ADAPTERS["price"], self.asset)
        return data["last"], rsi(data["closes"])

    async def manual_sell(self) -> str:
        async with self.lock:
            pos = self.paper["position"]
            if not pos:
                raise ValueError(f"{self.asset} no tiene una posición abierta")
            return await self._manual_sell(pos)

    async def _manual_sell(self, pos: dict) -> str:
        last, rsi_value = await self._quote()
        exit_fill = rules.fill(last, "sell" if pos["direction"] == "long" else "buy", self.slippage)
        await self._record_close(exit_fill, "manual_close", rsi_value)
        return f"{pos['direction']} cerrado a {exit_fill:g} (ejecución simulada, con comisión)"

    async def set_strategy(self, changes: dict) -> str:
        """Hand-edited strategy settings. They apply from the next tick; an open position's stop
        and target are re-aimed from its entry price, since those are the exits it will use."""
        async with self.lock:
            records = reflect.apply_manual(self.paths, changes, config.is_stock(self.asset))
            if not records:
                return "sin cambios"
            pos = self.paper["position"]
            note = ""
            if pos and any(r["variable"] in ("stop_loss_pct", "stop_atr_mult", "take_profit_r") for r in records):
                p = rules.params(load_yaml(self.paths.strategy))
                dist = rules.stop_distance(p, pos["entry_price"], pos.get("atr_at_entry"), self.min_stop_frac)
                pos["stop"], target = rules.levels(p, pos["direction"], pos["entry_price"], dist)
                pos["target"] = target if math.isfinite(target) else None
                self._save_paper()
                goal_text = f"{pos['target']:g}" if pos["target"] is not None else "sin objetivo"
                note = f"; la posición abierta ahora sale en stop {pos['stop']:g} / objetivo {goal_text}"
            changed = ", ".join(f"{r['variable']} {r['old_value']} → {r['new_value']}" for r in records)
            return f"v{records[0]['from_version']} → v{records[0]['to_version']}: {changed}{note}"

    async def manual_buy(self) -> str:
        async with self.lock:
            if self.paper["position"]:
                raise ValueError(f"{self.asset} ya tiene una posición abierta")
            strategy = load_yaml(self.paths.strategy)
            return await self._manual_buy(strategy, await self.signals(strategy))

    async def _manual_buy(self, strategy: dict, sig: dict) -> str:
        last, _ = await self._quote()
        side = sig.get("target") or ("long" if sig["p"]["direction"] == "both" else sig["p"]["direction"])
        fill_price = rules.fill(last, "buy" if side == "long" else "sell", self.slippage)
        self.paper["position"] = self._position(strategy, fill_price, self._entry_size(sig, fill_price), sig, {}, side, manual=True)
        self._save_paper()
        return f"{side} abierto a {fill_price:g} (ejecución simulada, con comisión)"


class StockBook(AssetBook):
    """A stock/ETF traded with real orders in the Alpaca paper account (long only, fractional
    shares). Fills are Alpaca's; the per-asset virtual account keeps scores comparable.

    Stops, targets and RSI exits are watched by this loop each minute during market hours;
    there is no broker-side stop (Alpaca has no bracket orders for fractional shares). If an
    exit is refused under the pattern-day-trader rule, the position is held and retried on the
    next trading day."""

    ADAPTERS = STOCK_ADAPTERS
    OHLCV = staticmethod(stocks.ohlcv)

    def __init__(self, asset: str, start_equity: float, costs: tuple[float, float] = (0.0, 0.0)) -> None:
        super().__init__(asset, start_equity, costs)
        self.session: dict = {}

    async def gate(self) -> dict | None:
        self.session = await alpaca.client().session()
        if not self.session["is_open"]:
            return {"market": "closed", "decision": f"market closed · opens {self.session['next_open'][:16].replace('T', ' ')} ET"}
        return None

    async def decide(self, strategy: dict, last: float, sig: dict, data: dict) -> str:
        p, rsi_value = sig["p"], sig["rsi"]
        if p["direction"] != "long":
            raise ValueError("stocks are long-only here (shorting is disabled on the Alpaca account)")
        broker = alpaca.client()
        held = await broker.position(self.asset)
        pos = self.paper["position"]

        if pos and held is None:
            return await self._reconcile_external_close(last, rsi_value)
        if pos:
            if pos.get("exit_blocked_on") == self.session["date"]:
                return "holding long (exit blocked by day-trade rule until next session)"
            reason = rules.exit_reason(pos, p, last, rsi_value, time.time() * 1000, sig.get("target"))
            if not reason:
                return "holding long"
            if reason == "stop_loss":
                self.paper["blocked"] = "long"
            return await self._sell(reason, rsi_value)

        if held is not None:
            return "skip: Alpaca holds a position this worker didn't open"
        minutes = self.session.get("minutes_since_open")
        if minutes is not None and minutes < STOCK_WARMUP_MIN:
            return f"warming up · entries from {STOCK_WARMUP_MIN} min after the open"
        side, why = self._entry_side(sig)
        if not side:
            return why
        blocked = self._risk_block(self._entry_size(sig, last) * last)
        if blocked:
            return f"skip: {blocked}"
        try:
            return await self._buy(strategy, last, sig, data)
        finally:
            self._release()

    async def _require_open_market(self) -> None:
        self.session = await alpaca.client().session()
        if not self.session["is_open"]:
            opens = self.session["next_open"][:16].replace("T", " ")
            raise ValueError(f"el mercado está cerrado: {self.asset} se puede operar desde {opens} ET")

    async def _manual_sell(self, pos: dict) -> str:
        await self._require_open_market()
        _, rsi_value = await self._quote()
        result = await self._sell("manual_close", rsi_value)
        if result.startswith("closed"):
            return result.replace("closed long (manual_close) @", "long vendido en Alpaca a")
        raise ValueError(result)

    async def _manual_buy(self, strategy: dict, sig: dict) -> str:
        if sig["p"]["direction"] != "long":
            raise ValueError("las acciones y los ETFs solo operan en long")
        await self._require_open_market()
        if await alpaca.client().position(self.asset) is not None:
            raise ValueError(f"Alpaca ya tiene {self.asset} fuera de este worker")
        last, _ = await self._quote()
        result = await self._buy(strategy, last, sig, {}, manual=True)
        if not result.startswith("opened"):
            raise ValueError(result)
        return result.replace("opened long", "long comprado en Alpaca:").replace(" @ ", " a ")

    async def _buy(self, strategy: dict, last: float, sig: dict, data: dict, **extra) -> str:
        qty = math.floor(self._entry_size(sig, last) * 1e6) / 1e6
        if qty * last < MIN_ORDER_USD:
            return f"skip: order ${qty * last:.2f} is below Alpaca's ${MIN_ORDER_USD:.0f} minimum"
        broker = alpaca.client()
        client_id = f"hermes-{config.asset_slug(self.asset)}-{uuid.uuid4().hex[:10]}"
        try:
            order = await broker.wait_filled((await broker.market_order(self.asset, qty, "buy", client_id))["id"])
        except AlpacaError as e:
            return f"entry rejected: {e.message}"[:200]
        fill, filled_qty = float(order["filled_avg_price"]), float(order["filled_qty"])
        self.paper["position"] = self._position(
            strategy, fill, filled_qty, sig, data, broker={"entry_order_id": order["id"], "client_order_id": client_id}, **extra
        )
        self._save_paper()
        return f"opened long {filled_qty:g} @ {fill:.2f}"

    async def _sell(self, reason: str, rsi_value: float) -> str:
        broker = alpaca.client()
        try:
            order = await broker.close_position(self.asset)
        except AlpacaError as e:
            text = e.message.lower()
            if e.status == 403 and ("day trad" in text or "pattern" in text):
                self.paper["position"]["exit_blocked_on"] = self.session["date"]
                self._save_paper()
                return f"exit ({reason}) blocked by day-trade rule — holding until next session"
            if e.status == 404:
                return "position already gone at Alpaca — reconciling next tick"
            raise
        order = await broker.wait_filled(order["id"])
        await self._record_close(float(order["filled_avg_price"]), reason, rsi_value, exit_order_id=order["id"])
        return f"closed long ({reason}) @ {float(order['filled_avg_price']):.2f}"

    async def _reconcile_external_close(self, last: float, rsi_value: float) -> str:
        """Our record says open, Alpaca says flat: someone closed it outside the worker."""
        pos = self.paper["position"]
        fill = await alpaca.client().last_sell_fill(self.asset, pos["opened_at"])
        if fill:
            await self._record_close(float(fill["filled_avg_price"]), "external_close", rsi_value, exit_order_id=fill["id"])
            return "position was closed outside the worker — recorded"
        await self._record_close(last, "missing_at_broker", rsi_value)
        return "position missing at Alpaca and no sell found — recorded at last price"


def make_book(asset: str, goal: dict) -> AssetBook:
    equity = config.start_equity(asset, goal)
    stock = config.is_stock(asset)
    # Stocks fill at Alpaca (real prices, commission-free), so only crypto gets simulated costs
    costs = rules.costs(goal, stock) if not stock else (0.0, 0.0)
    book = StockBook(asset, equity, costs) if stock else AssetBook(asset, equity, costs)
    book.min_stop_frac = rules.min_stop_frac(goal, stock)
    return book


class Worker:
    def __init__(self, assets: list[str], goal: dict) -> None:
        self.books = [make_book(asset, goal) for asset in assets]
        for book in self.books:
            book.worker = self
        self._reserved: set[str] = set()  # assets between a passed risk check and their position being recorded
        self.loop: asyncio.AbstractEventLoop | None = None  # set in run(); the state server posts actions to it
        self.reflection: dict = {}  # last reflection cycle, reported in the heartbeat
        self.last_prices: dict[str, float] = {}  # stocks keep their last price while the market is closed
        self._issues: dict[str, str | None] = {}  # asset -> the issue last logged, so each is logged once
        self._last_snapshot = float("-inf")  # first snapshot on the first tick
        self.reflect_every_s = float(REFLECT_EVERY_S)

    # --- dashboard records ---------------------------------------------------

    def event(self, kind: str, text: str, asset: str | None = None, level: str = "info") -> None:
        """One line in the activity feed. Trades and strategy changes aren't logged here: the
        dashboard reads them from trades.jsonl and hypotheses.jsonl."""
        try:
            append_jsonl(config.EVENTS_FILE, {"ts": utcnow(), "kind": kind, "level": level, "asset": asset, "text": text})
        except OSError as e:
            console.log(f"[red]event log failed:[/] {e}")

    def _log_issue(self, asset: str, s: dict) -> None:
        decision = s.get("decision") or ""
        if s.get("error"):
            issue, level = s["error"], "error"
        elif decision.startswith(("skip:", "entry rejected")) or "blocked" in decision:
            issue, level = decision, "warn"
        else:
            issue, level = None, "info"
        previous = self._issues.get(asset)
        if issue and issue != previous:
            self.event("issue", issue, asset, level)
        elif not issue and previous:
            self.event("recovered", "back to normal", asset)
        self._issues[asset] = issue

    def _snapshot(self) -> None:
        """Capital per category every SNAPSHOT_EVERY_S, for the equity curve and drawdown."""
        now = time.monotonic()
        if now - self._last_snapshot < SNAPSHOT_EVERY_S:
            return
        self._last_snapshot = now
        kinds = {k: {"start": 0.0, "realized": 0.0, "unrealized": 0.0, "invested": 0.0, "open": 0} for k in ("crypto", "stock")}
        for book in self.books:
            agg = kinds["stock" if config.is_stock(book.asset) else "crypto"]
            pos, last = book.paper["position"], self.last_prices.get(book.asset)
            agg["start"] += book.start_equity
            agg["realized"] += book.paper["equity"]
            if pos:
                price_now = last if last is not None else pos["entry_price"]
                sign = 1 if pos["direction"] == "long" else -1
                agg["unrealized"] += sign * pos["qty"] * (price_now - pos["entry_price"])
                agg["invested"] += pos["qty"] * price_now
                agg["open"] += 1
        record = {"ts": utcnow(), **{k: {f: round(v, 4) for f, v in agg.items()} for k, agg in kinds.items()}}
        try:
            append_jsonl(config.EQUITY_FILE, record)
        except OSError as e:
            console.log(f"[red]equity snapshot failed:[/] {e}")

    def risk_check(self, book: AssetBook, notional: float) -> str | None:
        """Portfolio limits for a new entry (goal.yaml `risk:`). Returns why it's refused, or
        None and reserves the slot until release()."""
        risk = {**DEFAULT_RISK, **(load_yaml(config.GOAL_FILE).get("risk") or {})}
        busy = [b for b in self.books if b is not book and (b.paper["position"] or b.asset in self._reserved)]
        if len(busy) >= int(risk["max_open_positions"]):
            return f"risk limit: {len(busy)} positions open (max {risk['max_open_positions']})"
        if not config.is_stock(book.asset):
            crypto = [b for b in busy if not config.is_stock(b.asset)]
            if len(crypto) >= int(risk["max_open_crypto"]):
                return f"risk limit: {len(crypto)} crypto positions open (max {risk['max_open_crypto']})"
        invested = sum(b.paper["position"]["qty"] * b.paper["position"]["entry_price"] for b in busy if b.paper["position"])
        equity = sum(b.paper["equity"] for b in self.books)
        exposure = (invested + notional) / equity * 100 if equity else 0
        if exposure > float(risk["max_exposure_pct"]):
            return f"risk limit: exposure would reach {exposure:.0f}% (max {risk['max_exposure_pct']}%)"
        self._reserved.add(book.asset)
        return None

    def release(self, asset: str) -> None:
        self._reserved.discard(asset)

    def book(self, asset: str) -> AssetBook:
        for book in self.books:
            if book.asset == asset:
                return book
        raise ValueError(f"{asset} no lo opera este worker")

    async def set_strategy_all(self, changes: dict, kind: str = "all") -> str:
        """The same hand-edited settings on every asset of a kind (all, crypto or stock)."""
        if kind not in ("all", "crypto", "stock"):
            raise ValueError("kind debe ser all, crypto o stock")
        if not isinstance(changes, dict) or not changes:
            raise ValueError("no hay cambios que guardar")
        changed, unchanged, failed = [], 0, []
        for book in list(self.books):
            if kind != "all" and kind != ("stock" if config.is_stock(book.asset) else "crypto"):
                continue
            try:
                message = await book.set_strategy(changes)
            except ValueError as e:
                failed.append(f"{book.asset}: {e}")
                continue
            if message == "sin cambios":
                unchanged += 1
            else:
                changed.append(book.asset)
        if not changed and failed:
            raise ValueError("; ".join(failed)[:300])
        text = f"{len(changed)} activos actualizados" + (f", {unchanged} ya lo tenían" if unchanged else "")
        return text + (f"; no se pudo en {'; '.join(failed)}" if failed else "")

    async def add_asset(self, asset: str, buy: bool) -> str:
        """Start trading a new asset: check it has prices, record it in goal.yaml, open a book."""
        asset = config.normalize_asset(asset, "stock" if config.is_stock(asset) else "crypto")
        if any(book.asset == asset for book in self.books):
            raise ValueError(f"{asset} ya se está operando")
        adapter = stocks.fetch if config.is_stock(asset) else price.fetch
        try:
            await adapter(asset)
        except Exception as e:
            raise ValueError(f"no hay datos de precio para {asset}: {e}"[:300]) from e
        config.add_goal_asset(config.GOAL_FILE, asset)
        book = make_book(asset, load_yaml(config.GOAL_FILE))
        book.worker = self
        self.books.append(book)
        message = f"{asset} añadido con una cuenta simulada de ${book.start_equity:,.0f}".replace(",", ".")
        self.event("asset_added", message, asset)
        if buy:
            try:
                message += "; " + await book.manual_buy()
            except ValueError as e:
                message += f"; no se compró: {e}"
        return message

    def _heartbeat(self, state: str, assets: dict | None = None, **fields) -> None:
        if self.reflection:
            fields.setdefault("reflection", self.reflection)
        write_json(
            config.HEARTBEAT_FILE,
            {"ts": utcnow(), "state": state, "mode": "paper", **fields, "assets": assets or {}},
        )

    # --- reflection ----------------------------------------------------------

    async def _reflect_book(self, book: AssetBook, goal: dict, mode: str) -> str:
        # The model call runs in a thread without the book's lock, so ticks keep trading;
        # only the write takes the lock, and it gives up if the strategy moved meanwhile.
        proposal = await asyncio.to_thread(reflect.propose, book.asset, goal, mode, False)
        if isinstance(proposal, str):
            return proposal
        async with book.lock:
            current = load_yaml(book.paths.strategy)
            if str(current.get("version")) != str(proposal.strategy.get("version")):
                return "strategy changed while reflecting — retrying next cycle."
            return reflect.apply_proposal(proposal, mode)

    async def reflect_once(self, mode: str) -> None:
        goal = load_yaml(config.GOAL_FILE)  # re-read: the dashboard may have changed it
        reports = {}
        for book in list(self.books):  # one at a time, like the local task did
            try:
                reports[book.asset] = await self._reflect_book(book, goal, mode)
            except Exception as e:  # one asset's failure must not block the others
                reports[book.asset] = f"FAILED — {type(e).__name__}: {e}"[:300]
                self.event("reflect_error", f"{type(e).__name__}: {e}"[:300], book.asset, "error")
            console.log(f"[magenta]reflect[/] {book.asset}: {reports[book.asset]}")
        self.reflection = {"ts": utcnow(), "mode": mode, "every_s": self.reflect_every_s, "reports": reports}

    async def reflect_forever(self, mode: str, every_s: float) -> None:
        console.print(f"[bold]Reflection on[/] mode={mode} every {every_s:g}s")
        self.reflect_every_s = every_s
        while True:
            started = time.monotonic()
            try:
                await self.reflect_once(mode)
            except Exception as e:  # e.g. goal.yaml unreadable; try again next cycle
                console.log(f"[red]reflection cycle failed:[/] {type(e).__name__}: {e}")
                self.reflection = {"ts": utcnow(), "mode": mode, "every_s": every_s, "error": f"{type(e).__name__}: {e}"[:300]}
                self.event("reflect_error", self.reflection["error"], None, "error")
            await asyncio.sleep(max(0.0, every_s - (time.monotonic() - started)))

    def _reflect_settings(self) -> tuple[str, float] | None:
        mode = config.env("HERMES_REFLECT", "off").lower()
        if mode == "off":
            return None
        if mode not in reflect.MODES:
            console.print(f"[red]HERMES_REFLECT={mode!r} not understood[/] (off, {', '.join(reflect.MODES)}) — reflection off.")
            return None
        if mode == "llm" and not config.env("LLM_API_KEY"):
            console.print("[red]HERMES_REFLECT=llm but LLM_API_KEY is not set[/] — reflection off.")
            self.reflection = {"ts": utcnow(), "mode": mode, "error": "LLM_API_KEY is not set"}
            return None
        return mode, float(config.env("HERMES_REFLECT_EVERY_S", str(REFLECT_EVERY_S)))

    async def tick(self) -> None:
        summaries = await asyncio.gather(*(book.tick() for book in self.books))
        by_asset = {book.asset: s for book, s in zip(self.books, summaries)}
        self._heartbeat("running", by_asset)
        for asset, s in by_asset.items():
            if s.get("last_price") is not None:
                self.last_prices[asset] = s["last_price"]
            self._log_issue(asset, s)
        self._snapshot()
        for asset, s in by_asset.items():
            if s.get("error"):
                console.log(f"[red]{asset} tick failed:[/] {s['error']}")
            elif s.get("market") == "closed":
                continue  # one line per minute per closed-market stock is noise
            else:
                console.log(
                    f"{asset} price={s.get('last_price', '—')} rsi={s.get('rsi', '—')} "
                    f"v{s.get('strategy_version')} → {s['decision']}"
                )

    async def run(self, once: bool = False) -> None:
        self.loop = asyncio.get_running_loop()
        names = ", ".join(book.asset for book in self.books)
        console.print(f"[bold]Booting hermes-trading worker[/] assets={names} mode=paper")
        settings = None if once else self._reflect_settings()
        reflector = asyncio.create_task(self.reflect_forever(*settings)) if settings else None
        try:
            while True:
                started = time.monotonic()
                try:
                    await self.tick()
                except SchemaError as e:
                    self._heartbeat("halted", error=f"SchemaError: {e}")
                    raise
                except Exception as e:
                    console.log(f"[red]tick failed:[/] {type(e).__name__}: {e}")
                    self._heartbeat("error", error=f"{type(e).__name__}: {e}"[:300])
                if once:
                    return
                await asyncio.sleep(max(0.0, TICK_S - (time.monotonic() - started)))
        finally:
            if reflector:
                reflector.cancel()
            await price.close()
            await alpaca.close()
