"""24/7 reliability loop: every minute, for each asset, pull data, evaluate that asset's
strategy.yaml, paper trade, log. One heartbeat covers all assets.

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
from .adapters import SchemaError, alpaca, check_schema, macro, news, onchain, price, stocks
from .adapters.alpaca import AlpacaError
from .storage import load_yaml, write_json

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
START_EQUITY = config.CRYPTO_START_EQUITY  # per crypto asset
RSI_PERIOD = 14
RSI_EXIT = {"long": 70.0, "short": 30.0}
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

    def __init__(self, asset: str, start_equity: float = START_EQUITY) -> None:
        self.asset = asset
        self.start_equity = start_equity
        self.paths = config.ensure_asset_state(asset)
        self.breakers = {name: Breaker() for name in self.ADAPTERS}
        self.paper = self._load_paper()
        self.lock = asyncio.Lock()  # a tick and a manual buy/sell never interleave on one asset

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

    @staticmethod
    def _check_strategy(strategy: dict) -> None:
        entry = strategy["entry"]
        if entry["indicator"] != "rsi":
            raise ValueError(f"unsupported indicator {entry['indicator']!r} (only 'rsi')")
        if entry["direction"] not in RSI_EXIT:
            raise ValueError(f"unsupported direction {entry['direction']!r}")

    @staticmethod
    def _exit_reason(pos: dict, last: float, rsi_value: float) -> str | None:
        is_long = pos["direction"] == "long"
        if (last <= pos["stop"]) if is_long else (last >= pos["stop"]):
            return "stop_loss"
        if (last >= pos["target"]) if is_long else (last <= pos["target"]):
            return "take_profit"
        if (rsi_value >= RSI_EXIT["long"]) if is_long else (rsi_value <= RSI_EXIT["short"]):
            return "rsi_exit"
        return None

    @staticmethod
    def _entry_fires(strategy: dict, rsi_value: float) -> bool:
        entry = strategy["entry"]
        threshold = float(entry["threshold"])
        return rsi_value < threshold if entry["direction"] == "long" else rsi_value > threshold

    def _size(self, strategy: dict, price_now: float) -> float:
        stop_frac = float(strategy["stop_loss_pct"]) / 100
        # position_size_r = % of equity put at risk if the stop is hit (0.5 -> 0.5%)
        risk_frac = float(strategy["position_size_r"]) / 100
        equity = self.paper["equity"]
        return min((equity * risk_frac) / (price_now * stop_frac), equity / price_now)  # no leverage

    def _position(self, strategy: dict, fill: float, qty: float, rsi_value: float, data: dict, **extra) -> dict:
        direction = strategy["entry"]["direction"]
        sign = 1 if direction == "long" else -1
        stop_frac = float(strategy["stop_loss_pct"]) / 100
        take_profit_r = float(strategy.get("take_profit_r", 2.0))
        return {
            "id": uuid.uuid4().hex[:12],
            "asset": self.asset,
            "direction": direction,
            "opened_at": utcnow(),
            "entry_price": fill,
            "qty": qty,
            "stop": fill * (1 - sign * stop_frac),
            "target": fill * (1 + sign * stop_frac * take_profit_r),
            "strategy_version": str(strategy["version"]),
            "rsi_at_entry": round(rsi_value, 2),
            "context": self._context(data),
            **extra,
        }

    async def _record_close(self, exit_price: float, reason: str, rsi_value: float | None, **extra) -> None:
        pos = self.paper["position"]
        sign = 1 if pos["direction"] == "long" else -1
        pnl = sign * pos["qty"] * (exit_price - pos["entry_price"])
        equity_before = self.paper["equity"]
        self.paper["equity"] = equity_before + pnl
        trade = {
            **pos,
            "mode": "paper",
            "closed_at": utcnow(),
            "exit_price": exit_price,
            "exit_reason": reason,
            "rsi_at_exit": round(rsi_value, 2) if rsi_value is not None else None,
            "pnl": round(pnl, 4),
            "pnl_pct": pnl / equity_before,
            "equity_after": round(self.paper["equity"], 4),
            **extra,
        }
        async with aiofiles.open(self.paths.trades, "a", encoding="utf-8") as f:
            await f.write(json.dumps(trade) + "\n")
        self.paper["position"] = None
        self._save_paper()

    # --- crypto execution: simulated at the last price -----------------------

    async def decide(self, strategy: dict, last: float, rsi_value: float, data: dict) -> str:
        self._check_strategy(strategy)
        pos = self.paper["position"]
        if pos:
            reason = self._exit_reason(pos, last, rsi_value)
            if reason:
                await self._record_close(last, reason, rsi_value)
                return f"closed {pos['direction']} ({reason})"
            return f"holding {pos['direction']}"
        if self._entry_fires(strategy, rsi_value):
            self.paper["position"] = self._position(strategy, last, self._size(strategy, last), rsi_value, data)
            self._save_paper()
            return f"opened {strategy['entry']['direction']}"
        return "no signal"

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
                    rsi_value = rsi(price_data["closes"])
                    summary.update(
                        last_price=price_data["last"],
                        price_source=price_data["source"],
                        rsi=rsi_value,
                        decision=await self.decide(strategy, price_data["last"], rsi_value, data),
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
        await self._record_close(last, "manual_close", rsi_value)
        return f"{pos['direction']} cerrado a {last:g} (ejecución simulada)"

    async def set_strategy(self, changes: dict) -> str:
        """Hand-edited strategy settings. They apply from the next tick; an open position's stop
        and target are re-aimed from its entry price, since those are the exits it will use."""
        async with self.lock:
            records = reflect.apply_manual(self.paths, changes, config.is_stock(self.asset))
            if not records:
                return "sin cambios"
            pos = self.paper["position"]
            note = ""
            if pos and any(r["variable"] in ("stop_loss_pct", "take_profit_r") for r in records):
                strategy = load_yaml(self.paths.strategy)
                sign = 1 if pos["direction"] == "long" else -1
                stop_frac = float(strategy["stop_loss_pct"]) / 100
                pos["stop"] = pos["entry_price"] * (1 - sign * stop_frac)
                pos["target"] = pos["entry_price"] * (1 + sign * stop_frac * float(strategy.get("take_profit_r", 2.0)))
                self._save_paper()
                note = f"; la posición abierta ahora sale en stop {pos['stop']:g} / objetivo {pos['target']:g}"
            changed = ", ".join(f"{r['variable']} {r['old_value']} → {r['new_value']}" for r in records)
            return f"v{records[0]['from_version']} → v{records[0]['to_version']}: {changed}{note}"

    async def manual_buy(self) -> str:
        async with self.lock:
            if self.paper["position"]:
                raise ValueError(f"{self.asset} ya tiene una posición abierta")
            strategy = load_yaml(self.paths.strategy)
            self._check_strategy(strategy)
            return await self._manual_buy(strategy)

    async def _manual_buy(self, strategy: dict) -> str:
        last, rsi_value = await self._quote()
        self.paper["position"] = self._position(strategy, last, self._size(strategy, last), rsi_value, {}, manual=True)
        self._save_paper()
        return f"{strategy['entry']['direction']} abierto a {last:g} (ejecución simulada)"


class StockBook(AssetBook):
    """A stock/ETF traded with real orders in the Alpaca paper account (long only, fractional
    shares). Fills are Alpaca's; the per-asset virtual account keeps scores comparable.

    Stops, targets and RSI exits are watched by this loop each minute during market hours;
    there is no broker-side stop (Alpaca has no bracket orders for fractional shares). If an
    exit is refused under the pattern-day-trader rule, the position is held and retried on the
    next trading day."""

    ADAPTERS = STOCK_ADAPTERS

    def __init__(self, asset: str, start_equity: float) -> None:
        super().__init__(asset, start_equity)
        self.session: dict = {}

    async def gate(self) -> dict | None:
        self.session = await alpaca.client().session()
        if not self.session["is_open"]:
            return {"market": "closed", "decision": f"market closed · opens {self.session['next_open'][:16].replace('T', ' ')} ET"}
        return None

    async def decide(self, strategy: dict, last: float, rsi_value: float, data: dict) -> str:
        self._check_strategy(strategy)
        if strategy["entry"]["direction"] != "long":
            raise ValueError("stocks are long-only here (shorting is disabled on the Alpaca account)")
        broker = alpaca.client()
        held = await broker.position(self.asset)
        pos = self.paper["position"]

        if pos and held is None:
            return await self._reconcile_external_close(last, rsi_value)
        if pos:
            if pos.get("exit_blocked_on") == self.session["date"]:
                return "holding long (exit blocked by day-trade rule until next session)"
            reason = self._exit_reason(pos, last, rsi_value)
            if not reason:
                return "holding long"
            return await self._sell(reason, rsi_value)

        if held is not None:
            return "skip: Alpaca holds a position this worker didn't open"
        minutes = self.session.get("minutes_since_open")
        if minutes is not None and minutes < STOCK_WARMUP_MIN:
            return f"warming up · entries from {STOCK_WARMUP_MIN} min after the open"
        if not self._entry_fires(strategy, rsi_value):
            return "no signal"
        return await self._buy(strategy, last, rsi_value, data)

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

    async def _manual_buy(self, strategy: dict) -> str:
        if strategy["entry"]["direction"] != "long":
            raise ValueError("las acciones y los ETFs solo operan en long")
        await self._require_open_market()
        if await alpaca.client().position(self.asset) is not None:
            raise ValueError(f"Alpaca ya tiene {self.asset} fuera de este worker")
        last, rsi_value = await self._quote()
        result = await self._buy(strategy, last, rsi_value, {}, manual=True)
        if not result.startswith("opened"):
            raise ValueError(result)
        return result.replace("opened long", "long comprado en Alpaca:").replace(" @ ", " a ")

    async def _buy(self, strategy: dict, last: float, rsi_value: float, data: dict, **extra) -> str:
        qty = math.floor(self._size(strategy, last) * 1e6) / 1e6
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
            strategy, fill, filled_qty, rsi_value, data, broker={"entry_order_id": order["id"], "client_order_id": client_id}, **extra
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
    return StockBook(asset, equity) if config.is_stock(asset) else AssetBook(asset, equity)


class Worker:
    def __init__(self, assets: list[str], goal: dict) -> None:
        self.books = [make_book(asset, goal) for asset in assets]
        self.loop: asyncio.AbstractEventLoop | None = None  # set in run(); the state server posts actions to it

    def book(self, asset: str) -> AssetBook:
        for book in self.books:
            if book.asset == asset:
                return book
        raise ValueError(f"{asset} no lo opera este worker")

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
        self.books.append(book)
        message = f"{asset} añadido con una cuenta simulada de ${book.start_equity:,.0f}".replace(",", ".")
        if buy:
            try:
                message += "; " + await book.manual_buy()
            except ValueError as e:
                message += f"; no se compró: {e}"
        return message

    def _heartbeat(self, state: str, assets: dict | None = None, **fields) -> None:
        write_json(
            config.HEARTBEAT_FILE,
            {"ts": utcnow(), "state": state, "mode": "paper", **fields, "assets": assets or {}},
        )

    async def tick(self) -> None:
        summaries = await asyncio.gather(*(book.tick() for book in self.books))
        by_asset = {book.asset: s for book, s in zip(self.books, summaries)}
        self._heartbeat("running", by_asset)
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
            await price.close()
            await alpaca.close()
