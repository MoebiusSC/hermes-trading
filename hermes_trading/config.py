"""Paths and .env loading. No keys are hardcoded — everything comes from the environment.

State layout (the Railway volume mirrors this):
  state/goal.yaml                    shared goal + the list of assets to trade
  state/strategy.template.yaml       starting strategy for an asset seen for the first time
  state/heartbeat.json               worker liveness, one section per asset
  state/equity.jsonl                 capital snapshot every 5 minutes (realized, unrealized, invested)
  state/events.jsonl                 worker issues, recoveries, reflection failures, added assets
  state/assets/<BASE-QUOTE>/         per-asset: strategy.yaml, trades.jsonl, hypotheses.jsonl,
                                     paper_account.json, history/
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("HERMES_TRADING_STATE", ROOT / "state"))
# Baked-in copy of state/ used to populate an empty persistent volume on first boot
STATE_SEED = ROOT / "state_seed"

GOAL_FILE = STATE / "goal.yaml"
STRATEGY_TEMPLATE = STATE / "strategy.template.yaml"
HEARTBEAT_FILE = STATE / "heartbeat.json"
EQUITY_FILE = STATE / "equity.jsonl"  # capital snapshots for the dashboard's equity curve
EVENTS_FILE = STATE / "events.jsonl"  # worker issues and actions for the dashboard's activity feed
ASSETS_DIR = STATE / "assets"

# Files the single-asset version kept at the state root; run.py migrates them on boot.
LEGACY_FILES = ("strategy.yaml", "trades.jsonl", "hypotheses.jsonl", "paper_account.json")


@dataclass(frozen=True)
class AssetPaths:
    asset: str
    root: Path

    @property
    def strategy(self) -> Path:
        return self.root / "strategy.yaml"

    @property
    def trades(self) -> Path:
        return self.root / "trades.jsonl"

    @property
    def hypotheses(self) -> Path:
        return self.root / "hypotheses.jsonl"

    @property
    def history(self) -> Path:
        return self.root / "history"

    @property
    def paper(self) -> Path:
        return self.root / "paper_account.json"


def asset_slug(asset: str) -> str:
    return asset.replace("/", "-")


def asset_paths(asset: str) -> AssetPaths:
    return AssetPaths(asset, ASSETS_DIR / asset_slug(asset))


def ensure_asset_state(asset: str) -> AssetPaths:
    """Create an asset's directory, starting its strategy from the template if it has none."""
    paths = asset_paths(asset)
    paths.history.mkdir(parents=True, exist_ok=True)
    if not paths.strategy.exists():
        shutil.copy2(STRATEGY_TEMPLATE, paths.strategy)
    return paths


CRYPTO_START_EQUITY = 10_000.0
DEFAULT_STOCK_EQUITY = 500.0


def is_stock(asset: str) -> bool:
    """Crypto pairs are written BASE/QUOTE (BTC/USDT); stocks and ETFs are bare tickers (SPY)."""
    return "/" not in asset


def start_equity(asset: str, goal: dict) -> float:
    """Each asset's virtual paper account. Stocks share the real Alpaca paper balance."""
    if is_stock(asset):
        # Per-ticker overrides keep accounts opened under an older default measured correctly
        overrides = goal.get("stock_equity_overrides") or {}
        if asset in overrides:
            return float(overrides[asset])
        return float(goal.get("stock_equity_per_asset", DEFAULT_STOCK_EQUITY))
    return CRYPTO_START_EQUITY


def goal_assets(goal: dict) -> list[str]:
    assets = goal.get("assets") or ([goal["asset"]] if goal.get("asset") else [])
    if not assets:
        raise ValueError("goal.yaml needs an `assets:` list")
    return [str(a) for a in assets]


def normalize_asset(raw: str, kind: str) -> str:
    """User input to the goal.yaml form: crypto 'ada' -> 'ADA/USDT', ETF 'gld' -> 'GLD'."""
    symbol = raw.strip().upper().replace("-", "/")
    if kind == "crypto":
        symbol = symbol if "/" in symbol else f"{symbol}/USDT"
        if not all(part.isalnum() for part in symbol.split("/")) or symbol.count("/") != 1:
            raise ValueError(f"{raw!r} no parece un símbolo de cripto")
        return symbol
    if not symbol.isalnum() or len(symbol) > 6:
        raise ValueError(f"{raw!r} no parece un ticker de acción o ETF")
    return symbol


def add_goal_asset(path: Path, asset: str) -> None:
    """Append an asset to goal.yaml's `assets:` list, keeping the file's comments intact."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("assets:"))
    end = start + 1
    while end < len(lines) and lines[end].lstrip().startswith("- "):
        end += 1
    listed = {line.split("#")[0].strip().lstrip("- ").strip().strip("\"'") for line in lines[start + 1:end]}
    if asset in listed:
        return
    lines.insert(end, f'  - "{asset}"\n')
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader. Real environment variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default
