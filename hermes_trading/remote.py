"""Sync with the Railway volume — the source of truth for the deployed worker's state.

  python -m hermes_trading.remote pull                 volume → remote_state/ (a fresh mirror)
  python -m hermes_trading.remote push [ASSET ...]     reflection files → volume (default: all assets)
  python -m hermes_trading.remote push-goal            local state/goal.yaml → volume (then redeploy)
  python -m hermes_trading.remote reflect --fallback   pull, reflect every asset, push the ones that changed
  python -m hermes_trading.remote reflect --hermes     same, with Hermes proposing the changes

The worker re-reads each asset's strategy.yaml every tick, so a push takes effect within a
minute; no redeploy. goal.yaml (the asset list) is read at boot, so push-goal needs a redeploy.
Only reflection outputs are pushed (history/, hypotheses.jsonl, strategy.yaml). Files the worker
writes (trades.jsonl, paper_account.json, heartbeat.json) are pull-only, so the two never race.

Uses the Railway CLI's volume file commands, which need an SSH key registered with Railway.
Overrides: RAILWAY_CMD (CLI path), RAILWAY_VOLUME (volume name).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .storage import load_yaml

REMOTE_DIR = config.ROOT / "remote_state"
ROOT_FILES = ("goal.yaml", "strategy.template.yaml", "heartbeat.json")
ASSET_FILES = ("strategy.yaml", "trades.jsonl", "hypotheses.jsonl", "paper_account.json")
# CLI failures worth retrying: dropped or reset connections to Railway's API
TRANSIENT_ERRORS = ("connection error", "error sending request", "os error 10054", "connection reset", "timed out")


def railway_cmd() -> list[str]:
    override = config.env("RAILWAY_CMD")
    if override:
        return shlex.split(override)
    found = shutil.which("railway")
    if found:
        return [found]
    npm_exe = Path(os.environ.get("APPDATA", "")) / "npm/node_modules/@railway/cli/bin/railway.exe"
    if npm_exe.exists():
        return [str(npm_exe)]
    raise SystemExit("Railway CLI not found — install it with `npm install -g @railway/cli`.")


def _volume_files(*args: str) -> str:
    volume = config.env("RAILWAY_VOLUME", "hermes-trading-volume")
    cmd = [*railway_cmd(), "volume", "files", "--volume", volume, *args]
    # Railway's API drops connections now and then; retry those instead of failing the whole sync
    for attempt in range(6):  # backoff totals ~30s, enough to ride out a DNS or Wi-Fi blip
        proc = subprocess.run(cmd, cwd=config.ROOT, capture_output=True, text=True, encoding="utf-8")
        output = (proc.stderr or proc.stdout).lower()
        if proc.returncode == 0 or not any(m in output for m in TRANSIENT_ERRORS):
            break
        time.sleep(2 ** attempt)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-500:]
        raise SystemExit(f"railway volume files {' '.join(args)} failed:\n{detail}")
    return proc.stdout


def _list(remote_dir: str) -> list[dict]:
    """Entries in a volume directory; [] if it doesn't exist."""
    try:
        out = _volume_files("list", remote_dir, "--json")
    except SystemExit:
        return []
    return json.loads(out[out.index("{"):])["files"]


def _download(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    _volume_files("download", remote, str(local), "--overwrite")


def pull(dest: Path = REMOTE_DIR) -> None:
    """Rebuild `dest` as a fresh mirror of the volume. Downloads into a sibling temp dir and
    swaps it in at the end, so readers never see a half-finished pull."""
    staging = dest.with_name(dest.name + ".incoming")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    root = {e["name"] for e in _list("/") if e["type"] == "file"}
    for name in ROOT_FILES:
        if name in root and name != "heartbeat.json":
            _download(f"/{name}", staging / name)
    for asset_dir in (e["name"] for e in _list("/assets") if e["type"] == "directory"):
        base = f"/assets/{asset_dir}"
        local = staging / "assets" / asset_dir
        present = {e["name"] for e in _list(base) if e["type"] == "file"}
        for name in ASSET_FILES:
            if name in present:
                _download(f"{base}/{name}", local / name)
        (local / "history").mkdir(parents=True, exist_ok=True)
        for entry in _list(f"{base}/history"):
            if entry["type"] == "file":
                _download(f"{base}/history/{entry['name']}", local / "history" / entry["name"])
    # Heartbeat last: a pull of many assets takes minutes, and an early copy would look stale
    if "heartbeat.json" in root:
        _download("/heartbeat.json", staging / "heartbeat.json")
    (staging / ".pulled").write_text(dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    print(f"Pulled Railway state → {dest}", flush=True)


def _asset_dirs() -> list[str]:
    assets_root = REMOTE_DIR / "assets"
    return sorted(p.name for p in assets_root.iterdir() if p.is_dir()) if assets_root.is_dir() else []


def push(asset_dirs: list[str]) -> None:
    for asset_dir in asset_dirs:
        local = REMOTE_DIR / "assets" / asset_dir
        if not (local / "strategy.yaml").exists():
            raise SystemExit(f"Nothing to push for {asset_dir} — run `pull` first.")
        base = f"/assets/{asset_dir}"
        # History and the hypothesis log go first, so a new strategy never lands without its record.
        remote_history = {e["name"] for e in _list(f"{base}/history")}
        for path in sorted((local / "history").glob("*.yaml")):
            if path.name not in remote_history:
                _volume_files("upload", str(path), f"{base}/history/{path.name}")
        if (local / "hypotheses.jsonl").exists():
            _volume_files("upload", str(local / "hypotheses.jsonl"), f"{base}/hypotheses.jsonl", "--overwrite")
        _volume_files("upload", str(local / "strategy.yaml"), f"{base}/strategy.yaml", "--overwrite")
        print(f"Pushed {asset_dir}: strategy.yaml, hypotheses.jsonl, history/", flush=True)


def push_goal() -> None:
    config.goal_assets(load_yaml(config.GOAL_FILE))  # refuse to upload a goal the worker can't read
    _volume_files("upload", str(config.GOAL_FILE), "/goal.yaml", "--overwrite")
    print("Pushed goal.yaml. Redeploy (`railway up --detach`) so the worker reloads it.", flush=True)


def _strategies() -> dict[str, str]:
    return {
        d: (REMOTE_DIR / "assets" / d / "strategy.yaml").read_text(encoding="utf-8")
        for d in _asset_dirs()
        if (REMOTE_DIR / "assets" / d / "strategy.yaml").exists()
    }


def reflect(hermes: bool, force: bool) -> int:
    pull()
    before = _strategies()
    cmd = [sys.executable, "-m", "hermes_trading.reflect", "--hermes" if hermes else "--fallback"]
    if force:
        cmd.append("--force")
    env = {**os.environ, "HERMES_TRADING_STATE": str(REMOTE_DIR)}
    rc = subprocess.run(cmd, cwd=config.ROOT, env=env).returncode
    changed = [d for d, text in _strategies().items() if before.get(d) != text]
    if changed:
        push(changed)  # push what did change, even if another asset's reflection failed
        print("The worker picks up the new strategies on its next tick.", flush=True)
    else:
        print("No strategy changed — nothing pushed.", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(description="Sync state with the Railway volume.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pull", help="download the worker's state")
    p = sub.add_parser("push", help="upload reflection outputs")
    p.add_argument("assets", nargs="*", help="e.g. SOL/USDT (default: every pulled asset)")
    sub.add_parser("push-goal", help="upload local state/goal.yaml")
    r = sub.add_parser("reflect", help="pull, run one reflection cycle per asset, push changes")
    which = r.add_mutually_exclusive_group(required=True)
    which.add_argument("--fallback", action="store_true")
    which.add_argument("--hermes", action="store_true")
    r.add_argument("--force", action="store_true", help="ignore reflection_every cadence")
    args = parser.parse_args(argv)

    if args.command == "pull":
        pull()
    elif args.command == "push":
        push([config.asset_slug(a) for a in args.assets] or _asset_dirs())
    elif args.command == "push-goal":
        push_goal()
    else:
        return reflect(args.hermes, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
