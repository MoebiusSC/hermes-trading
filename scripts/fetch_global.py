"""Download adjusted daily closes (USD) of European, Chinese and US stocks for scripts/research_global.py.

  uv run python scripts/fetch_global.py

Source: Yahoo Finance (yfinance, auto_adjust: splits and dividends), from June 2015 so the 10-year test
(Sep 2016 - Sep 2026) has a year of history for its first 12-1 month ranking. Alpaca's free plan does not
serve OTC bars (Tencent, Nestle, Roche, LVMH... trade at Alpaca only as OTC ADRs: 403 "subscription does
not permit querying OTC data") and its SIP daily bars start in January 2016, too late for that ranking.
The 5-year test keeps using the Alpaca cache for US stocks (scripts/fetch_stocks.py), as
scripts/research_stocks.py does; the 10-year test takes them from here.

Delisted ADRs are rebuilt from their Hong Kong line in USD, until their last NYSE session, so a universe
isn't chosen with hindsight (dropping them would keep only the survivors). A-shares (only reachable
through a broker with Stock Connect, not Alpaca) are converted from CNY.
Cached in cache/global/closes.csv (date x symbol).
"""
from pathlib import Path

import yfinance as yf

from fetch_stocks import UNIVERSE_2016, UNIVERSE_2021

OUT = Path(__file__).resolve().parent.parent / "cache" / "global"

# Largest companies by market cap in September 2021 (start of the 5-year test) with a line Alpaca can
# trade (checked against /v2/assets on 2026-09-27). "listed": NYSE/NASDAQ, bars in Alpaca's free plan.
# "otc": tradable OTC ADRs, no bars in the free plan. L'Oreal (LRLCY), Meituan (MPNGY) and ABB (ABBNY)
# are not tradable at Alpaca; China Mobile's ADR was delisted in January 2021, before the test.
EU_LISTED = ["ASML", "NVO", "NVS", "AZN", "SAP", "SHEL", "UL", "TTE", "SNY", "BUD", "DEO", "HSBC", "RIO", "GSK",
             "BTI", "BP", "EQNR", "SAN", "STLA", "UBS"]
EU_OTC = ["LVMUY", "NSRGY", "RHHBY", "HESAY", "SIEGY", "PROSY", "IDEXY", "EADSY", "ALIZY", "DTEGY", "SBGSY"]
CN_LISTED = ["BABA", "PDD", "PTR", "LFC", "JD", "NTES", "NIO", "SNP", "BIDU", "LI", "XPEV", "BILI", "YUMC", "BEKE",
             "ZTO", "TCOM", "TME", "FUTU", "HTHT", "VIPS"]
CN_OTC = ["TCEHY", "IDCBY", "CIHKY", "CICHY", "ACGBY", "PNGAY", "BACHY", "BYDDY", "XIACY"]
# Top 20 A-shares (Shanghai/Shenzhen) of September 2021: not reachable through Alpaca, only through a
# broker with Stock Connect (IBKR, Futu/moomoo, Tiger...).
A_SHARES = ["600519.SS", "300750.SZ", "601398.SS", "600036.SS", "601318.SS", "000858.SZ", "601939.SS", "601288.SS",
            "002594.SZ", "601012.SS", "601988.SS", "600900.SS", "601857.SS", "601628.SS", "300760.SZ", "601166.SS",
            "000568.SZ", "000333.SZ", "002415.SZ", "600276.SS"]
# 10-year test: the 20 largest NYSE/NASDAQ-listed European and Chinese companies of September 2016
# (approximate market caps; Unilever NV and plc, unified in 2020, count once as UL). Shire (SHPG, taken
# over by Takeda in January 2019) had no price history left to download and is replaced by the next one,
# National Grid (NGG).
EU_2016 = ["BUD", "NVS", "SHEL", "HSBC", "UL", "BTI", "TTE", "SAP", "BP", "GSK", "NVO", "SNY", "AZN", "VOD", "DEO",
           "SAN", "RIO", "EQNR", "UBS", "NGG"]
CN_2016 = ["BABA", "CHL", "PTR", "SNP", "LFC", "BIDU", "CEO", "CHA", "JD", "NTES", "CHU", "TCOM", "HNP", "WB", "ZNH",
           "CEA", "VIPS", "ACH", "EDU", "SHI"]
US_2016 = [s.replace(".", "-") for s in UNIVERSE_2016]
US_2021 = [s.replace(".", "-") for s in UNIVERSE_2021]
ETFS = ["VGK", "EZU", "FEZ", "IEUR", "EWG", "EWU", "EWQ", "EWL", "EWP", "HEDJ",
        "FXI", "MCHI", "KWEB", "ASHR", "GXC", "CQQQ", "SPY", "QQQ"]
# Delisted (or ticker since reused by another company: CHA, ACH) ADRs: Hong Kong line and last NYSE session.
# Xiaomi's OTC ADR (XIACY) only has Yahoo history from August 2024: its Hong Kong line stands in for it.
HK_PROXY = {"PTR": ("0857.HK", "2022-09-08"), "LFC": ("2628.HK", "2022-09-08"), "SNP": ("0386.HK", "2022-09-08"),
            "ACH": ("2600.HK", "2022-09-08"), "SHI": ("0338.HK", "2022-09-08"), "HNP": ("0902.HK", "2022-07-07"),
            "ZNH": ("1055.HK", "2023-02-02"), "CEA": ("0670.HK", "2023-02-02"), "CHL": ("0941.HK", "2021-01-08"),
            "CHA": ("0728.HK", "2021-01-08"), "CHU": ("0762.HK", "2021-01-08"), "CEO": ("0883.HK", "2021-03-08"),
            "XIACY": ("1810.HK", None)}
DELISTED = {adr: last for adr, (_, last) in HK_PROXY.items() if last}
FX = {".HK": "HKDUSD=X", ".SS": "CNYUSD=X", ".SZ": "CNYUSD=X"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    direct = sorted({s for s in EU_LISTED + EU_OTC + CN_LISTED + CN_OTC + EU_2016 + CN_2016 + US_2016 + US_2021 + ETFS
                     if s not in HK_PROXY})
    local = A_SHARES + [hk for hk, _ in HK_PROXY.values()]
    tickers = direct + local + sorted(set(FX.values()))
    px = yf.download(tickers, start="2015-06-01", end="2026-09-27", auto_adjust=True, progress=False)["Close"]
    fx = px[sorted(set(FX.values()))].ffill()
    out = px[direct].copy()
    for sym in local:
        pair = next(v for k, v in FX.items() if sym.endswith(k))
        out[sym] = px[sym] * fx[pair]
    for adr, (hk, last) in HK_PROXY.items():
        out[adr] = out[hk].where(out.index <= (last or "2099"))
    out = out.drop(columns=[hk for hk, _ in HK_PROXY.values()])
    out = out.rename(columns={s: s.replace("-", ".") for s in out if s.startswith("BRK")})
    missing = [s for s in out if out[s].notna().sum() < 250]
    if missing:
        print("sin datos suficientes:", missing)
    out.to_csv(OUT / "closes.csv")
    print(out.shape, out.index[0].date(), out.index[-1].date())


if __name__ == "__main__":
    main()
