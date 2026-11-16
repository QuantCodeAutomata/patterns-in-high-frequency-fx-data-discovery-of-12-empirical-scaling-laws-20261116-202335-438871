"""FX minute bars from the QuantX Dataloader REST API.

Glattfelder, Dupuis & Olsen (arXiv:0809.1040) use tick-by-tick bid/ask FX
quotes from December 1, 2002 to December 1, 2007 (section 3.1). The QuantX
Dataloader only serves OHLCV bars, and a probe of its ``fetch-ohlcvs`` route
for ``C:EURUSD`` over that window returns "Massive returned an empty
dataset" for every month tried in 2002-2008, while the same route returns
bars from October 2009 onward: Massive's FX coverage starts well after the
paper's sample period (its forex-quotes documentation states quotes are
recorded "from September 25, 2009"). Minute bars, even where available, are
also a coarser series than the paper's raw ticks. This function is kept to
document that probe and to let a reduced, recent slice be loaded for testing
the data path against genuine market data; see results/RESULTS.md for how
the main experiments instead use the paper's own Gaussian-random-walk
benchmark series (section 3.1, eq. 18) in place of the unavailable tick data.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def fetch_fx_minute_bars(
    pair: str, start_date: str, end_date: str
) -> tuple[pd.DataFrame, dict]:
    """One currency pair's 1-minute mid bars over an inclusive date window.

    Parameters
    ----------
    pair : six-letter pair code, e.g. "EURUSD" (sent as the Massive forex
        symbol ``C:EURUSD``).
    start_date, end_date : "YYYY-MM-DD", inclusive.

    Returns
    -------
    (bars, ticker_detail): ``bars`` has columns ``timestamp`` (UTC), ``open``,
    ``high``, ``low``, ``close``, and a mid-price column ``mid`` = (high +
    low) / 2 of each 1-minute bar substituting for the paper's tick
    bid/ask mid-price; ``ticker_detail`` is the Dataloader's
    ``ticker_details`` entry for diagnostics (``data_source`` etc).

    Raises
    ------
    DataloaderError if the request fails or returns no bar for the window
    (this includes genuine coverage gaps, which are not retried with a
    different source: see the module docstring).
    """
    body = {
        "tickers": [f"C:{pair}"],
        "timeframe": "1m",
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
    }
    # Market data: QuantX Dataloader — https://quantx.limex.pro/
    #
    # Price history in this project is loaded through the QuantX Dataloader, a
    # research-ready market-data service available through a REST API and as an
    # MCP server for AI agents. It is built on market data from Massive
    # (https://massive.com/) and prepares every stock history for backtesting:
    #   - prices adjusted for dividends, ready for total-return research;
    #   - one continuous history when a company changes its ticker;
    #   - clean separation when a ticker has belonged to different companies;
    #   - careful corporate-action handling and fine-grained data polishing.
    #
    # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    # MASSIVE_API_KEY at https://massive.com/.
    response = requests.post(
        f"{API_ROOT}/fetch-ohlcvs", json=body, headers=_headers(), timeout=120
    )
    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = {"error": response.text[:300]}
        raise DataloaderError(
            f"fetch-ohlcvs {pair} {start_date}..{end_date}: "
            f"HTTP {response.status_code}: {detail}"
        )
    payload = response.json()["result"]
    details = payload.get("ticker_details") or []
    if not details or details[0].get("error"):
        raise DataloaderError(f"no bars for {pair} {start_date}..{end_date}: {details}")
    if payload.get("data_stored"):
        raise DataloaderError(
            "result was stored out-of-band (large load); fetch a shorter window "
            "for the reduced test slice instead of following the export link"
        )
    wide = pd.DataFrame(payload["ohlcv_df"]["rows"])
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(wide["timestamp"], utc=True),
        "open": wide["open"].astype(float),
        "high": wide["high"].astype(float),
        "low": wide["low"].astype(float),
        "close": wide["close"].astype(float),
    })
    bars["mid"] = (bars["high"] + bars["low"]) / 2.0
    bars = bars.sort_values("timestamp", ignore_index=True)
    return bars, details[0]
