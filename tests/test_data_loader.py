"""Market-data path test: probes the QuantX Dataloader for a small, recent
FX slice (section 3.1's tick data is not obtainable for 2002-2007; see
results/RESULTS.md and data/data_loader.py). Skipped automatically when
DATALOADER_API_KEY / MASSIVE_API_KEY are not set.
"""

from __future__ import annotations

import pytest


@pytest.mark.market_data
def test_fx_minute_slice_has_plausible_mid_prices(fx_minute_slice):
    assert len(fx_minute_slice["mid"]) > 0
    assert all(m > 0 for m in fx_minute_slice["mid"])
    assert len(fx_minute_slice["timestamp"]) == len(fx_minute_slice["mid"])
    detail = fx_minute_slice["ticker_detail"]
    assert detail.get("ticker") == "C:EURUSD"
