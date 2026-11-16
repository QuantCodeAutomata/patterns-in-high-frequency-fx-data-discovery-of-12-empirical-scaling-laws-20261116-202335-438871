"""Shared pytest fixtures.

Registers the ``market_data`` marker (tests that make a live Dataloader call)
and provides a session-cached, reduced real-data slice for them. These tests
are skipped automatically when ``DATALOADER_API_KEY`` is not set.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from data.generate_synthetic import generate_grw_series


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: requires a live QuantX Dataloader / Massive call"
    )


@pytest.fixture(scope="session")
def small_grw_series() -> tuple[np.ndarray, np.ndarray]:
    """A small, fast GRW series (section 3.1, eq. 18) for unit tests."""
    return generate_grw_series(n_ticks=5_000, x0=1.336723, dt_seconds=1, sigma=1.0 / 6769.6, seed=123)


@pytest.fixture(scope="session")
def fx_minute_slice(request: pytest.FixtureRequest):
    """A few days of real EUR/USD 1-minute bars (2011), cached across runs.

    Skipped when ``DATALOADER_API_KEY``/``MASSIVE_API_KEY`` are absent.
    """
    if not (os.environ.get("DATALOADER_API_KEY") and os.environ.get("MASSIVE_API_KEY")):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    cache_dir = request.config.cache.mkdir("fx_minute_slice")
    cache_file = cache_dir / "eurusd_2011.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    from data.data_loader import fetch_fx_minute_bars

    bars, ticker_detail = fetch_fx_minute_bars("EURUSD", "2011-01-03", "2011-01-07")
    payload = {
        "timestamp": bars["timestamp"].astype(str).tolist(),
        "mid": bars["mid"].tolist(),
        "ticker_detail": ticker_detail,
    }
    cache_file.write_text(json.dumps(payload, default=str))
    return payload
