"""Polymarket price quantile forecaster (stdlib only, deterministic).

Approach
--------
Each market receives quantile forecasts for the next `horizon_minutes`
minutes. The quantiles are a centred random-walk fan around the latest price:

    q(h) = clamp01_6(level + z * sigma * sqrt(h))

where `z` comes from a fixed standard-normal quantile table and `sigma` is an
EWMA volatility estimate tuned to each market's own recent minute-to-minute
changes (240-minute half-life). A heavily damped mean-reversion pullback
gently bends the fan toward the 60-minute mean whenever the price has drifted
unusually far, acting as a slow drift guard.

Deterministic and standard-library only.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Standard-normal quantile values for the five output levels.
Z_VALUES = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Floor mirrors the house random-walk baseline spread of 0.0025.
FLOOR_SIGMA = 0.0025

# Ceiling prevents wild markets from producing unboundedly wide fans.
CEILING_SIGMA = 0.012

# EWMA smoothing factor derived from the 240-minute volatility half-life.
LAMBDA = 0.5 ** (1.0 / 240.0)

# Pullback settings: revert toward the 60-minute mean, damped hard.
PULL_WINDOW = 60
PULL_DAMPING = 0.05
PULL_RAMP_MINUTES = 60.0

# A deviation is "unusual" beyond this many times the threshold spread.
DEVIATION_MULTIPLE = 8.0


def clamp01_6(value: float) -> float:
    """Clamp a value into [0, 1] and round to six decimal places."""
    bounded = min(1.0, max(0.0, value))
    return round(bounded * 1e6) / 1e6


def estimate_sigma(prices) -> float:
    """Estimate the per-minute Gaussian spread from recent |price changes|.

    Absolute minute-to-minute changes are combined with an exponentially
    weighted moving average whose half-life is 240 minutes, so recent moves
    dominate while older history still contributes. The mean absolute change
    is rescaled into a Gaussian spread and clipped to the configured bounds.
    """
    if len(prices) < 30:
        return FLOOR_SIGMA

    abs_changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]

    weighted_sum = 0.0
    weight_sum = 0.0
    weight = 1.0
    for change in reversed(abs_changes):
        weighted_sum += weight * change
        weight_sum += weight
        weight *= LAMBDA

    mean_abs = weighted_sum / weight_sum if weight_sum > 0.0 else 0.0
    # 0.6745 is the MAD-to-sigma factor for a Gaussian distribution, so a
    # mean absolute change maps directly onto a per-minute standard deviation.
    sigma = mean_abs / 0.6745
    return max(FLOOR_SIGMA, min(CEILING_SIGMA, sigma))


def forecast_market(prices, horizon: int):
    """Produce quantile rows for one market over `horizon` minutes."""
    last_price = prices[-1]
    sigma = estimate_sigma(prices)

    # Mean-reversion pullback toward the recent 60-minute average. The drift
    # correction is tiny by design and only activates on a large deviation.
    recent_prices = prices[-PULL_WINDOW:]
    window_mean = sum(recent_prices) / len(recent_prices)
    deviation = last_price - window_mean
    pullback = 0.0
    threshold = DEVIATION_MULTIPLE * max(FLOOR_SIGMA, sigma / 3.0)
    if abs(deviation) > threshold:
        pullback = -PULL_DAMPING * deviation

    rows = []
    for minute in range(1, horizon + 1):
        ramp = min(1.0, minute / PULL_RAMP_MINUTES)
        level = last_price + pullback * ramp
        row = [
            clamp01_6(level + z * sigma * math.sqrt(minute)) for z in Z_VALUES
        ]
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(INPUT_DIR.iterdir()):
        if input_path.suffix != ".in":
            continue
        case = json.loads(input_path.read_text())
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": market["id"], "q": forecast_market(market["prices"], horizon)}
            for market in case["markets"]
        ]
        output_path = OUTPUT_DIR / (input_path.stem + ".out")
        output_path.write_text(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
