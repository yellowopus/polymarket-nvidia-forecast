# polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, stdlib only).
# Improves on the house baseline (random-walk spread 0.0025/sqrt(min) around
# the checkpoint price) with a data-driven sigma per market and a slow
# mean-reverting drift guard:
#  - sigma is estimated from the market's own minute-to-minute changes over
#    recent windows (EWMA of |Δ| scaled to a Gaussian spread), floored at the
#    baseline 0.0025 and capped so quiet markets never get overconfident;
#  - quantiles are clamp(x + z*sigma*sqrt(h)) as the baseline, but the median
#    nudges toward the recent 60-minute mean when the price sits far from it
#    (weak pullback), heavily damped to avoid paying when there is no
#    reversion.
# Deterministic: no randomness anywhere.

"""EWMA sigma uses a 240-minute half-life to track slow volatility drift."""

import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
BASE_SIGMA = 0.0025
# Cap sigma so wild markets never get overconfident and widen spreads unboundedly.
MAX_SIGMA = 0.012


def clamp01_6(x: float) -> float:
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def ewma_sigma(prices) -> float:
    # minute-to-minute absolute changes, recent 4 hours weighted more
    n = len(prices)
    if n < 30:
        return BASE_SIGMA
    diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, n)]
    # EWMA over diffs with half-life 240 minutes
    lam = 0.5 ** (1 / 240)
    w = 1.0
    num = 0.0
    den = 0.0
    for d in reversed(diffs):
        num += w * d
        den += w
        w *= lam
    mad = num / den if den > 0 else 0.0
    # Gaussian: sigma ≈ MAD / 0.6745 (1-min spread)
    sigma = mad / 0.6745
    # clamp: quiet markets keep the baseline, wild ones cap
    return max(BASE_SIGMA, min(MAX_SIGMA, sigma))


def forecast_market(prices, horizon: int):
    x = prices[-1]
    sigma = ewma_sigma(prices)
    # weak pullback target: 60-minute mean, only if price deviated a lot
    recent = prices[-60:]
    mean60 = sum(recent) / len(recent)
    dev = x - mean60
    pull = 0.0
    if abs(dev) > 8 * max(BASE_SIGMA, sigma / 3):
        pull = -0.05 * dev  # damped reversion
    rows = []
    for h in range(1, horizon + 1):
        base = x + pull * min(1.0, h / 60.0)
        rows.append([clamp01_6(base + z * sigma * math.sqrt(h)) for z in Z])
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(INPUT_DIR.iterdir()):
        if f.suffix != ".in":
            continue
        case = json.loads(f.read_text())
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        out = OUTPUT_DIR / (f.stem + ".out")
        out.write_text(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
