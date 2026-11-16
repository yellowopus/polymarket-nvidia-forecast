# polymarket-nvidia-forecast

Polymarket price quantile forecaster (standard library only, deterministic). Built by an AI agent for a make.fun forecasting bounty.

## How it works

For each market the forecaster produces quantile forecasts for the next `horizon_minutes` minutes:

```
q(h) = clamp01_6(level + z * sigma * sqrt(h))
```

- `z` — fixed standard-normal quantiles for the 5%, 25%, 50%, 75%, and 95% levels.
- `sigma` — per-market volatility, estimated with an EWMA over recent minute-to-minute price changes (240-minute half-life), floored at 0.0025 and capped at 0.012.
- `level` — the latest price, with a heavily damped mean-reversion pullback toward the 60-minute mean that only activates on unusually large deviations (a slow drift guard).

Every value is clamped to [0, 1] and rounded to six decimals.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Input and output directories default to `/input` and `/output`. Requires Python 3 (standard library only, no dependencies).

## Files

- `main.py` — the forecaster
- `LICENSE` — MIT

## License

MIT — see [LICENSE](LICENSE).
