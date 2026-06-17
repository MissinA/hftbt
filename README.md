# HFTBT

Lightweight high-performance HFT backtest engine with a hftbacktest-style
`@njit` strategy API.

## Install

```bash
cd /Users/chenchen/Desktop/quant/hftbt
uv sync
```

## Run Demo

```bash
uv run python examples/njit_market_making_demo.py \
  --events-npz data/demo_events.npz \
  --tick-size 0.0001 \
  --lot-size 0.1
```

With a fast equity curve:

```bash
uv run python examples/njit_market_making_demo.py \
  --events-npz data/demo_events.npz \
  --tick-size 0.0001 \
  --lot-size 0.1 \
  --equity-svg results/equity.svg
```

## Strategy Shape

```python
from numba import njit
from hftbt import GTX, LIMIT, FastConfig, load_hft_events, run_njit_strategy, save_equity_svg


@njit
def market_making_algo(hbt):
    asset_no = 0
    while hbt.elapse(10_000_000) == 0:
        depth = hbt.depth(asset_no)
        if depth.best_bid <= 0 or depth.best_ask <= depth.best_bid:
            continue
        hbt.submit_buy_order(asset_no, depth.best_bid_tick, depth.best_bid, 1.0, GTX, LIMIT, False)
    return True


events = load_hft_events("data/demo_events.npz")
result = run_njit_strategy(
    events,
    market_making_algo,
    FastConfig(tick_size=0.0001, lot_size=0.1, record_interval_ns=60_000_000_000),
)
equity = result["stat_equity"]
save_equity_svg(result, "results/equity.svg")
```

## Download CryptoHFTData

```bash
# preferred local secret file:
mkdir -p ~/.config/hftbt
chmod 700 ~/.config/hftbt
$EDITOR ~/.config/hftbt/secrets.toml
```

```toml
[cryptohftdata]
api_key = "..."
```

uv run hftbt download-cryptohftdata \
  --symbol ARBUSDT \
  --exchange binance_futures \
  --start-date 2026-06-01 \
  --end-date 2026-06-01 \
  --out data/ARBUSDT_2026-06-01.npz
```

## Check

```bash
uv run hftbt check
```
