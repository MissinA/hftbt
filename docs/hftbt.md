# HFTBT

This package intentionally keeps one public backtest path:

```python
run_njit_strategy(events, algo, FastConfig(...))
```

The strategy is a numba `@njit` function that receives a hftbacktest-style
`hbt` object.

Supported API surface:

- `hbt.elapse(ns)`
- `hbt.depth(asset_no)`
- `hbt.position(asset_no)`
- `hbt.orders(asset_no).values()`
- `hbt.clear_inactive_orders(asset_no)`
- `hbt.cancel(asset_no, order_id, wait)`
- `hbt.submit_buy_order(asset_no, order_id, price, qty, GTX, LIMIT, wait)`
- `hbt.submit_sell_order(asset_no, order_id, price, qty, GTX, LIMIT, wait)`
- `hbt.wait_order_response(asset_no, order_id, timeout)`

Event files must contain:

- `ev`
- `exch_ts`
- `local_ts`
- `px`
- `qty`

Run the demo:

```bash
uv run python examples/njit_market_making_demo.py
```

Check the package:

```bash
uv run hftbt check
```

Fast equity curve:

```python
result = run_njit_strategy(events, algo, FastConfig(tick_size=0.0001, lot_size=0.1))
equity = result["stat_equity"]
save_equity_svg(result, "results/equity.svg")
```

The curve is recorded inside the njit replay path, so plotting does not replay
the raw event stream again. The SVG writer uses only the Python standard
library plus numpy already required by the engine.
