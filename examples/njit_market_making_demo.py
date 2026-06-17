#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit

from hftbt import BUY, GTX, LIMIT, SELL, FastConfig, load_hft_events, run_njit_strategy, save_equity_svg


@njit
def market_making_algo(hbt):
    asset_no = 0
    depth = hbt.depth(asset_no)
    tick_size = depth.tick_size
    lot_size = depth.lot_size

    while hbt.elapse(10_000_000) == 0:
        hbt.clear_inactive_orders(asset_no)
        depth = hbt.depth(asset_no)
        if not (depth.best_bid > 0.0 and depth.best_ask > depth.best_bid):
            continue

        position = hbt.position(asset_no)
        mid_price = (depth.best_bid + depth.best_ask) / 2.0
        max_notional_position = 1000.0
        notional_qty = 100.0
        half_spread = tick_size
        reservation_price = mid_price - 0.000001 * position

        new_bid_tick = min(int(np.round((reservation_price - half_spread) / tick_size)), depth.best_bid_tick)
        new_ask_tick = max(int(np.round((reservation_price + half_spread) / tick_size)), depth.best_ask_tick)
        order_qty = np.round(notional_qty / mid_price / lot_size) * lot_size

        if hbt.elapse(1_000_000) != 0:
            return False

        update_bid = True
        update_ask = True
        buy_limit_exceeded = position * mid_price > max_notional_position
        sell_limit_exceeded = position * mid_price < -max_notional_position

        last_order_id = -1
        order_values = hbt.orders(asset_no).values()
        while order_values.has_next():
            order = order_values.get()
            if order.side == BUY:
                if order.price_tick == new_bid_tick or buy_limit_exceeded:
                    update_bid = False
                if order.cancellable and (update_bid or buy_limit_exceeded):
                    hbt.cancel(asset_no, order.order_id, False)
                    last_order_id = order.order_id
            elif order.side == SELL:
                if order.price_tick == new_ask_tick or sell_limit_exceeded:
                    update_ask = False
                if order.cancellable and (update_ask or sell_limit_exceeded):
                    hbt.cancel(asset_no, order.order_id, False)
                    last_order_id = order.order_id

        if update_bid and not buy_limit_exceeded:
            order_id = new_bid_tick
            hbt.submit_buy_order(asset_no, order_id, new_bid_tick * tick_size, order_qty, GTX, LIMIT, False)
            last_order_id = order_id
        if update_ask and not sell_limit_exceeded:
            order_id = new_ask_tick
            hbt.submit_sell_order(asset_no, order_id, new_ask_tick * tick_size, order_qty, GTX, LIMIT, False)
            last_order_id = order_id

        if last_order_id >= 0:
            if not hbt.wait_order_response(asset_no, last_order_id, 5_000_000_000):
                return False

    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HFTBT njit market-making API demo.")
    p.add_argument("--events-npz", type=Path, default=Path("data/demo_events.npz"))
    p.add_argument("--tick-size", type=float, default=0.0001)
    p.add_argument("--lot-size", type=float, default=0.1)
    p.add_argument("--record-interval-ms", type=int, default=60_000)
    p.add_argument("--equity-svg", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events = load_hft_events(args.events_npz)
    result = run_njit_strategy(
        events,
        market_making_algo,
        FastConfig(
            tick_size=args.tick_size,
            lot_size=args.lot_size,
            fill_model="queue_tail",
            record_interval_ns=args.record_interval_ms * 1_000_000,
        ),
    )
    svg_summary = save_equity_svg(result, args.equity_svg, title="HFTBT njit market making") if args.equity_svg is not None else {}
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "fills": int(len(result["fill_ts"])),
                "curve_points": int(len(result["stat_ts"])),
                "position": result["position"],
                "balance": result["balance"],
                "fee": result["fee"],
                "trading_value": result["trading_value"],
                "final_equity": float(result["stat_equity"][-1]) if len(result["stat_equity"]) else None,
                "max_drawdown": svg_summary.get("max_drawdown"),
                "equity_svg": str(args.equity_svg) if args.equity_svg is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
