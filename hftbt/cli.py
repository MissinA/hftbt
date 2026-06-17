from __future__ import annotations

import argparse
import json
import py_compile
from pathlib import Path

import numpy as np
from numba import njit

from .data.cryptohftdata import download_and_convert
from .events import BUY_EVENT, DEPTH_EVENT, SELL_EVENT, TRADE_EVENT, EventData, load_hft_npz
from .fast import GTX, LIMIT, FastConfig, run_njit_strategy


@njit
def _check_algo(hbt):
    asset_no = 0
    while hbt.elapse(10_000_000) == 0:
        depth = hbt.depth(asset_no)
        if depth.best_bid > 0.0 and depth.best_ask > depth.best_bid:
            hbt.submit_buy_order(asset_no, depth.best_bid_tick, depth.best_bid, 1.0, GTX, LIMIT, False)
    return True


def download_command(args: argparse.Namespace) -> None:
    download_and_convert(
        symbol=args.symbol,
        exchange=args.exchange,
        start_date=args.start_date,
        end_date=args.end_date,
        api_key=args.api_key,
        max_workers=args.max_workers,
        output_filename=args.out,
        base_latency_ms=args.base_latency_ms,
    )
    events = load_hft_npz(args.out)
    print(json.dumps({"out": str(args.out), "rows": int(len(events.data)), "start_ns": events.start_ns, "end_ns": events.end_ns}, indent=2))


def check_command(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    files = [*root.glob("hftbt/**/*.py")]
    for file in files:
        py_compile.compile(str(file), doraise=True)
    payload = run_njit_strategy(_demo_events(), _check_algo, FastConfig(tick_size=0.0001, lot_size=0.1))
    fills = int(len(payload["fill_ts"]))
    if not payload["ok"] or fills <= 0:
        raise AssertionError(payload)
    print(
        json.dumps(
            {
                "ok": True,
                "compiled": len(files),
                "fills": fills,
                "final_equity": float(payload["stat_equity"][-1]) if len(payload["stat_equity"]) else None,
            },
            indent=2,
        )
    )


def _demo_events() -> EventData:
    dtype = np.dtype(
        [
            ("ev", "<u8"),
            ("exch_ts", "<i8"),
            ("local_ts", "<i8"),
            ("px", "<f8"),
            ("qty", "<f8"),
            ("order_id", "<u8"),
            ("ival", "<i8"),
            ("fval", "<f8"),
        ],
        align=True,
    )
    rows = [
        (BUY_EVENT | DEPTH_EVENT, 0, 0, 0.1000, 10.0, 0, 0, 0.0),
        (SELL_EVENT | DEPTH_EVENT, 0, 0, 0.1002, 10.0, 0, 0, 0.0),
        (SELL_EVENT | TRADE_EVENT, 0, 20_000_000, 0.1000, 11.0, 0, 0, 0.0),
        (BUY_EVENT | DEPTH_EVENT, 0, 30_000_000, 0.1000, 9.0, 0, 0, 0.0),
        (SELL_EVENT | DEPTH_EVENT, 0, 30_000_000, 0.1002, 10.0, 0, 0, 0.0),
    ]
    return EventData(data=np.array(rows, dtype=dtype), source=Path("<generated>"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hftbt", description="Small njit HFT backtest toolkit.")
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download-cryptohftdata", help="Download and convert CryptoHFTData.")
    dl.add_argument("--symbol", required=True)
    dl.add_argument("--exchange", default="binance_futures")
    dl.add_argument("--start-date", required=True)
    dl.add_argument("--end-date", required=True)
    dl.add_argument("--api-key", default=None)
    dl.add_argument("--max-workers", type=int, default=10)
    dl.add_argument("--base-latency-ms", type=float, default=0.0)
    dl.add_argument("--out", type=Path, required=True)
    dl.set_defaults(func=download_command)

    check = sub.add_parser("check", help="Run the minimal package check.")
    check.set_defaults(func=check_command)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
