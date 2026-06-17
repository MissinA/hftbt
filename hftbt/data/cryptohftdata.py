from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hftbt.events import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
)


EVENT_DTYPE = np.dtype(
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

ORDERBOOK_REQUIRED_COLUMNS = {
    "received_time",
    "event_time",
    "event_type",
    "side",
    "price",
    "quantity",
}
TRADES_REQUIRED_COLUMNS = {
    "received_time",
    "event_time",
    "price",
    "quantity",
    "is_buyer_maker",
}
ORDERBOOK_MESSAGE_ID_COLUMNS = (
    "symbol",
    "first_update_id",
    "final_update_id",
    "prev_final_update_id",
    "last_update_id",
)


def _get_client(api_key: str | None, client: Any) -> Any:
    if client is not None:
        return client
    try:
        from cryptohftdata import CryptoHFTDataClient
    except ImportError as exc:
        raise ImportError(
            "cryptohftdata SDK is required for download support. Install it separately, "
            "then set CRYPTOHFTDATA_API_KEY or ~/.config/hftbt/secrets.toml."
        ) from exc
    resolved = api_key or os.environ.get("CRYPTOHFTDATA_API_KEY") or os.environ.get("HFTBACKTESTDATA_API_KEY") or _config_api_key()
    if not resolved:
        raise ValueError(
            "CryptoHFTData API key is required. Pass api_key, set CRYPTOHFTDATA_API_KEY, "
            "or add api_key to ~/.config/hftbt/secrets.toml."
        )
    return CryptoHFTDataClient(api_key=resolved)


def _config_api_key() -> str | None:
    path = Path.home() / ".config" / "hftbt" / "secrets.toml"
    if not path.exists():
        return None
    data = tomllib.loads(path.read_text())
    for section in ("cryptohftdata", "hftbacktestdata"):
        value = data.get(section, {}).get("api_key")
        if value:
            return str(value)
    return None


def download(
    symbol: str,
    exchange: str,
    start_date: str | datetime,
    end_date: str | datetime,
    *,
    api_key: str | None = None,
    client: Any = None,
    max_workers: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download orderbook and trades from CryptoHFTData through its SDK."""
    sdk = _get_client(api_key, client)
    orderbook = sdk.get_orderbook(
        symbol=symbol,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
    )
    trades = sdk.get_trades(
        symbol=symbol,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
    )
    return _to_pandas(orderbook, "orderbook"), _to_pandas(trades, "trades")


def download_and_convert(
    symbol: str,
    exchange: str,
    start_date: str | datetime,
    end_date: str | datetime,
    *,
    api_key: str | None = None,
    client: Any = None,
    max_workers: int = 10,
    output_filename: str | Path | None = None,
    base_latency_ms: float = 0.0,
) -> np.ndarray:
    orderbook, trades = download(
        symbol=symbol,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        client=client,
        max_workers=max_workers,
    )
    return convert(orderbook, trades, output_filename=output_filename, base_latency_ms=base_latency_ms)


def convert(
    orderbook: pd.DataFrame | Any,
    trades: pd.DataFrame | Any,
    *,
    output_filename: str | Path | None = None,
    base_latency_ms: float = 0.0,
) -> np.ndarray:
    """Convert CryptoHFTData orderbook/trades frames into HFTBT event arrays."""
    orderbook_df = _normalize_orderbook(_to_pandas(orderbook, "orderbook"))
    trades_df = _normalize_trades(_to_pandas(trades, "trades"))
    chunks = [
        _convert_trades(trades_df),
        _convert_orderbook_updates(orderbook_df),
        _convert_orderbook_snapshots(orderbook_df),
    ]
    total = sum(len(x) for x in chunks)
    data = np.empty(total, dtype=EVENT_DTYPE)
    offset = 0
    for chunk in chunks:
        if len(chunk):
            data[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
    data = data[:offset]
    if len(data):
        if base_latency_ms:
            data["local_ts"] += int(base_latency_ms * 1_000_000)
        data = np.sort(data, order=["local_ts", "exch_ts"], kind="stable")
        if np.any(data["local_ts"] < data["exch_ts"]):
            data["local_ts"] = np.maximum(data["local_ts"], data["exch_ts"])
    if output_filename is not None:
        out = Path(output_filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, data=data)
    return data


def _to_pandas(data: pd.DataFrame | Any, name: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    raise TypeError(f"{name} must be a pandas DataFrame or expose to_pandas().")


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _require_notna(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    bad = [col for col in columns if df[col].isna().any()]
    if bad:
        raise ValueError(f"{name} contains null values in required columns: {bad}")


def _normalize_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ORDERBOOK_REQUIRED_COLUMNS, "orderbook")
    out = pd.DataFrame(index=df.index)
    transaction_time = df["transaction_time"] if "transaction_time" in df.columns else pd.Series(pd.NA, index=df.index)
    out["local_ts"] = pd.to_numeric(df["received_time"], errors="coerce").astype("Int64")
    exch_ms = pd.to_numeric(transaction_time, errors="coerce").fillna(pd.to_numeric(df["event_time"], errors="coerce"))
    out["exch_ts"] = (exch_ms * 1_000_000).astype("Int64")
    out["event_type"] = df["event_type"].astype(str).str.lower()
    side = df["side"].astype(str).str.lower()
    out["side_code"] = np.where(side.isin(["bid", "buy"]), BUY_EVENT, np.where(side.isin(["ask", "sell"]), SELL_EVENT, np.nan))
    out["px"] = pd.to_numeric(df["price"], errors="coerce")
    out["qty"] = pd.to_numeric(df["quantity"], errors="coerce")
    for column in ORDERBOOK_MESSAGE_ID_COLUMNS:
        if column in df.columns:
            out[column] = df[column]
    _require_notna(out, ("local_ts", "exch_ts", "event_type", "side_code", "px", "qty"), "orderbook")
    unsupported = sorted(set(out.loc[~out["event_type"].isin(["update", "snapshot"]), "event_type"]))
    if unsupported:
        raise ValueError(f"orderbook contains unsupported event_type values: {unsupported}")
    out["local_ts"] = out["local_ts"].astype("int64")
    out["exch_ts"] = out["exch_ts"].astype("int64")
    out["side_code"] = out["side_code"].astype("uint64")
    return out


def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, TRADES_REQUIRED_COLUMNS, "trades")
    out = pd.DataFrame(index=df.index)
    trade_time = df["trade_time"] if "trade_time" in df.columns else pd.Series(pd.NA, index=df.index)
    out["local_ts"] = pd.to_numeric(df["received_time"], errors="coerce").astype("Int64")
    exch_ms = pd.to_numeric(trade_time, errors="coerce").fillna(pd.to_numeric(df["event_time"], errors="coerce"))
    out["exch_ts"] = (exch_ms * 1_000_000).astype("Int64")
    out["px"] = pd.to_numeric(df["price"], errors="coerce")
    out["qty"] = pd.to_numeric(df["quantity"], errors="coerce")
    out["is_buyer_maker"] = df["is_buyer_maker"].astype("boolean")
    _require_notna(out, ("local_ts", "exch_ts", "px", "qty", "is_buyer_maker"), "trades")
    out["local_ts"] = out["local_ts"].astype("int64")
    out["exch_ts"] = out["exch_ts"].astype("int64")
    out["is_buyer_maker"] = out["is_buyer_maker"].astype(bool)
    return out


def _convert_trades(df: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(df), dtype=EVENT_DTYPE)
    if not len(df):
        return out
    out["ev"] = np.where(df["is_buyer_maker"].to_numpy(), SELL_EVENT | TRADE_EVENT, BUY_EVENT | TRADE_EVENT)
    out["exch_ts"] = df["exch_ts"].to_numpy("int64")
    out["local_ts"] = df["local_ts"].to_numpy("int64")
    out["px"] = df["px"].to_numpy("float64")
    out["qty"] = df["qty"].to_numpy("float64")
    out["order_id"] = 0
    out["ival"] = 0
    out["fval"] = 0.0
    return out


def _convert_orderbook_updates(df: pd.DataFrame) -> np.ndarray:
    updates = df[df["event_type"] == "update"]
    out = np.empty(len(updates), dtype=EVENT_DTYPE)
    if not len(updates):
        return out
    out["ev"] = updates["side_code"].to_numpy("uint64") + DEPTH_EVENT
    out["exch_ts"] = updates["exch_ts"].to_numpy("int64")
    out["local_ts"] = updates["local_ts"].to_numpy("int64")
    out["px"] = updates["px"].to_numpy("float64")
    out["qty"] = updates["qty"].to_numpy("float64")
    out["order_id"] = 0
    out["ival"] = 0
    out["fval"] = 0.0
    return out


def _convert_orderbook_snapshots(df: pd.DataFrame) -> np.ndarray:
    snapshots = df[df["event_type"] == "snapshot"]
    if not len(snapshots):
        return np.empty(0, dtype=EVENT_DTYPE)
    group_keys = ["local_ts", "exch_ts", "side_code"]
    group_keys.extend([col for col in ORDERBOOK_MESSAGE_ID_COLUMNS if col in snapshots.columns])
    rows: list[tuple[int, int, int, float, float, int, int, float]] = []
    for _, group in snapshots.groupby(group_keys, sort=False, dropna=False):
        side_code = int(group["side_code"].iloc[0])
        local_ts = int(group["local_ts"].iloc[0])
        exch_ts = int(group["exch_ts"].iloc[0])
        if side_code == BUY_EVENT:
            clear_ev = DEPTH_CLEAR_EVENT | BUY_EVENT
            snapshot_ev = DEPTH_SNAPSHOT_EVENT | BUY_EVENT
            clear_px = float(group["px"].min())
            ordered = group.sort_values("px", ascending=False)
        elif side_code == SELL_EVENT:
            clear_ev = DEPTH_CLEAR_EVENT | SELL_EVENT
            snapshot_ev = DEPTH_SNAPSHOT_EVENT | SELL_EVENT
            clear_px = float(group["px"].max())
            ordered = group.sort_values("px", ascending=True)
        else:
            raise ValueError(f"unsupported snapshot side_code: {side_code}")
        rows.append((clear_ev, exch_ts, local_ts, clear_px, 0.0, 0, 0, 0.0))
        for row in ordered.itertuples(index=False):
            rows.append((snapshot_ev, exch_ts, local_ts, float(row.px), float(row.qty), 0, 0, 0.0))
    return np.array(rows, dtype=EVENT_DTYPE)
