from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BUY_EVENT = 536_870_912
SELL_EVENT = 268_435_456
DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4


@dataclass(frozen=True)
class EventData:
    data: np.ndarray
    source: Path

    @property
    def start_ns(self) -> int:
        return int(self.data["local_ts"][0]) if len(self.data) else 0

    @property
    def end_ns(self) -> int:
        return int(self.data["local_ts"][-1]) if len(self.data) else 0


REQUIRED_EVENT_FIELDS = {"ev", "exch_ts", "local_ts", "px", "qty"}


def _event_array_from_npz(path: Path, z: np.lib.npyio.NpzFile) -> np.ndarray:
    if "data" in z.files:
        return z["data"]
    candidates = []
    for key in z.files:
        arr = z[key]
        names = set(arr.dtype.names or ())
        if REQUIRED_EVENT_FIELDS <= names:
            candidates.append(key)
    if len(candidates) == 1:
        return z[candidates[0]]
    if len(candidates) > 1:
        raise ValueError(f"{path} contains multiple event-like arrays: {candidates}; use a file with one event array")
    schemas = {key: list(z[key].dtype.names or ()) for key in z.files}
    raise ValueError(f"{path} does not contain an event array with fields {sorted(REQUIRED_EVENT_FIELDS)}; schemas={schemas}")


def _load_event_array(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return _event_array_from_npz(path, loaded)
    return loaded


def _validate_event_array(path: Path, data: np.ndarray) -> np.ndarray:
    missing = REQUIRED_EVENT_FIELDS - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"{path} missing event fields: {sorted(missing)}; found={list(data.dtype.names or ())}")
    if len(data) and np.any(data["local_ts"][1:] < data["local_ts"][:-1]):
        data = np.sort(data, order=["local_ts", "exch_ts"], kind="stable")
    return data


def load_hft_npz(path: Path) -> EventData:
    data = _validate_event_array(path, _load_event_array(path))
    return EventData(data=data, source=path)


def _event_files_from_path(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix not in {".npz", ".npy"}:
            raise ValueError(f"{path} is not a supported event array file; expected .npz or .npy")
        return [path]
    if path.is_dir():
        files = sorted([*path.glob("*.npz"), *path.glob("*.npy")])
        if not files:
            raise ValueError(f"{path} does not contain .npz or .npy event files")
        return files
    raise FileNotFoundError(path)


def load_hft_events(paths: Path | str | Iterable[Path | str]) -> EventData:
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(p) for p in paths]
    files: list[Path] = []
    for path in path_list:
        files.extend(_event_files_from_path(path))
    if not files:
        raise ValueError("no event files supplied")
    if len(files) == 1:
        return load_hft_npz(files[0])
    arrays = [_validate_event_array(path, _load_event_array(path)) for path in files]
    data = np.concatenate(arrays)
    if len(data) and np.any(data["local_ts"][1:] < data["local_ts"][:-1]):
        data = np.sort(data, order=["local_ts", "exch_ts"], kind="stable")
    return EventData(data=data, source=Path("<multiple>"))


def event_side(ev: int) -> int:
    if ev & BUY_EVENT:
        return 1
    if ev & SELL_EVENT:
        return -1
    return 0


def is_trade(ev: int) -> bool:
    return bool(ev & TRADE_EVENT)


def is_depth(ev: int) -> bool:
    return bool(ev & DEPTH_EVENT) or bool(ev & DEPTH_SNAPSHOT_EVENT) or bool(ev & DEPTH_CLEAR_EVENT)
