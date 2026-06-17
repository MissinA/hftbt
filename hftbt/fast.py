from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import int8, int64, float64, boolean
from numba import njit, types
from numba.experimental import jitclass
from numba.typed import Dict

from .events import BUY_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, SELL_EVENT, TRADE_EVENT, EventData


BUY = 1
SELL = -1
GTX = 0
LIMIT = 0
FILL_TRADE_THROUGH = 1
FILL_QUEUE_TAIL = 2


@dataclass(frozen=True)
class FastConfig:
    tick_size: float
    lot_size: float
    maker_fee: float = 0.000144
    fill_model: str = "queue_tail"
    queue_ahead_multiplier: float = 1.0
    max_orders: int = 16_384
    record_interval_ns: int = 60_000_000_000


@njit(cache=True)
def _same_price(left: float, right: float, tick_size: float) -> bool:
    eps = max(1.0e-12, tick_size * 1.0e-9)
    return abs(left - right) <= eps


@njit(cache=True)
def _round_qty(qty: float, lot_size: float) -> float:
    return np.floor(qty / lot_size + 1.0e-12) * lot_size


@njit(cache=True)
def _refresh_bid(book) -> tuple[float, float]:
    best = -1.0e300
    qty = 0.0
    for px in book:
        if px > best:
            best = px
            qty = book[px]
    return best, qty


@njit(cache=True)
def _refresh_ask(book) -> tuple[float, float]:
    best = 1.0e300
    qty = 0.0
    for px in book:
        if px < best:
            best = px
            qty = book[px]
    return best, qty


depth_spec = [
    ("tick_size", float64),
    ("lot_size", float64),
    ("best_bid", float64),
    ("best_ask", float64),
    ("best_bid_qty", float64),
    ("best_ask_qty", float64),
    ("best_bid_tick", int64),
    ("best_ask_tick", int64),
]


@jitclass(depth_spec)
class Depth:
    def __init__(self, tick_size, lot_size, best_bid, best_ask, best_bid_qty, best_ask_qty):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.best_bid_qty = best_bid_qty
        self.best_ask_qty = best_ask_qty
        self.best_bid_tick = int(np.round(best_bid / tick_size)) if best_bid > 0.0 else -1
        self.best_ask_tick = int(np.round(best_ask / tick_size)) if best_ask < 1.0e299 else -1


order_spec = [
    ("order_id", int64),
    ("side", int8),
    ("price", float64),
    ("price_tick", int64),
    ("qty", float64),
    ("cancellable", boolean),
]


@jitclass(order_spec)
class Order:
    def __init__(self, order_id, side, price, tick_size, qty, cancellable):
        self.order_id = order_id
        self.side = side
        self.price = price
        self.price_tick = int(np.round(price / tick_size))
        self.qty = qty
        self.cancellable = cancellable


order_iter_spec = [
    ("order_ids", int64[:]),
    ("order_side", int8[:]),
    ("order_price", float64[:]),
    ("order_qty", float64[:]),
    ("order_active", boolean[:]),
    ("order_count", int64),
    ("tick_size", float64),
    ("cursor", int64),
    ("next_idx", int64),
]


@jitclass(order_iter_spec)
class OrderIterator:
    def __init__(self, order_ids, order_side, order_price, order_qty, order_active, order_count, tick_size):
        self.order_ids = order_ids
        self.order_side = order_side
        self.order_price = order_price
        self.order_qty = order_qty
        self.order_active = order_active
        self.order_count = order_count
        self.tick_size = tick_size
        self.cursor = 0
        self.next_idx = -1

    def has_next(self):
        i = self.cursor
        while i < self.order_count:
            if self.order_active[i]:
                self.next_idx = i
                return True
            i += 1
        self.next_idx = -1
        return False

    def get(self):
        i = self.next_idx
        if i < 0:
            i = self.cursor
            while i < self.order_count and not self.order_active[i]:
                i += 1
        self.cursor = i + 1
        return Order(self.order_ids[i], self.order_side[i], self.order_price[i], self.tick_size, self.order_qty[i], True)


orders_spec = [
    ("order_ids", int64[:]),
    ("order_side", int8[:]),
    ("order_price", float64[:]),
    ("order_qty", float64[:]),
    ("order_active", boolean[:]),
    ("order_count", int64),
    ("tick_size", float64),
]


@jitclass(orders_spec)
class Orders:
    def __init__(self, order_ids, order_side, order_price, order_qty, order_active, order_count, tick_size):
        self.order_ids = order_ids
        self.order_side = order_side
        self.order_price = order_price
        self.order_qty = order_qty
        self.order_active = order_active
        self.order_count = order_count
        self.tick_size = tick_size

    def values(self):
        return OrderIterator(
            self.order_ids,
            self.order_side,
            self.order_price,
            self.order_qty,
            self.order_active,
            self.order_count,
            self.tick_size,
        )

    def len(self):
        total = 0
        for i in range(self.order_count):
            if self.order_active[i]:
                total += 1
        return total


hbt_spec = [
    ("evs", int64[:]),
    ("local_ts", int64[:]),
    ("pxs", float64[:]),
    ("qtys", float64[:]),
    ("tick_size", float64),
    ("lot_size", float64),
    ("maker_fee", float64),
    ("fill_model", int64),
    ("queue_ahead_multiplier", float64),
    ("idx", int64),
    ("now", int64),
    ("done", boolean),
    ("bids", types.DictType(float64, float64)),
    ("asks", types.DictType(float64, float64)),
    ("best_bid", float64),
    ("best_ask", float64),
    ("best_bid_qty", float64),
    ("best_ask_qty", float64),
    ("order_ids", int64[:]),
    ("order_side", int8[:]),
    ("order_price", float64[:]),
    ("order_qty", float64[:]),
    ("order_queue", float64[:]),
    ("order_active", boolean[:]),
    ("order_count", int64),
    ("position_value", float64),
    ("balance", float64),
    ("fee", float64),
    ("trading_value", float64),
    ("fill_ts", int64[:]),
    ("fill_order_id", int64[:]),
    ("fill_side", int8[:]),
    ("fill_price", float64[:]),
    ("fill_qty", float64[:]),
    ("fill_fee", float64[:]),
    ("fill_count", int64),
    ("record_interval_ns", int64),
    ("next_record_ts", int64),
    ("stat_ts", int64[:]),
    ("stat_mid", float64[:]),
    ("stat_position", float64[:]),
    ("stat_balance", float64[:]),
    ("stat_fee", float64[:]),
    ("stat_equity", float64[:]),
    ("stat_count", int64),
]


@jitclass(hbt_spec)
class FastHBT:
    def __init__(
        self,
        evs,
        local_ts,
        pxs,
        qtys,
        tick_size,
        lot_size,
        maker_fee,
        fill_model,
        queue_ahead_multiplier,
        max_orders,
        record_interval_ns,
    ):
        self.evs = evs
        self.local_ts = local_ts
        self.pxs = pxs
        self.qtys = qtys
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.maker_fee = maker_fee
        self.fill_model = fill_model
        self.queue_ahead_multiplier = queue_ahead_multiplier
        self.idx = 0
        self.now = int(local_ts[0]) if len(local_ts) else 0
        self.done = len(local_ts) == 0
        self.bids = Dict.empty(key_type=types.float64, value_type=types.float64)
        self.asks = Dict.empty(key_type=types.float64, value_type=types.float64)
        self.best_bid = -1.0e300
        self.best_ask = 1.0e300
        self.best_bid_qty = 0.0
        self.best_ask_qty = 0.0
        self.order_ids = np.zeros(max_orders, np.int64)
        self.order_side = np.zeros(max_orders, np.int8)
        self.order_price = np.zeros(max_orders, np.float64)
        self.order_qty = np.zeros(max_orders, np.float64)
        self.order_queue = np.zeros(max_orders, np.float64)
        self.order_active = np.zeros(max_orders, np.bool_)
        self.order_count = 0
        self.position_value = 0.0
        self.balance = 0.0
        self.fee = 0.0
        self.trading_value = 0.0
        max_fills = max(8, len(local_ts) + max_orders)
        self.fill_ts = np.zeros(max_fills, np.int64)
        self.fill_order_id = np.zeros(max_fills, np.int64)
        self.fill_side = np.zeros(max_fills, np.int8)
        self.fill_price = np.zeros(max_fills, np.float64)
        self.fill_qty = np.zeros(max_fills, np.float64)
        self.fill_fee = np.zeros(max_fills, np.float64)
        self.fill_count = 0
        self.record_interval_ns = record_interval_ns
        self.next_record_ts = self.now
        max_stats = 1
        if record_interval_ns > 0 and len(local_ts):
            max_stats = max(2, int((local_ts[-1] - local_ts[0]) // record_interval_ns) + 4)
        self.stat_ts = np.zeros(max_stats, np.int64)
        self.stat_mid = np.zeros(max_stats, np.float64)
        self.stat_position = np.zeros(max_stats, np.float64)
        self.stat_balance = np.zeros(max_stats, np.float64)
        self.stat_fee = np.zeros(max_stats, np.float64)
        self.stat_equity = np.zeros(max_stats, np.float64)
        self.stat_count = 0

    def elapse(self, ns):
        if self.done:
            return 1
        target = self.now + ns
        while self.idx < len(self.local_ts) and self.local_ts[self.idx] <= target:
            self.now = self.local_ts[self.idx]
            self._process_event(self.evs[self.idx], self.pxs[self.idx], self.qtys[self.idx])
            self._record_due()
            self.idx += 1
        self.now = target
        self._record_due()
        if self.idx >= len(self.local_ts):
            self.done = True
            return 1
        return 0

    def clear_inactive_orders(self, asset_no):
        return None

    def depth(self, asset_no):
        return Depth(self.tick_size, self.lot_size, self.best_bid, self.best_ask, self.best_bid_qty, self.best_ask_qty)

    def orders(self, asset_no):
        return Orders(
            self.order_ids,
            self.order_side,
            self.order_price,
            self.order_qty,
            self.order_active,
            self.order_count,
            self.tick_size,
        )

    def position(self, asset_no):
        return self.position_value

    def cancel(self, asset_no, order_id, wait):
        idx = self._find_order(order_id)
        if idx >= 0:
            self.order_active[idx] = False
        return True

    def wait_order_response(self, asset_no, order_id, timeout):
        return True

    def submit_buy_order(self, asset_no, order_id, price, qty, time_in_force, order_type, wait):
        return self._submit(order_id, BUY, price, qty)

    def submit_sell_order(self, asset_no, order_id, price, qty, time_in_force, order_type, wait):
        return self._submit(order_id, SELL, price, qty)

    def _submit(self, order_id, side, price, qty):
        qty = _round_qty(qty, self.lot_size)
        if qty <= 0.0:
            return False
        if side == BUY and self.best_ask < 1.0e299 and price >= self.best_ask:
            return False
        if side == SELL and self.best_bid > 0.0 and price <= self.best_bid:
            return False
        idx = self._find_order(order_id)
        if idx < 0:
            if self.order_count >= len(self.order_ids):
                return False
            idx = self.order_count
            self.order_count += 1
        self.order_ids[idx] = order_id
        self.order_side[idx] = side
        self.order_price[idx] = price
        self.order_qty[idx] = qty
        self.order_queue[idx] = self._queue_ahead(side, price)
        self.order_active[idx] = True
        return True

    def _find_order(self, order_id):
        for i in range(self.order_count):
            if self.order_ids[i] == order_id:
                return i
        return -1

    def _queue_ahead(self, side, price):
        qty = 0.0
        book = self.bids if side == BUY else self.asks
        for px in book:
            if _same_price(px, price, self.tick_size):
                qty += book[px]
        if self.queue_ahead_multiplier < 0.0:
            return 0.0
        return qty * self.queue_ahead_multiplier

    def _process_event(self, ev, px, qty):
        side = 0
        if ev & BUY_EVENT:
            side = BUY
        elif ev & SELL_EVENT:
            side = SELL
        kind = ev & 255
        if kind == DEPTH_CLEAR_EVENT:
            if side == BUY:
                self.bids.clear()
                self.best_bid = -1.0e300
                self.best_bid_qty = 0.0
            elif side == SELL:
                self.asks.clear()
                self.best_ask = 1.0e300
                self.best_ask_qty = 0.0
            return
        if (ev & DEPTH_EVENT) or (ev & DEPTH_SNAPSHOT_EVENT):
            self._update_depth(side, px, qty)
            return
        if ev & TRADE_EVENT:
            self._match_trade(side, px, qty)

    def _update_depth(self, side, px, qty):
        if side == BUY:
            if qty <= 0.0:
                if px in self.bids:
                    del self.bids[px]
                if _same_price(px, self.best_bid, self.tick_size):
                    self.best_bid, self.best_bid_qty = _refresh_bid(self.bids)
            else:
                self.bids[px] = qty
                if self.best_bid <= 0.0 or px >= self.best_bid:
                    self.best_bid = px
                    self.best_bid_qty = qty
        elif side == SELL:
            if qty <= 0.0:
                if px in self.asks:
                    del self.asks[px]
                if _same_price(px, self.best_ask, self.tick_size):
                    self.best_ask, self.best_ask_qty = _refresh_ask(self.asks)
            else:
                self.asks[px] = qty
                if self.best_ask >= 1.0e299 or px <= self.best_ask:
                    self.best_ask = px
                    self.best_ask_qty = qty

    def _match_trade(self, trade_side, trade_price, trade_qty):
        for i in range(self.order_count):
            if not self.order_active[i]:
                continue
            fill_qty = 0.0
            side = self.order_side[i]
            price = self.order_price[i]
            if self.fill_model == FILL_QUEUE_TAIL:
                fill_qty = self._queue_tail_fill_qty(i, trade_side, trade_price, trade_qty)
            else:
                if side == BUY:
                    if trade_price < price or (trade_side == SELL and _same_price(trade_price, price, self.tick_size)):
                        fill_qty = self.order_qty[i]
                else:
                    if trade_price > price or (trade_side == BUY and _same_price(trade_price, price, self.tick_size)):
                        fill_qty = self.order_qty[i]
            if fill_qty > 0.0:
                self._fill(i, fill_qty)
                self.order_qty[i] = _round_qty(self.order_qty[i] - fill_qty, self.lot_size)
                if self.order_qty[i] <= 0.0:
                    self.order_active[i] = False

    def _queue_tail_fill_qty(self, i, trade_side, trade_price, trade_qty):
        side = self.order_side[i]
        price = self.order_price[i]
        if side == BUY:
            if trade_price < price and not _same_price(trade_price, price, self.tick_size):
                return self.order_qty[i]
            if trade_side != SELL or not _same_price(trade_price, price, self.tick_size):
                return 0.0
        else:
            if trade_price > price and not _same_price(trade_price, price, self.tick_size):
                return self.order_qty[i]
            if trade_side != BUY or not _same_price(trade_price, price, self.tick_size):
                return 0.0
        remaining = max(0.0, trade_qty)
        if self.order_queue[i] > 0.0:
            consumed = min(self.order_queue[i], remaining)
            self.order_queue[i] -= consumed
            remaining -= consumed
        if remaining <= 0.0:
            return 0.0
        return _round_qty(min(self.order_qty[i], remaining), self.lot_size)

    def _fill(self, i, qty):
        price = self.order_price[i]
        side = self.order_side[i]
        value = price * qty
        fill_fee = value * self.maker_fee
        if side == BUY:
            self.position_value += qty
            self.balance -= value
        else:
            self.position_value -= qty
            self.balance += value
        self.fee += fill_fee
        self.trading_value += value
        j = self.fill_count
        if j < len(self.fill_ts):
            self.fill_ts[j] = self.now
            self.fill_order_id[j] = self.order_ids[i]
            self.fill_side[j] = side
            self.fill_price[j] = price
            self.fill_qty[j] = qty
            self.fill_fee[j] = fill_fee
            self.fill_count += 1

    def _record_due(self):
        if self.record_interval_ns <= 0:
            return
        while self.next_record_ts <= self.now and self.stat_count < len(self.stat_ts):
            mid = 0.0
            if self.best_bid > 0.0 and self.best_ask < 1.0e299 and self.best_ask > self.best_bid:
                mid = (self.best_bid + self.best_ask) * 0.5
            equity = self.balance + self.position_value * mid - self.fee
            j = self.stat_count
            self.stat_ts[j] = self.next_record_ts
            self.stat_mid[j] = mid
            self.stat_position[j] = self.position_value
            self.stat_balance[j] = self.balance
            self.stat_fee[j] = self.fee
            self.stat_equity[j] = equity
            self.stat_count += 1
            self.next_record_ts += self.record_interval_ns


def run_njit_strategy(events: EventData, algo, config: FastConfig) -> dict:
    fill_model = FILL_QUEUE_TAIL if config.fill_model == "queue_tail" else FILL_TRADE_THROUGH
    hbt = FastHBT(
        events.data["ev"].astype(np.int64),
        events.data["local_ts"].astype(np.int64),
        events.data["px"].astype(np.float64),
        events.data["qty"].astype(np.float64),
        config.tick_size,
        config.lot_size,
        config.maker_fee,
        fill_model,
        config.queue_ahead_multiplier,
        config.max_orders,
        config.record_interval_ns,
    )
    ok = bool(algo(hbt))
    n = int(hbt.fill_count)
    m = int(hbt.stat_count)
    return {
        "ok": ok,
        "position": float(hbt.position_value),
        "balance": float(hbt.balance),
        "fee": float(hbt.fee),
        "trading_value": float(hbt.trading_value),
        "fill_ts": hbt.fill_ts[:n].copy(),
        "fill_order_id": hbt.fill_order_id[:n].copy(),
        "fill_side": hbt.fill_side[:n].copy(),
        "fill_price": hbt.fill_price[:n].copy(),
        "fill_qty": hbt.fill_qty[:n].copy(),
        "fill_fee": hbt.fill_fee[:n].copy(),
        "stat_ts": hbt.stat_ts[:m].copy(),
        "stat_mid": hbt.stat_mid[:m].copy(),
        "stat_position": hbt.stat_position[:m].copy(),
        "stat_balance": hbt.stat_balance[:m].copy(),
        "stat_fee": hbt.stat_fee[:m].copy(),
        "stat_equity": hbt.stat_equity[:m].copy(),
    }
