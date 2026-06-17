from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np


def save_equity_svg(result: dict, path: str | Path, *, title: str = "HFTBT equity") -> dict:
    ts = np.asarray(result["stat_ts"], dtype=np.int64)
    mid = np.asarray(result["stat_mid"], dtype=float)
    position = np.asarray(result["stat_position"], dtype=float)
    balance = np.asarray(result["stat_balance"], dtype=float)
    fee = np.asarray(result["stat_fee"], dtype=float)
    equity = np.asarray(result["stat_equity"], dtype=float)
    if len(ts) == 0:
        raise ValueError("result has no stat_ts/stat_equity points")

    gross_equity = balance + position * mid
    base = max(1.0, float(np.max(np.abs(position * mid))))
    net_return = equity / base * 100.0
    gross_return = gross_equity / base * 100.0
    drawdown = np.maximum.accumulate(equity) - equity
    drawdown_pct = drawdown / base * 100.0
    summary = {
        "return_base": base,
        "final_equity": float(equity[-1]),
        "final_return_pct": float(net_return[-1]),
        "gross_return_pct": float(gross_return[-1]),
        "max_drawdown": float(np.max(drawdown)),
        "max_drawdown_pct": float(np.max(drawdown_pct)),
        "fills": int(len(result["fill_ts"])),
        "turnover": float(result["trading_value"]),
        "fee": float(result["fee"]),
        "final_position": float(result["position"]),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_svg(ts, mid, position, net_return, gross_return, drawdown_pct, summary, title), encoding="utf-8")
    return summary


def _svg(
    ts: np.ndarray,
    mid: np.ndarray,
    position: np.ndarray,
    net_return: np.ndarray,
    gross_return: np.ndarray,
    drawdown_pct: np.ndarray,
    summary: dict,
    title: str,
) -> str:
    width, height = 1180, 720
    left, right = 84, 84
    top1, bottom1 = 124, 364
    top2, bottom2 = 442, 650
    x = (ts - ts[0]) / max(1, ts[-1] - ts[0])
    xs = left + x * (width - left - right)

    ret_lo = min(float(np.min(net_return)), float(np.min(gross_return)), -float(np.max(drawdown_pct)))
    ret_hi = max(float(np.max(net_return)), float(np.max(gross_return)), 0.0)
    pos_lo = float(np.min(position))
    pos_hi = float(np.max(position))
    px_lo = float(np.min(mid[mid > 0])) if np.any(mid > 0) else 0.0
    px_hi = float(np.max(mid)) if len(mid) else 1.0

    net_points = _points(xs, _scale(net_return, ret_lo, ret_hi, top1, bottom1))
    gross_points = _points(xs, _scale(gross_return, ret_lo, ret_hi, top1, bottom1))
    price_top_points = _points(xs, _scale(mid, px_lo, px_hi, top1, bottom1))
    position_points = _points(xs, _scale(position, pos_lo, pos_hi, top2, bottom2))
    price_bottom_points = _points(xs, _scale(mid, px_lo, px_hi, top2, bottom2))
    dd_area = _area(xs, _scale(-drawdown_pct, ret_lo, ret_hi, top1, bottom1), _scale(np.zeros_like(drawdown_pct), ret_lo, ret_hi, top1, bottom1))
    zero1 = _scale_value(0.0, ret_lo, ret_hi, top1, bottom1)
    zero2 = _scale_value(0.0, pos_lo, pos_hi, top2, bottom2)
    hours = (ts[-1] - ts[0]) / 3_600_000_000_000

    cards = [
        ("Return", f'{summary["final_return_pct"]:.2f}%'),
        ("Gross", f'{summary["gross_return_pct"]:.2f}%'),
        ("Max DD", f'{summary["max_drawdown_pct"]:.2f}%'),
        ("Fills", f'{summary["fills"]}'),
        ("Turnover", f'{summary["turnover"]:.0f}'),
        ("Fee", f'{summary["fee"]:.2f}'),
    ]
    card_svg = "".join(_card(left + i * 172, 42, k, v) for i, (k, v) in enumerate(cards))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #17202a; }}
  .title {{ font-size: 21px; font-weight: 700; }}
  .muted {{ font-size: 12px; fill: #667085; }}
  .axis-label {{ font-size: 13px; fill: #344054; font-weight: 600; }}
  .label {{ font-size: 11px; fill: #667085; }}
  .value {{ font-size: 14px; font-weight: 650; }}
  .grid {{ stroke: #d8dee6; stroke-width: 1; }}
  .panel {{ fill: #ffffff; stroke: #d0d7de; stroke-width: 1; }}
  .price {{ fill: none; stroke: #b7bec8; stroke-width: 2; opacity: 0.72; }}
  .net {{ fill: none; stroke: #0f9f6e; stroke-width: 2.1; }}
  .gross {{ fill: none; stroke: #df6b2f; stroke-width: 1.8; }}
  .pos {{ fill: none; stroke: #2677bf; stroke-width: 1.5; }}
  .dd-area {{ fill: #f5b8a5; opacity: 0.28; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="28" class="title">{escape(title)}</text>
<text x="{width - right - 280}" y="28" class="muted">duration {hours:.2f}h · points {len(ts)} · base {summary["return_base"]:.2f} USDT</text>
{card_svg}
<rect x="{left}" y="{top1}" width="{width-left-right}" height="{bottom1-top1}" class="panel"/>
{_grid(left, right, top1, bottom1, width, 5)}
<path class="dd-area" d="{dd_area}"/>
<line x1="{left}" y1="{zero1:.2f}" x2="{width-right}" y2="{zero1:.2f}" stroke="#98a2b3" stroke-width="1"/>
<polyline class="price" points="{price_top_points}"/>
<polyline class="gross" points="{gross_points}"/>
<polyline class="net" points="{net_points}"/>
<text x="24" y="{(top1+bottom1)/2:.0f}" class="axis-label" transform="rotate(-90 24 {(top1+bottom1)/2:.0f})">Cumulative returns (%)</text>
<text x="{width-32}" y="{(top1+bottom1)/2:.0f}" class="axis-label" transform="rotate(90 {width-32} {(top1+bottom1)/2:.0f})">Price</text>
{_legend(width - right, top1 - 34, [("Price", "#b7bec8"), ("Equity", "#0f9f6e"), ("Equity w/o fee", "#df6b2f")])}
{_axis_text(left, right, top1, bottom1, width, ret_lo, ret_hi, px_lo, px_hi)}

<rect x="{left}" y="{top2}" width="{width-left-right}" height="{bottom2-top2}" class="panel"/>
{_grid(left, right, top2, bottom2, width, 4)}
<line x1="{left}" y1="{zero2:.2f}" x2="{width-right}" y2="{zero2:.2f}" stroke="#98a2b3" stroke-width="1"/>
<polyline class="price" points="{price_bottom_points}"/>
<polyline class="pos" points="{position_points}"/>
<text x="24" y="{(top2+bottom2)/2:.0f}" class="axis-label" transform="rotate(-90 24 {(top2+bottom2)/2:.0f})">Position (Qty)</text>
<text x="{width-32}" y="{(top2+bottom2)/2:.0f}" class="axis-label" transform="rotate(90 {width-32} {(top2+bottom2)/2:.0f})">Price</text>
{_legend(width - right, top2 - 34, [("Price", "#b7bec8"), ("Position", "#2677bf")])}
{_axis_text(left, right, top2, bottom2, width, pos_lo, pos_hi, px_lo, px_hi)}
</svg>
'''


def _card(x: int, y: int, label: str, value: str) -> str:
    return (
        f'<g><rect x="{x}" y="{y}" width="156" height="44" rx="7" fill="#f8fafc" stroke="#dbe3ea"/>'
        f'<text x="{x + 10}" y="{y + 17}" class="label">{escape(label)}</text>'
        f'<text x="{x + 10}" y="{y + 35}" class="value">{escape(value)}</text></g>'
    )


def _scale(values: np.ndarray, lo: float, hi: float, top: int, bottom: int) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.06
    lo -= pad
    hi += pad
    return bottom - (values - lo) / (hi - lo) * (bottom - top)


def _scale_value(value: float, lo: float, hi: float, top: int, bottom: int) -> float:
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.06
    lo -= pad
    hi += pad
    return float(bottom - (value - lo) / (hi - lo) * (bottom - top))


def _points(xs: np.ndarray, ys: np.ndarray) -> str:
    xs, ys = _downsample(xs, ys)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=False))


def _area(xs: np.ndarray, ys: np.ndarray, baseline: np.ndarray) -> str:
    xs, ys, baseline = _downsample3(xs, ys, baseline)
    forward = " ".join(f"L{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=False))
    back = " ".join(f"L{x:.2f},{y:.2f}" for x, y in zip(xs[::-1], baseline[::-1], strict=False))
    return f"M{xs[0]:.2f},{baseline[0]:.2f} {forward} {back} Z"


def _downsample(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) > 3500:
        step = int(np.ceil(len(xs) / 3500))
        return xs[::step], ys[::step]
    return xs, ys


def _downsample3(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xs) > 3500:
        step = int(np.ceil(len(xs) / 3500))
        return xs[::step], ys[::step], zs[::step]
    return xs, ys, zs


def _grid(left: int, right: int, top: int, bottom: int, width: int, n: int) -> str:
    return "".join(
        f'<line x1="{left}" y1="{top + (bottom - top) * i / n:.2f}" x2="{width-right}" y2="{top + (bottom - top) * i / n:.2f}" class="grid"/>'
        for i in range(n + 1)
    )


def _axis_text(left: int, right: int, top: int, bottom: int, width: int, left_lo: float, left_hi: float, right_lo: float, right_hi: float) -> str:
    return (
        f'<text x="{left - 50}" y="{top + 12}" class="muted">{left_hi:.2f}</text>'
        f'<text x="{left - 50}" y="{bottom}" class="muted">{left_lo:.2f}</text>'
        f'<text x="{width-right+10}" y="{top + 12}" class="muted">{right_hi:.6g}</text>'
        f'<text x="{width-right+10}" y="{bottom}" class="muted">{right_lo:.6g}</text>'
    )


def _legend(right_x: int, y: int, items: list[tuple[str, str]]) -> str:
    item_widths = [46 + max(34, len(label) * 7) for label, _ in items]
    width = sum(item_widths) + 10
    x = right_x - width
    parts = [f'<g><rect x="{x}" y="{y}" width="{width}" height="26" rx="6" fill="#ffffff" stroke="#dbe3ea"/>']
    cx = x + 10
    for (label, color), item_width in zip(items, item_widths, strict=False):
        parts.append(f'<line x1="{cx}" y1="{y+14}" x2="{cx+24}" y2="{y+14}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{cx+30}" y="{y+18}" class="muted">{escape(label)}</text>')
        cx += item_width
    parts.append("</g>")
    return "".join(parts)
