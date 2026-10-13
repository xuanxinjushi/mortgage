"""Yield-curve strategy with mortgages: where on the curve a pool's duration sits.

A Treasury's key-rate durations come from its cash-flow dates alone. A pool's
come from two places: its cash flows are discounted along the whole curve, and
its borrowers refinance off one point of it -- mortgage rates follow the
ten-year. Bump the ten-year knot and the cash flows move too; bump the
two-year or the thirty-year and only the discounting does.

The run measures key-rate durations at 2, 5, 10 and 30 years for the
chapter's coupon stack (tent-shaped bumps, as in Chapter 3), then puts the
index-weighted portfolio through a steepener, a flattener and a butterfly,
hedged two ways: with a ten-year Treasury bullet, or with a two-year/thirty-
year barbell of the same DV01. The curve starts flat at the chapter's 6.5%;
the shapes of the moves, not the level, are the point.

    python code/curve_strategy.py
"""

from __future__ import annotations

import numpy as np

from coupon_allocation import BENCHMARK, COUPONS, RATE, WACS, ZSPREAD
from duration_hedging import SERVICING, TERM, cpr_curve
from mortgagekit.mortgage import mortgage_payment
from mortgagekit.prepayment import cpr_to_smm

KNOTS = np.array([2.0, 5.0, 10.0, 30.0])
BUMP = 0.0010                                   # 10 bp key-rate bumps
SCENARIOS = {                                    # bp moves at each knot
    "parallel +25":  [25, 25, 25, 25],
    "steepener":     [-25, -10, 0, 25],
    "flattener":     [25, 10, 0, -25],
    "butterfly":     [-15, 0, 20, -15],          # belly cheapens against the wings
}


def tent(t: np.ndarray, k: int) -> np.ndarray:
    """Chapter 3's tent: 1 at knot k, falling linearly to 0 at its neighbours."""
    x = KNOTS
    left = x[k - 1] if k > 0 else None
    right = x[k + 1] if k < len(x) - 1 else None
    w = np.zeros_like(t)
    if left is None:
        w = np.where(t <= x[k], 1.0, w)
    else:
        w = np.where((t > left) & (t <= x[k]), (t - left) / (x[k] - left), w)
    if right is None:
        w = np.where(t >= x[k], 1.0, w)
    else:
        w = np.where((t > x[k]) & (t < right), (right - t) / (right - x[k]), w)
    return w


def curve_shift(t: np.ndarray, moves: np.ndarray) -> np.ndarray:
    return sum(m * tent(t, k) for k, m in enumerate(moves))


def pool_price(wac: float, spread: float, moves: np.ndarray) -> float:
    """Pool price with the curve moved by `moves` (decimals at each knot).

    Discounting sees the whole moved curve. The borrower sees only the ten-year:
    the mortgage rate, and with it the refinancing incentive, moves one for one
    with the ten-year knot.
    """
    mortgage_rate = RATE + moves[2]
    smm = cpr_to_smm(min(cpr_curve(wac - mortgage_rate), 0.99))
    times = np.arange(1, TERM + 1) / 12.0
    shift = curve_shift(times, moves)
    balance, price = 100.0, 0.0
    for month, t in enumerate(times, start=1):
        if balance <= 1e-10:
            break
        payment = mortgage_payment(balance, wac, TERM - month + 1)
        interest = balance * wac / 12.0
        scheduled = min(payment - interest, balance)
        prepaid = (balance - scheduled) * smm
        flow = interest * (1.0 - SERVICING / wac) + scheduled + prepaid
        price += flow * np.exp(-(RATE + spread + shift[month - 1]) * t)
        balance -= scheduled + prepaid
    return price


def treasury_price(maturity: float, moves: np.ndarray, coupon: float = RATE) -> float:
    times = np.arange(0.5, maturity + 1e-9, 0.5)
    flows = np.full_like(times, 100.0 * coupon / 2.0)
    flows[-1] += 100.0
    return float(np.sum(flows * np.exp(-(RATE + curve_shift(times, moves)) * times)))


def key_rate_durations(price_fn) -> np.ndarray:
    """KRD_k = (P(-bump at k) - P(+bump at k)) / (2 P bump)."""
    base = price_fn(np.zeros(4))
    out = []
    for k in range(4):
        e = np.zeros(4)
        e[k] = BUMP
        out.append((price_fn(-e) - price_fn(e)) / (2.0 * base * BUMP))
    return np.array(out)


def main() -> None:
    print(f"Key-rate durations (years){'':>4}" + "".join(f"{k:>7.0f}y" for k in KNOTS) + "    sum")
    krds = []
    for c, wac, z in zip(COUPONS, WACS, ZSPREAD):
        krd = key_rate_durations(lambda m: pool_price(wac, z, m))
        krds.append(krd)
        print(f"  {c:>6.1%} pool{'':>18}" + "".join(f"{v:8.2f}" for v in krd) + f"{krd.sum():8.2f}")
    index_krd = BENCHMARK @ np.array(krds)
    print(f"  {'index':>11}{'':>17}" + "".join(f"{v:8.2f}" for v in index_krd) + f"{index_krd.sum():8.2f}")
    t10 = key_rate_durations(lambda m: treasury_price(10.0, m))
    print(f"  {'10y Treasury':>12}{'':>16}" + "".join(f"{v:8.2f}" for v in t10) + f"{t10.sum():8.2f}")

    # index value per 100, and hedges sized to the index's total DV01
    index_price = lambda m: sum(w * pool_price(wac, z, m) for w, wac, z in zip(BENCHMARK, WACS, ZSPREAD))
    p0 = index_price(np.zeros(4))
    dv01 = lambda f: (f(-np.full(4, 1e-4)) - f(np.full(4, 1e-4))) / 2.0
    p10, p2, p30 = (lambda m: treasury_price(10.0, m)), (lambda m: treasury_price(2.0, m)), (lambda m: treasury_price(30.0, m))
    n10 = dv01(index_price) / dv01(p10)
    # barbell: equal DV01 in the 2-year and the 30-year
    n2, n30 = 0.5 * dv01(index_price) / dv01(p2), 0.5 * dv01(index_price) / dv01(p30)

    print(f"\nIndex portfolio, P&L per 100 of pool (bp of value), hedged to zero DV01")
    print(f"{'scenario':>14} {'unhedged':>9} {'10y bullet':>11} {'2s/30s barbell':>15}")
    for name, bp in SCENARIOS.items():
        m = np.array(bp) / 1e4
        pool = index_price(m) - p0
        bullet = pool - n10 * (p10(m) - p10(np.zeros(4)))
        barbell = pool - n2 * (p2(m) - p2(np.zeros(4))) - n30 * (p30(m) - p30(np.zeros(4)))
        print(f"{name:>14} {pool / p0 * 1e4:9.0f} {bullet / p0 * 1e4:11.1f} {barbell / p0 * 1e4:15.1f}")


if __name__ == "__main__":
    main()
