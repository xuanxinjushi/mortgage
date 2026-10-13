"""Liability-driven strategies with a mortgage pool: two tests it fails.

First, Redington's immunization test. A liability due in a single payment is
funded at today's value by either a mortgage pool or a pair of Treasury zeros
with the same duration. Shift rates instantly and see what is left. The zeros
straddle the liability date, so they are more convex than it is and the
surplus can only grow; the pool is less convex than a zero and the surplus
shrinks whichever way rates go.

Second, a bank's economic value of equity (EVE). Assets are the pool; they
are funded nine-tenths by deposits that reprice at once, so the liabilities
carry no duration, and one-tenth by equity -- the shape of the balance sheet
in Chapter 3's SVB case. Shown unhedged, and with a pay-fixed swap that
matches the pool's DV01 at inception and is never touched again.

The pool is the chapter's 6.0% coupon (6.5% note rate) at its quoted 65 bp
static spread, priced with coupon_allocation.py.

    python code/liability_driven.py
"""

from __future__ import annotations

import numpy as np

from coupon_allocation import RATE, WACS, ZSPREAD, pool_price

POOL = 3                                   # index of the 6.0% coupon in the stack
WAC, SPREAD = WACS[POOL], ZSPREAD[POOL]
BUMP = 0.0025
SHIFTS = np.array([-200, -100, -50, 50, 100, 200]) / 1e4
DEPOSITS, EQUITY = 0.90, 0.10              # funding, as fractions of assets
SWAP_TENOR = 5.0                           # the pay-fixed swap's maturity, years


def pool_value(shift: float) -> float:
    """Pool price per 100 face after an instantaneous parallel shift, spread unchanged."""
    return pool_price(WAC, RATE + shift, SPREAD)


def zero(maturity: float, shift: float) -> float:
    """Value of 1 due at `maturity`, discounted at the pool's own rate (RATE + SPREAD)."""
    return float(np.exp(-(RATE + SPREAD + shift) * maturity))


def swap_value(shift: float, tenor: float = SWAP_TENOR) -> float:
    """Value per 1 notional of receiving a fixed par coupon struck today (quarterly)."""
    times = np.arange(0.25, tenor + 1e-9, 0.25)
    disc = lambda s: np.exp(-(RATE + s) * times)
    fixed = (1.0 - disc(0.0)[-1]) / (0.25 * disc(0.0).sum())   # par rate today
    return float(fixed * 0.25 * disc(shift).sum() + disc(shift)[-1] - 1.0)


def main() -> None:
    p0 = pool_value(0.0)
    up, down = pool_value(BUMP), pool_value(-BUMP)
    duration = (down - up) / (2.0 * p0 * BUMP)
    convexity = (down + up - 2.0 * p0) / (p0 * BUMP**2)
    print(f"6.0% pool: price {p0:.2f}, effective duration {duration:.2f}, "
          f"effective convexity {convexity:.1f}")

    # --- Redington: fund a zero due at the pool's duration, worth p0 today.
    face = p0 / zero(duration, 0.0)
    short, long_ = duration - 3.0, duration + 3.0           # zeros straddling the date
    w_short = (long_ - duration) / (long_ - short)           # value weights matching duration
    print(f"\nImmunizing a liability of {face:.2f} due in {duration:.2f} years "
          f"(surplus per 100 of assets)")
    print(f"{'shift bp':>10} {'pool':>8} {'Treasury barbell':>17}")
    for s in SHIFTS:
        liability = face * zero(duration, s)
        barbell = p0 * (w_short * zero(short, s) / zero(short, 0.0)
                        + (1 - w_short) * zero(long_, s) / zero(long_, 0.0))
        print(f"{s * 1e4:+10.0f} {(pool_value(s) - liability) / p0 * 100:+8.2f}"
              f" {(barbell - liability) / p0 * 100:+17.2f}")

    # --- Bank EVE: 100 of assets, 90 of zero-duration deposits, 10 of equity.
    swap_dv01 = (swap_value(-0.0001) - swap_value(0.0001)) / 2.0
    pool_dv01 = (pool_value(-0.0001) - pool_value(0.0001)) / 2.0 / p0
    notional = pool_dv01 / swap_dv01                          # pay fixed on this much
    print(f"\nBank EVE, % of equity (assets 100, deposits 90, equity 10); "
          f"swap notional {notional * 100:.1f}")
    print(f"{'shift bp':>10} {'unhedged':>9} {'swap-hedged':>12}")
    for s in SHIFTS:
        assets = 100.0 * pool_value(s) / p0 - 100.0
        hedge = -100.0 * notional * swap_value(s)               # paying fixed
        print(f"{s * 1e4:+10.0f} {assets / (EQUITY * 100) * 100:+9.1f}"
              f" {(assets + hedge) / (EQUITY * 100) * 100:+12.1f}")


if __name__ == "__main__":
    main()
