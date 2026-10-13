"""One month of an MBS portfolio, taken apart: where the return came from.

Elena holds the coupon-stack optimizer's OAS-maximizing portfolio against the index. During
the month rates fall 25 bp, the higher coupons' spread widens, and the two
highest coupons prepay 15% faster than the model said. Each coupon's return
is split, by repricing one change at a time with coupon_allocation.py's pool
model, into

    Treasury carry   what the risk-free rate pays for a month
    spread carry     what the static spread pays for a month
    duration         the return of a Treasury zero matched to the coupon's duration
    convexity        what the rate move did beyond that duration
    spread change    the spread move, at the new rate
    prepayment       speeds coming in faster than modeled (repriced at the new speed)

and the portfolio's active return against the index is the same split,
weighted by the active weights. The month's events are illustrative.

It ends with the two ratios a committee asks for: the information ratio the
optimizer expected, and the Sharpe ratio of the three-month hedged returns
coupon_allocation.py simulates, with their skew beside it -- for the index,
the OAS-maximizing portfolio, and one that chased the static spread instead.

    python code/performance_attribution.py
"""

from __future__ import annotations

import numpy as np

from coupon_allocation import (BENCHMARK, COUPONS, RATE, TE_BUDGET, WACS, ZSPREAD,
                               exposures, hedged_returns, optimal_weights, pool_price,
                               tail)

MONTH = 1.0 / 12.0
RATE_MOVE = -0.0025                                         # rates fall 25 bp
SPREAD_MOVE = np.array([-2.0, 0.0, 3.0, 8.0, 10.0]) / 1e4       # higher coupons cheapen
SPEED = np.array([1.0, 1.0, 1.0, 1.15, 1.15])               # 6.0% and 6.5% prepay faster

PARTS = ["Treasury carry", "spread carry", "duration", "convexity",
         "spread change", "prepayment"]


def attribution(i: int, duration: float) -> np.ndarray:
    """Return of 100 of coupon i over the month, split into PARTS (fractions)."""
    wac, spread = WACS[i], ZSPREAD[i]
    p0 = pool_price(wac, RATE, spread)
    rate1 = RATE + RATE_MOVE
    p_rate = pool_price(wac, rate1, spread)
    p_spread = pool_price(wac, rate1, spread + SPREAD_MOVE[i])
    p_speed = pool_price(wac, rate1, spread + SPREAD_MOVE[i], speed=SPEED[i])
    rate_return = (p_rate - p0) / p0
    zero_return = np.exp(-RATE_MOVE * duration) - 1.0       # duration-matched Treasury zero
    return np.array([
        RATE * MONTH,
        spread * MONTH,
        zero_return,
        rate_return - zero_return,
        (p_spread - p_rate) / p0,
        (p_speed - p_spread) / p0,
    ])


def main() -> None:
    x = exposures()
    parts = np.array([attribution(i, x["duration"][i]) for i in range(len(COUPONS))])
    w = optimal_weights(x)
    active = w - BENCHMARK

    print("Monthly return by coupon, bp")
    print(f"{'':>16}" + "".join(f"{c:>8.1%}" for c in COUPONS))
    for k, name in enumerate(PARTS):
        print(f"{name:>16}" + "".join(f"{parts[i, k] * 1e4:8.1f}" for i in range(len(COUPONS))))
    print(f"{'total':>16}" + "".join(f"{parts[i].sum() * 1e4:8.1f}" for i in range(len(COUPONS))))

    print(f"\nPortfolio against the index, bp")
    print(f"{'':>16} {'portfolio':>10} {'index':>8} {'active':>8}")
    for k, name in enumerate(PARTS):
        print(f"{name:>16} {w @ parts[:, k] * 1e4:10.1f} {BENCHMARK @ parts[:, k] * 1e4:8.1f}"
              f" {active @ parts[:, k] * 1e4:+8.1f}")
    total_p, total_b = w @ parts.sum(axis=1), BENCHMARK @ parts.sum(axis=1)
    print(f"{'total':>16} {total_p * 1e4:10.1f} {total_b * 1e4:8.1f} {(total_p - total_b) * 1e4:+8.1f}")

    active_oas = active @ x["oas"]
    print(f"\nEx-ante information ratio: active OAS {active_oas * 1e4:.1f} bp / "
          f"tracking error {TE_BUDGET * 1e4:.0f} bp = {active_oas / TE_BUDGET:.2f}")

    print("\nSimulated three-month hedged returns")
    print(f"{'':>16} {'mean':>7} {'st dev':>7} {'Sharpe':>7} {'skew':>6} {'1% ES':>7}")
    chaser = optimal_weights(x, reward=ZSPREAD)
    for name, weights in (("index", BENCHMARK), ("portfolio", w), ("static chaser", chaser)):
        r = hedged_returns(weights, x)
        skew, es, _, sd = tail(r)
        sharpe = r.mean() / sd * np.sqrt(4.0)            # quarterly to annual
        print(f"{name:>16} {r.mean():7.3%} {sd:7.3%} {sharpe:7.2f} {skew:6.2f} {es:7.2%}")


if __name__ == "__main__":
    main()
