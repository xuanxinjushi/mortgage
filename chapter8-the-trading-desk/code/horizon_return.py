"""Horizon total return: the scenario table every MBS portfolio strategy is read from.

Buy 100 of each coupon today, let rates jump by a parallel shift, hold six
months, and add up what you have: the cash the pool paid (interest, scheduled
principal, and prepayments at par, all at the speed the new rate implies),
reinvested to the horizon, plus what the remaining balance is worth then at an
unchanged spread. Divide by what you paid. That is the total-return framework of
the bond portfolio literature, applied to the chapter's own coupon stack and
pool model from coupon_allocation.py.

The second table subtracts the return of a Treasury zero whose maturity equals
the coupon's effective duration -- the excess return that MBS indices report.
It removes the duration bet and leaves what an MBS strategy is really about:
carry against convexity. It is also an immunization test: a liability due at
the duration date, funded with the pool, ends up short in both directions.

    python code/horizon_return.py
"""

from __future__ import annotations

import numpy as np

from coupon_allocation import BENCHMARK, COUPONS, RATE, WACS, ZSPREAD, exposures
from duration_hedging import SERVICING, TERM, cpr_curve
from mortgagekit.mortgage import mortgage_payment
from mortgagekit.prepayment import cpr_to_smm

HORIZON = 6                                            # months
SHIFTS = np.array([-100, -50, 0, 50, 100]) / 1e4      # instantaneous parallel moves

PORTFOLIOS = {
    "index":          BENCHMARK,
    "bullet 5.5%":    np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    "barbell 4.5/6.5": np.array([0.5, 0.0, 0.0, 0.0, 0.5]),
    "up in coupon":   np.array([0.0, 0.20, 0.25, 0.40, 0.15]),
}


def pool_flows(wac: float, rate: float) -> np.ndarray:
    """Monthly pass-through cash flows of 100 of a pool, at the speed `rate` implies.

    Same arithmetic as coupon_allocation.pool_price, so today's prices match
    the coupon-stack table exactly.
    """
    smm = cpr_to_smm(min(cpr_curve(wac - rate), 0.99))
    balance, flows = 100.0, []
    for month in range(1, TERM + 1):
        if balance <= 1e-10:
            break
        payment = mortgage_payment(balance, wac, TERM - month + 1)   # re-amortized monthly
        interest = balance * wac / 12.0
        scheduled = min(payment - interest, balance)
        prepaid = (balance - scheduled) * smm
        flows.append(interest * (1.0 - SERVICING / wac) + scheduled + prepaid)
        balance -= scheduled + prepaid
    return np.array(flows)


def price_today(wac: float, spread: float) -> float:
    flows = pool_flows(wac, RATE)
    months = np.arange(1, len(flows) + 1)
    return float(np.sum(flows * np.exp(-(RATE + spread) * months / 12.0)))


def horizon_return(wac: float, spread: float, shift: float) -> float:
    """Six-month total return of 100 bought today, rates shifted by `shift`, spread unchanged."""
    rate = RATE + shift
    flows = pool_flows(wac, rate)
    months = np.arange(1, len(flows) + 1)
    early, late = months <= HORIZON, months > HORIZON
    # cash received before the horizon, reinvested at the new rate until then
    received = np.sum(flows[early] * np.exp(rate * (HORIZON - months[early]) / 12.0))
    # what the rest of the pool is worth at the horizon, at the same spread
    held = np.sum(flows[late] * np.exp(-(rate + spread) * (months[late] - HORIZON) / 12.0))
    return (received + held) / price_today(wac, spread) - 1.0


def treasury_zero_return(duration: float, shift: float) -> float:
    """Six-month return of a zero maturing at `duration` years, yield RATE then RATE + shift."""
    h = HORIZON / 12.0
    return float(np.exp(RATE * duration - (RATE + shift) * (duration - h)) - 1.0)


def main() -> None:
    x = exposures()
    total = np.array([[horizon_return(w, o, s) for s in SHIFTS] for w, o in zip(WACS, ZSPREAD)])
    excess = np.array([[total[i, j] - treasury_zero_return(x["duration"][i], s)
                        for j, s in enumerate(SHIFTS)] for i in range(len(COUPONS))])

    head = "".join(f"{s * 1e4:+8.0f}" for s in SHIFTS)
    print(f"Six-month total return, %, by parallel shift (bp), spread held constant")
    print(f"{'coupon':>16}{head}")
    for c, row in zip(COUPONS, total):
        print(f"{c:16.1%}" + "".join(f"{r * 100:8.2f}" for r in row))

    print(f"\nExcess over a duration-matched Treasury zero, bp")
    print(f"{'coupon':>16}{head}   dur  convex")
    for c, row, d, cv in zip(COUPONS, excess, x["duration"], x["convexity"]):
        print(f"{c:16.1%}" + "".join(f"{r * 1e4:8.0f}" for r in row) + f"  {d:4.2f} {cv:7.1f}")

    print(f"\nPortfolios: excess return, bp")
    print(f"{'':>16}{head}   dur  convex spread")
    for name, w in PORTFOLIOS.items():
        row = w @ excess
        print(f"{name:>16}" + "".join(f"{r * 1e4:8.0f}" for r in row)
              + f"  {w @ x['duration']:4.2f} {w @ x['convexity']:7.1f} {w @ ZSPREAD * 1e4:6.1f}")


if __name__ == "__main__":
    main()
