"""Rebalancing an MBS portfolio: why it drifts every month, and what holding still costs.

Elena sets her coupon weights to the optimizer's target and then leaves them.
Three things move them anyway, every month: the pools pay down, the high
coupons fastest, and that cash sits uninvested; prices move with rates, by each
coupon's duration and convexity; and the index adds a month of new production
to the current coupon. Her active weights drift away from the target, and the
drift is tracking error she did not choose.

The run follows one year of simulated rates under five policies -- rebalance
fully every month or every quarter; only reinvest the paydown cash each month,
into whatever coupons are now underweight; reinvest it that way and rebalance
fully only when drift tracking error passes 5 bp;
and never trade -- and reports what each costs and what it leaves in drift.
Trades cost half a 2/32 bid-ask spread on what is traded. Paydown cash is
reinvested only when the portfolio rebalances; until then it is cash.

    python code/rebalancing.py
"""

from __future__ import annotations

import numpy as np

from coupon_allocation import (BENCHMARK, RATE, RATE_VOL, WACS, ZSPREAD, exposures,
                               optimal_weights, tracking_error)
from duration_hedging import cpr_curve
from mortgagekit.prepayment import cpr_to_smm

MONTHS = 12
SCHEDULED = 0.0015                 # scheduled principal, a month, as a share of balance
NEW_ISSUANCE = 0.012               # new production added to the 6.5% coupon each month, share of index
COST = 1.0 / 32.0 / 100.0          # half of a 2/32 bid-ask, per unit of value traded
BAND = 0.0005                      # rebalance when drift tracking error passes 5 bp
PATHS, SEED = 200, 3


def month_step(values: np.ndarray, rate: float, shock: float, x: dict) -> tuple[np.ndarray, float]:
    """One month for holdings by coupon: carry, price move, and paydown. Returns (values, cash)."""
    carry = (RATE + ZSPREAD) / 12.0
    price = -x["duration"] * shock + 0.5 * x["convexity"] * shock**2
    grown = values * (1.0 + carry + price)
    paid = grown * np.array([cpr_to_smm(cpr_curve(w - rate)) + SCHEDULED for w in WACS])
    return grown - paid, float(paid.sum())


def drift_te(values: np.ndarray, cash: float, index: np.ndarray, target_active: np.ndarray,
             x: dict) -> float:
    """Tracking error of today's active weights against the target active weights.

    Cash is a sixth holding with no duration and no spread: being short the
    index's duration is a rate bet, and the factor model prices it as one.
    """
    total = values.sum() + cash
    weights, idx = values / total, index / index.sum()
    miss = (weights - idx) - target_active
    cash_share = cash / total
    xx = {k: v for k, v in x.items()}
    te_coupons = tracking_error(miss, xx)
    rate_gap = cash_share * (idx @ x["duration"]) * RATE_VOL
    return float(np.sqrt(te_coupons**2 + rate_gap**2))


def run(policy: str, shocks: np.ndarray, x: dict, target_active: np.ndarray) -> tuple[float, float, float, int]:
    """Turnover share, cost in bp, average and worst drift tracking error, number of rebalances."""
    values, index, cash = BENCHMARK + target_active, BENCHMARK.copy(), 0.0
    rate, traded, rebalances, drift = RATE, 0.0, 0, []
    for m in range(MONTHS):
        values, paid = month_step(values, rate, shocks[m], x)
        index, _ = month_step(index, rate, shocks[m], x)
        index[-1] += NEW_ISSUANCE * index.sum()      # new production joins the current coupon
        cash += paid
        rate += shocks[m]
        te = drift_te(values, cash, index, target_active, x)
        if policy in ("reinvest", "band") and cash > 0.0:
            # cash-flow rebalancing: the paydown cash goes to whatever is now underweight
            total = values.sum() + cash
            short = np.maximum(total * (index / index.sum() + target_active) - values, 0.0)
            buy = short * min(1.0, cash / short.sum()) if short.sum() > 0 else 0.0 * short
            values, cash = values + buy, cash - float(np.sum(buy))
            traded += float(np.sum(buy)) / total
            te = drift_te(values, cash, index, target_active, x)
        due = (policy == "monthly" or (policy == "quarterly" and m % 3 == 2)
               or (policy == "band" and te > BAND))
        if due:
            total = values.sum() + cash
            wanted = total * (index / index.sum() + target_active)
            traded += np.abs(wanted - values).sum() / total
            values, cash, rebalances = wanted, 0.0, rebalances + 1
            te = 0.0
        drift.append(te)
    return traded, traded * COST * 1e4, float(np.mean(drift)), float(np.max(drift)), rebalances


def main() -> None:
    x = exposures()
    target_active = optimal_weights(x) - BENCHMARK
    rng = np.random.default_rng(SEED)
    shocks = rng.normal(0.0, RATE_VOL / np.sqrt(12.0), (PATHS, MONTHS))

    print(f"One year, {PATHS} rate paths; drift tracking error in bp a year, cost in bp of the portfolio\n")
    print(f"{'policy':>14} {'rebalances':>11} {'turnover':>9} {'cost bp':>8} {'avg drift':>10} {'worst':>7}")
    for policy in ("monthly", "quarterly", "reinvest", "band", "never"):
        res = np.array([run(policy, shocks[p], x, target_active) for p in range(PATHS)])
        turnover, cost, avg, worst, n = res.mean(axis=0)
        print(f"{policy:>14} {n:11.1f} {turnover:9.0%} {cost:8.1f} {avg * 1e4:10.1f} {worst * 1e4:7.1f}")

    _, paid = month_step(BENCHMARK + target_active, RATE, 0.0, x)
    print(f"\nIn a month with no rate move, {paid:.1%} of the portfolio comes back as cash "
          f"from prepayments and scheduled principal: {1 - (1 - paid) ** 12:.0%} a year.")


if __name__ == "__main__":
    main()
