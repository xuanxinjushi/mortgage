"""Leveraged and unleveraged MBS portfolios: what borrowing changes and what it does not.

An unleveraged portfolio owns its bonds outright. A leveraged one -- a mortgage
REIT, a hedge fund, a bank -- owns L times its equity, borrowing the rest in the
repo market, where the lender keeps a haircut. For a duration-hedged position
the return on equity is

    equity excess return = L x (asset excess return) - (L - 1) x funding spread.

Leverage multiplies the volatility and the tail one for one. It multiplies the
mean only by what the asset earns above the funding spread, so it leaves the
Sharpe ratio alone only when that margin is wide. A duration-hedged agency
pool's margin is thin, and the run shows what happens then. The haircut sets
how far leverage can go: the most a lender allows is 1 / haircut.

The asset is the chapter's index portfolio, duration-hedged, with the
three-month returns coupon_allocation.py simulates. The funding spread over the
risk-free rate and the haircut are illustrative.

    python code/leverage.py
"""

from __future__ import annotations

import numpy as np

from coupon_allocation import BENCHMARK, exposures, hedged_returns, tail

FUNDING_SPREAD = 0.0020          # repo over the risk-free rate, a year
HAIRCUT = 0.05                   # agency repo haircut, illustrative
HORIZON = 0.25                   # the simulation's three months


def main() -> None:
    x = exposures()
    asset = hedged_returns(BENCHMARK, x)
    duration = BENCHMARK @ x["duration"]

    print(f"Index portfolio, duration-hedged, three-month returns; haircut {HAIRCUT:.0%} "
          f"allows leverage up to {1 / HAIRCUT:.0f}x\n")
    print(f"{'leverage':>9} {'mean/yr':>8} {'st dev/yr':>10} {'Sharpe':>7} {'1% ES':>7} "
          f"{'P(lose 5%)':>11} {'unhedged dur':>13}")
    for lev in (1.0, 3.0, 7.0, 12.0):
        equity = lev * asset - (lev - 1.0) * FUNDING_SPREAD * HORIZON
        mean, sd = equity.mean() / HORIZON, equity.std() / np.sqrt(HORIZON)
        _, es, _, _ = tail(equity)
        lose = float(np.mean(equity < -0.05))
        print(f"{lev:8.0f}x {mean:8.2%} {sd:10.2%} {mean / sd:7.2f} {es:7.2%} {lose:11.2%} "
              f"{lev * duration:12.1f}y")

    print(f"\nIf the haircut doubles to {2 * HAIRCUT:.0%}, the most a lender allows falls from "
          f"{1 / HAIRCUT:.0f}x to {1 / (2 * HAIRCUT):.0f}x: a 12x book must sell "
          f"{1 - (1 / (2 * HAIRCUT)) / 12:.0%} of its assets, whatever the price.")


if __name__ == "__main__":
    main()
