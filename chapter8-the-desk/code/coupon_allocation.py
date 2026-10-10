"""Allocating across the coupon stack: factors, a tracking-error budget, and the tail.

Elena runs an MBS portfolio against an index. Her decision is not which pool to
buy but how much of each coupon to hold relative to the benchmark. This file
prices a five-coupon UMBS stack with the same pool model as
`duration_hedging.py`, measures each coupon's exposure to rates, spreads and
prepayment speeds, finds the allocation that earns the most OAS within a
tracking-error budget, and then shows what the variance-based budget misses:
a hedged MBS position is short an option, so its losses are skewed.

The OAS quotes, benchmark weights and factor volatilities are illustrative.

Run from this chapter's folder:

    python code/coupon_allocation.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from duration_hedging import SERVICING, TERM, cpr_curve
from mortgagekit.mortgage import mortgage_payment
from mortgagekit.prepayment import cpr_to_smm

RATE = 0.065                                         # today's mortgage rate, as in the rest of the chapter
COUPONS = np.array([0.045, 0.050, 0.055, 0.060, 0.065])
WACS = COUPONS + SERVICING                           # the borrowers' note rate behind each coupon
OAS = np.array([40.0, 45.0, 55.0, 65.0, 60.0]) / 1e4  # illustrative quotes
BENCHMARK = np.array([0.15, 0.20, 0.25, 0.25, 0.15])  # illustrative index weights

# Factor volatilities, per year, treated as independent: a 1-point move in
# rates, a 25 bp move in OAS, and prepayment speeds 15% away from forecast.
RATE_VOL, SPREAD_VOL, PREPAY_VOL = 0.0100, 0.0025, 0.15
IDIO_VOL = 0.005                                     # what the three factors leave unexplained
TE_BUDGET = 0.0015                                   # 15 bp a year of tracking error


def pool_price(wac: float, rate: float, oas: float, speed: float = 1.0) -> float:
    """Price of 100 of a pool with note rate `wac`, prepaying at `speed` x the S-curve."""
    smm = cpr_to_smm(min(cpr_curve(wac - rate) * speed, 0.99))
    balance, payment, price = 100.0, mortgage_payment(100.0, wac, TERM), 0.0
    for month in range(1, TERM + 1):
        if balance <= 1e-10:
            break
        interest = balance * wac / 12.0
        scheduled = min(payment - interest, balance)
        prepaid = (balance - scheduled) * smm
        flow = interest * (1.0 - SERVICING / wac) + scheduled + prepaid
        price += flow * np.exp(-(rate + oas) * month / 12.0)
        balance -= scheduled + prepaid
    return price


def exposures(bump: float = 0.0025) -> dict[str, np.ndarray]:
    """Per coupon: price, effective duration and convexity, and prepayment exposure.

    Prepayment exposure is the return, in percent of price, if speeds run 15%
    faster than the S-curve says -- one standard deviation of the prepay factor.
    """
    out = {k: [] for k in ("price", "duration", "convexity", "prepay")}
    for wac, oas in zip(WACS, OAS):
        base = pool_price(wac, RATE, oas)
        up, down = pool_price(wac, RATE + bump, oas), pool_price(wac, RATE - bump, oas)
        fast = pool_price(wac, RATE, oas, speed=1.0 + PREPAY_VOL)
        out["price"].append(base)
        out["duration"].append((down - up) / (2.0 * base * bump))
        out["convexity"].append((down + up - 2.0 * base) / (base * bump**2))
        out["prepay"].append((fast - base) / base)
    return {k: np.array(v) for k, v in out.items()}


def tracking_error(active: np.ndarray, x: dict[str, np.ndarray]) -> float:
    """Annual tracking error of active weights, from the three factors plus noise."""
    rate = active @ x["duration"] * RATE_VOL          # spread duration ~ effective duration
    spread = active @ x["duration"] * SPREAD_VOL
    prepay = active @ x["prepay"]                     # already one standard deviation
    idio = np.sqrt(np.sum((active * IDIO_VOL) ** 2))
    return float(np.sqrt(rate**2 + spread**2 + prepay**2 + idio**2))


def optimal_weights(x: dict[str, np.ndarray], max_short_convexity: float | None = None) -> np.ndarray:
    """Most OAS over the benchmark, duration-neutral, within the tracking-error budget.

    `max_short_convexity` optionally caps how much convexity the portfolio may
    sell relative to the benchmark -- the limit a variance budget cannot express.
    """
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w: (w - BENCHMARK) @ x["duration"]},
        {"type": "ineq", "fun": lambda w: TE_BUDGET - tracking_error(w - BENCHMARK, x)},
    ]
    if max_short_convexity is not None:
        cons.append({"type": "ineq",
                     "fun": lambda w: (w - BENCHMARK) @ x["convexity"] + max_short_convexity})
    res = minimize(lambda w: -(w - BENCHMARK) @ OAS, BENCHMARK, method="SLSQP",
                   bounds=[(0.0, 0.5)] * len(COUPONS), constraints=cons)
    return res.x


def hedged_returns(weights: np.ndarray, x: dict[str, np.ndarray], n: int = 20_000,
                   horizon: float = 0.25, seed: int = 7) -> np.ndarray:
    """Three-month returns of the portfolio with its duration hedged in Treasuries.

    Each coupon is fully repriced after the rate shock; the hedge offsets only
    the duration. What is left is the OAS carried and the convexity paid.
    """
    shocks = np.random.default_rng(seed).normal(0.0, RATE_VOL * np.sqrt(horizon), n)
    grid = np.linspace(-0.03, 0.03, 121)
    returns = np.zeros(n)
    for w, wac, oas, price, dur in zip(weights, WACS, OAS, x["price"], x["duration"]):
        curve = np.array([pool_price(wac, RATE + g, oas) for g in grid])
        repriced = np.interp(shocks, grid, curve)
        returns += w * ((repriced - price) / price + dur * shocks + oas * horizon)
    return returns


def tail(returns: np.ndarray, level: float = 0.01) -> tuple[float, float, float, float]:
    """Skewness, simulated 1% expected shortfall, and what a normal fit would say."""
    mu, sd = returns.mean(), returns.std()
    skew = float(np.mean(((returns - mu) / sd) ** 3))
    cutoff = np.quantile(returns, level)
    es = float(returns[returns <= cutoff].mean())
    z = 2.3263                                       # standard normal 1% quantile
    es_normal = float(mu - sd * np.exp(-z * z / 2) / (np.sqrt(2 * np.pi) * level))
    return skew, es, es_normal, float(sd)


if __name__ == "__main__":
    x = exposures()
    print(f"{'coupon':>7} {'price':>7} {'OAS bp':>7} {'eff dur':>8} {'convex':>7} {'prepay':>7}")
    for c, p, o, d, cv, pp in zip(COUPONS, x["price"], OAS, x["duration"], x["convexity"], x["prepay"]):
        print(f"{c:7.1%} {p:7.2f} {o * 1e4:7.0f} {d:8.2f} {cv:7.1f} {pp:7.2%}")

    w = optimal_weights(x)
    active = w - BENCHMARK
    print(f"\n{'coupon':>7} {'bench':>6} {'optimal':>8} {'active':>7}")
    for c, b, wi, a in zip(COUPONS, BENCHMARK, w, active):
        print(f"{c:7.1%} {b:6.0%} {wi:8.1%} {a:+7.1%}")
    print(f"\nActive OAS:        {active @ OAS * 1e4:+.1f} bp a year")
    print(f"Active duration:   {active @ x['duration']:+.2f} years")
    print(f"Active convexity:  {active @ x['convexity']:+.1f}")
    print(f"Tracking error:    {tracking_error(active, x) * 1e4:.0f} bp a year (budget {TE_BUDGET * 1e4:.0f})")

    capped = optimal_weights(x, max_short_convexity=20.0)
    capped_active = capped - BENCHMARK
    print(f"\nWith active convexity capped at -20: active OAS {capped_active @ OAS * 1e4:+.1f} bp, "
          f"tracking error {tracking_error(capped_active, x) * 1e4:.0f} bp, "
          f"active convexity {capped_active @ x['convexity']:+.1f}")

    print(f"\nThree-month hedged returns, 1% tail:  {'skew':>5} {'ES sim':>7} {'ES normal':>9}")
    for label, weights in (("benchmark", BENCHMARK), ("optimal", w), ("convexity-capped", capped)):
        skew, es, es_normal, _ = tail(hedged_returns(weights, x))
        print(f"  {label:<17} {skew:+5.2f} {es:7.2%} {es_normal:9.2%}")
