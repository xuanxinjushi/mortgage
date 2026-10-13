"""Allocating across the coupon stack: factors, a tracking-error budget, and the tail.

Elena runs an MBS portfolio against an index. Her decision is not which pool to
buy but how much of each coupon to hold relative to the benchmark. This file
prices a five-coupon UMBS stack with the same pool model as
`duration_hedging.py`, measures each coupon's exposure to rates, spreads and
prepayment speeds, finds the allocation that earns the most OAS within a
tracking-error budget, and then shows what the variance-based budget misses:
a hedged MBS position is short an option, so its losses are skewed.

The spread quotes, benchmark weights and factor volatilities are illustrative.

Each coupon is quoted at a static, zero-volatility spread: discount one
projected cash-flow path at the market rate plus the spread, and that sets the
coupon's market price. The spread still contains the cost of the borrower's
option. To take it out, the script reprices the same cash flows on simulated
rate paths -- Chapter 6's method, on this chapter's pool model -- and solves
for the spread that matches the market price: the OAS. The difference is the
option cost, and the optimizer maximizes OAS, not the static spread.

Run from this chapter's folder:

    python code/coupon_allocation.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize

from duration_hedging import SERVICING, TERM, cpr_curve
from mortgagekit.mortgage import mortgage_payment
from mortgagekit.prepayment import cpr_to_smm

RATE = 0.065                                         # today's mortgage rate, as in the rest of the chapter
COUPONS = np.array([0.045, 0.050, 0.055, 0.060, 0.065])
WACS = COUPONS + SERVICING                           # the borrowers' note rate behind each coupon
ZSPREAD = np.array([40.0, 45.0, 55.0, 65.0, 60.0]) / 1e4  # illustrative static spreads
BENCHMARK = np.array([0.15, 0.20, 0.25, 0.25, 0.15])  # illustrative index weights

# Factor volatilities, per year, treated as independent: a 1-point move in
# rates, a 25 bp move in the spread, and prepayment speeds 15% away from forecast.
RATE_VOL, SPREAD_VOL, PREPAY_VOL = 0.0100, 0.0025, 0.15
IDIO_VOL = 0.005                                     # what the three factors leave unexplained
TE_BUDGET = 0.0015                                   # 15 bp a year of tracking error

# Rate paths for the OAS: a mean-reverting (Vasicek) mortgage rate around
# today's level, at the same 1-point annual volatility as the factor model.
PATHS, MEAN_REVERSION, PATH_SEED = 1000, 0.15, 11


def pool_price(wac: float, rate: float, spread: float, speed: float = 1.0) -> float:
    """Price of 100 of a pool with note rate `wac`, prepaying at `speed` x the S-curve."""
    smm = cpr_to_smm(min(cpr_curve(wac - rate) * speed, 0.99))
    balance, price = 100.0, 0.0
    for month in range(1, TERM + 1):
        if balance <= 1e-10:
            break
        payment = mortgage_payment(balance, wac, TERM - month + 1)   # re-amortized monthly
        interest = balance * wac / 12.0
        scheduled = min(payment - interest, balance)
        prepaid = (balance - scheduled) * smm
        flow = interest * (1.0 - SERVICING / wac) + scheduled + prepaid
        price += flow * np.exp(-(rate + spread) * month / 12.0)
        balance -= scheduled + prepaid
    return price


def rate_paths(n: int = PATHS, seed: int = PATH_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Monthly mortgage-rate paths, and discount factors fitted to today's flat curve.

    The raw path discount factors are rescaled month by month so that their
    average equals exp(-RATE * t): the paths reprice today's curve exactly, the
    condition Hull-White's theta(t) imposes in Chapter 4. Without it, the
    convexity of discounting alone would shift every OAS.
    """
    dt = 1.0 / 12.0
    eps = np.random.default_rng(seed).standard_normal((n, TERM))
    rates = np.empty((n, TERM))
    r = np.full(n, RATE)
    for m in range(TERM):
        r = r + MEAN_REVERSION * (RATE - r) * dt + RATE_VOL * np.sqrt(dt) * eps[:, m]
        rates[:, m] = r
    disc = np.exp(-np.cumsum(rates, axis=1) * dt)
    target = np.exp(-RATE * dt * np.arange(1, TERM + 1))
    return rates, disc * (target / disc.mean(axis=0))


def path_price(wac: float, rates: np.ndarray, disc: np.ndarray, oas: float) -> float:
    """Average price of 100 of the pool over the paths, discounted at the curve plus `oas`."""
    balance = np.full(rates.shape[0], 100.0)
    price = np.zeros(rates.shape[0])
    for month in range(1, TERM + 1):
        smm = cpr_to_smm(np.minimum(cpr_curve(wac - rates[:, month - 1]), 0.99))
        payment = mortgage_payment(balance, wac, TERM - month + 1)
        interest = balance * wac / 12.0
        scheduled = np.minimum(payment - interest, balance)
        prepaid = (balance - scheduled) * smm
        flow = interest * (1.0 - SERVICING / wac) + scheduled + prepaid
        price += flow * disc[:, month - 1] * np.exp(-oas * month / 12.0)
        balance -= scheduled + prepaid
    return float(price.mean())


def oas_quotes(prices: np.ndarray) -> np.ndarray:
    """The OAS that reprices each coupon's market price on the simulated paths."""
    rates, disc = rate_paths()
    return np.array([brentq(lambda o: path_price(wac, rates, disc, o) - price, -0.05, 0.05)
                     for wac, price in zip(WACS, prices)])


def exposures(bump: float = 0.0025) -> dict[str, np.ndarray]:
    """Per coupon: price, effective and spread duration, convexity, and prepayment exposure.

    Spread duration moves only the spread, so the cash flows stay put: it is the
    sensitivity behind Chapter 6's OAS01. Effective duration moves the rate, so
    the borrower re-decides too. Prepayment exposure is the return, in percent of
    price, if speeds run 15% faster than the S-curve says -- one standard
    deviation of the prepay factor.
    """
    out = {k: [] for k in ("price", "duration", "spread_duration", "convexity", "prepay")}
    for wac, spread in zip(WACS, ZSPREAD):
        base = pool_price(wac, RATE, spread)
        up, down = pool_price(wac, RATE + bump, spread), pool_price(wac, RATE - bump, spread)
        fast = pool_price(wac, RATE, spread, speed=1.0 + PREPAY_VOL)
        wide, tight = pool_price(wac, RATE, spread + 0.0001), pool_price(wac, RATE, spread - 0.0001)
        out["price"].append(base)
        out["duration"].append((down - up) / (2.0 * base * bump))
        out["spread_duration"].append((tight - wide) / (2.0 * base * 0.0001))
        out["convexity"].append((down + up - 2.0 * base) / (base * bump**2))
        out["prepay"].append((fast - base) / base)
    out = {k: np.array(v) for k, v in out.items()}
    out["oas"] = oas_quotes(out["price"])
    return out


def risk_parts(active: np.ndarray, x: dict[str, np.ndarray]) -> dict[str, float]:
    """Annual active variance from each factor; the factors are independent, so they add."""
    return {
        "rates": (active @ x["duration"] * RATE_VOL) ** 2,
        "spread": (active @ x["spread_duration"] * SPREAD_VOL) ** 2,
        "prepayment": (active @ x["prepay"]) ** 2,      # already one standard deviation
        "idiosyncratic": float(np.sum((active * IDIO_VOL) ** 2)),
    }


def tracking_error(active: np.ndarray, x: dict[str, np.ndarray]) -> float:
    """Annual tracking error of active weights, from the three factors plus noise."""
    return float(np.sqrt(sum(risk_parts(active, x).values())))


def optimal_weights(x: dict[str, np.ndarray], max_short_convexity: float | None = None,
                    reward: np.ndarray | None = None) -> np.ndarray:
    """Most OAS over the benchmark, duration-neutral, within the tracking-error budget.

    `max_short_convexity` optionally caps how much convexity the portfolio may
    sell relative to the benchmark -- the limit a variance budget cannot express.
    `reward` replaces the OAS, e.g. with the static spread, to show what chasing
    a spread that still contains the option does.
    """
    reward = x["oas"] if reward is None else reward
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w: (w - BENCHMARK) @ x["duration"]},
        {"type": "ineq", "fun": lambda w: TE_BUDGET - tracking_error(w - BENCHMARK, x)},
    ]
    if max_short_convexity is not None:
        cons.append({"type": "ineq",
                     "fun": lambda w: (w - BENCHMARK) @ x["convexity"] + max_short_convexity})
    res = minimize(lambda w: -(w - BENCHMARK) @ reward, BENCHMARK, method="SLSQP",
                   bounds=[(0.0, 0.5)] * len(COUPONS), constraints=cons)
    return res.x


def hedged_returns(weights: np.ndarray, x: dict[str, np.ndarray], n: int = 20_000,
                   horizon: float = 0.25, seed: int = 7) -> np.ndarray:
    """Three-month returns of the portfolio with its duration hedged in Treasuries.

    Each coupon is fully repriced after the rate shock; the hedge offsets only
    the duration. What is left is the spread carried and the convexity paid.
    """
    shocks = np.random.default_rng(seed).normal(0.0, RATE_VOL * np.sqrt(horizon), n)
    grid = np.linspace(-0.03, 0.03, 121)
    returns = np.zeros(n)
    for w, wac, spread, price, dur in zip(weights, WACS, ZSPREAD, x["price"], x["duration"]):
        curve = np.array([pool_price(wac, RATE + g, spread) for g in grid])
        repriced = np.interp(shocks, grid, curve)
        returns += w * ((repriced - price) / price + dur * shocks + spread * horizon)
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
    print(f"{'coupon':>7} {'price':>7} {'Z bp':>5} {'OAS bp':>7} {'option':>7} {'eff dur':>8} {'spr dur':>8} {'convex':>7} {'prepay':>7}")
    for c, p, z, o, d, sd, cv, pp in zip(COUPONS, x["price"], ZSPREAD, x["oas"], x["duration"],
                                          x["spread_duration"], x["convexity"], x["prepay"]):
        print(f"{c:7.1%} {p:7.2f} {z * 1e4:5.0f} {o * 1e4:7.1f} {(z - o) * 1e4:7.1f} {d:8.2f} {sd:8.2f} {cv:7.1f} {pp:7.2%}")

    w = optimal_weights(x)
    active = w - BENCHMARK
    print(f"\n{'coupon':>7} {'bench':>6} {'optimal':>8} {'active':>7}")
    for c, b, wi, a in zip(COUPONS, BENCHMARK, w, active):
        print(f"{c:7.1%} {b:6.0%} {wi:8.1%} {a:+7.1%}")
    print(f"\nActive OAS:        {active @ x['oas'] * 1e4:+.1f} bp a year"
          f" (static spread {active @ ZSPREAD * 1e4:+.1f})")
    print(f"Active duration:   {active @ x['duration']:+.2f} years")
    print(f"Active convexity:  {active @ x['convexity']:+.1f}")
    print(f"Tracking error:    {tracking_error(active, x) * 1e4:.0f} bp a year (budget {TE_BUDGET * 1e4:.0f})")

    print("\nWhere the tracking error comes from (share of active variance):")
    print(f"  {'':<18}" + "".join(f"{k:>14}" for k in ("rates", "spread", "prepayment", "idiosyncratic")))
    for label, weights in (("most OAS", w), ("most static spread", optimal_weights(x, reward=ZSPREAD))):
        parts = risk_parts(weights - BENCHMARK, x)
        total = sum(parts.values())
        print(f"  {label:<18}" + "".join(f"{v / total:14.0%}" for v in parts.values()))

    chaser = optimal_weights(x, reward=ZSPREAD)
    chaser_active = chaser - BENCHMARK
    print(f"\nChasing the static spread instead: static spread {chaser_active @ ZSPREAD * 1e4:+.1f} bp, "
          f"OAS {chaser_active @ x['oas'] * 1e4:+.1f} bp, "
          f"active convexity {chaser_active @ x['convexity']:+.1f}")
    print("  weights " + " ".join(f"{c:.1%} {wi:.1%}" for c, wi in zip(COUPONS, chaser)))

    print(f"\nThree-month hedged returns, 1% tail:  {'skew':>5} {'ES sim':>7} {'ES normal':>9}")
    for label, weights in (("benchmark", BENCHMARK), ("most OAS", w), ("most static spread", chaser)):
        skew, es, es_normal, _ = tail(hedged_returns(weights, x))
        print(f"  {label:<18} {skew:+5.2f} {es:7.2%} {es_normal:9.2%}")
