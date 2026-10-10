"""Two current coupon models: a constant spread, and a constant OAS.

The engine in `oas_engine.py` quotes the borrower a 10-year rate plus a fixed
spread -- a constant-spread current coupon model. Production analytics also use
an option-adjusted one, whose published idea (Citigroup's MOATS, now in LSEG's
Yield Book) is: take today's OAS of the current-coupon TBA and hold it fixed;
on every rate path, the current coupon is the coupon that prices a new pool at
par at that OAS. This file builds a simplified version of that idea on the
chapter's own model and compares the two.

Simplifications, all stated: the curve is the chapter's one-factor Vasicek
model, so the current coupon is a function of the short rate alone; a new pool's
note rate is its coupon plus the 0.5% servicing strip, and the borrower is
quoted that same note rate; the prepayment curve is the engine's S-curve.

Run from this chapter's folder:

    python code/current_coupon.py
"""

from __future__ import annotations

import math

import numpy as np

from oas_engine import (
    A, MORTGAGE_SPREAD_OVER_CURVE, R0, SIGMA_R, THETA, primary_mortgage_rate,
)

STRIP = 0.005            # servicing + guarantee fee: note rate = coupon + strip
MONTHS = 360
N_PATHS = 600            # per valuation; antithetic pairs
GRID = np.round(np.linspace(-0.01, 0.09, 11), 4) + 0.0   # short-rate levels the model is solved on (+0.0 clears -0.0)


def smm(incentive: np.ndarray) -> np.ndarray:
    """The engine's refinance S-curve, vectorized."""
    cpr = 0.06 + (0.45 - 0.06) / (1.0 + np.exp(-100.0 * (incentive - 0.0125)))
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def short_rate_paths(r_start: float, sigma: float, n: int = N_PATHS, seed: int = 11) -> np.ndarray:
    """Vasicek paths from `r_start`, with antithetic shocks to steady the estimate."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n // 2, MONTHS))
    z = np.vstack([z, -z])
    dt = 1.0 / 12.0
    r = np.empty((n, MONTHS + 1))
    r[:, 0] = r_start
    for k in range(MONTHS):
        r[:, k + 1] = r[:, k] + A * (THETA - r[:, k]) * dt + sigma * math.sqrt(dt) * z[:, k]
    return r


def pool_value(paths: np.ndarray, coupon: float, oas: float, quote) -> float:
    """Price per 100 of a new pool paying `coupon`, its borrowers refinancing
    against `quote(r_t)`, discounted at r_t + oas. Vectorized over paths."""
    note = coupon + STRIP
    i_note, i_cpn, dt = note / 12.0, coupon / 12.0, 1.0 / 12.0
    pmt = 100.0 * i_note / (1.0 - (1.0 + i_note) ** (-MONTHS))
    bal = np.full(paths.shape[0], 100.0)
    disc = np.ones(paths.shape[0])
    value = np.zeros(paths.shape[0])
    for k in range(1, MONTHS + 1):
        r_t = paths[:, k]
        disc *= np.exp(-(r_t + oas) * dt)
        sched = np.minimum(pmt - bal * i_note, bal)
        prepaid = (bal - sched) * smm(note - quote(r_t))
        value += (sched + prepaid + bal * i_cpn) * disc
        bal = bal - sched - prepaid
    return float(value.mean())


def bisect(f, lo: float, hi: float, iters: int = 40) -> float:
    """Root of an increasing-then-decreasing-free f on [lo, hi] (f(lo) and f(hi) differ in sign)."""
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def constant_spread_cc(r, sigma: float = SIGMA_R):
    """The engine's model: 10-year Vasicek rate plus the measured spread, less the strip."""
    r = np.atleast_1d(r)
    return np.array([primary_mortgage_rate(x, sigma_r=sigma) for x in r]) - STRIP


def calibrate_oas(cc_today: float, sigma: float, quote) -> float:
    """The OAS at which today's current-coupon pool prices at par."""
    paths = short_rate_paths(R0, sigma)
    return bisect(lambda s: pool_value(paths, cc_today, s, quote) - 100.0, -0.02, 0.05)


def constant_oas_cc(oas: float, sigma: float, rounds: int = 3) -> np.ndarray:
    """Current coupon on GRID at a fixed OAS, iterated to a fixed point.

    The new pool's borrowers refinance against the current coupon itself, so
    the model appears inside its own valuation. Start from the constant-spread
    model and re-solve until the function stops moving.
    """
    cc = constant_spread_cc(GRID, sigma)
    for _ in range(rounds):
        quote = lambda r_t, cc=cc: np.interp(r_t, GRID, cc) + STRIP
        new = []
        for r in GRID:
            paths = short_rate_paths(r, sigma)
            new.append(bisect(lambda c: pool_value(paths, c, oas, quote) - 100.0, 0.0, 0.15))
        cc = np.array(new)
    return cc


if __name__ == "__main__":
    cc_today = float(constant_spread_cc(R0)[0])
    cs_quote = lambda r_t: np.asarray(constant_spread_cc(r_t)) + STRIP
    # Calibrate once against the constant-spread quotes, then let the fixed point
    # take over; recalibrate on the model's own quotes so CC(today) is matched.
    oas = calibrate_oas(cc_today, SIGMA_R, cs_quote)
    cc_oas = constant_oas_cc(oas, SIGMA_R)
    oas = calibrate_oas(cc_today, SIGMA_R, lambda r_t: np.interp(r_t, GRID, cc_oas) + STRIP)
    cc_oas = constant_oas_cc(oas, SIGMA_R)

    print(f"Today: short rate {R0:.2%}, current coupon {cc_today:.2%}, "
          f"calibrated OAS {oas * 1e4:.0f} bp (sigma {SIGMA_R:.1%})")
    print(f"\n{'short rate':>10} {'const spread':>13} {'const OAS':>10} {'gap bp':>7}")
    cc_cs = constant_spread_cc(GRID)
    for r, a_, b_ in zip(GRID, cc_cs, cc_oas):
        print(f"{r:10.1%} {a_:13.2%} {b_:10.2%} {(b_ - a_) * 1e4:7.0f}")

    # What the choice does to the chapter's own pool: a 6.0% coupon at par,
    # with its borrowers quoted by each model. Duration is a parallel move of the
    # whole curve, as in oas_engine.compute_oas_metrics: every short rate and the
    # quote both move by dy, so the two models differ only in how the quote
    # varies along the simulated paths.
    paths = short_rate_paths(R0, SIGMA_R, n=2000, seed=7)
    oas_quote = lambda r_t: np.interp(r_t, GRID, cc_oas) + STRIP
    print(f"\nThe chapter's 6.0% pool at par:")
    print(f"{'quote model':>13} {'OAS bp':>7} {'eff dur':>8}")
    for label, quote in (("const spread", cs_quote), ("const OAS", oas_quote)):
        pool_oas = bisect(lambda x: pool_value(paths, 0.060, x, quote) - 100.0, -0.02, 0.06)
        up = pool_value(paths + 0.001, 0.060, pool_oas, lambda r_t: quote(r_t - 0.001) + 0.001)
        down = pool_value(paths - 0.001, 0.060, pool_oas, lambda r_t: quote(r_t + 0.001) - 0.001)
        print(f"{label:>13} {pool_oas * 1e4:7.0f} {(down - up) / (2 * 100.0 * 0.001):8.2f}")

    print(f"\nVolatility, same short rate and same OAS:")
    print(f"{'sigma':>6} {'const spread':>13} {'const OAS':>10}")
    for sigma in (0.015, 0.025, 0.035):
        cc_s = constant_oas_cc(oas, sigma)
        print(f"{sigma:6.1%} {constant_spread_cc(R0, sigma)[0]:13.2%} {np.interp(R0, GRID, cc_s):10.2%}")
