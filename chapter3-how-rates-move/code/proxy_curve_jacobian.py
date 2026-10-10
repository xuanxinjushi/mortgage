"""Proxy curves in the forward basis, and why the Jacobian carries the hedge.

The chapter stands a mortgage-basis curve on top of the Treasury curve in two
ways -- a spread on the levels, or a basis on the forwards -- and claims that
the second carries the pool's risk back onto bonds Benny can trade through a
chain of Jacobians. This file checks each link of that claim with numbers:

1. A spread interpolated on the levels lands on the forwards as s + T s',
   which jumps wherever the spread's slope does.
2. A pool's risk at an unquoted 12-year point splits onto the 10- and 30-year
   knots by the tent weights 0.9 and 0.1, exactly.
3. The bootstrap's own Jacobian, d(knots)/d(par quotes), taken from the
   implicit function theorem without rebuilding anything, matches a brute
   bump-and-rebuild.
4. In par-quote coordinates each hedge bond loads on its own quote only, so
   the hedge is a division, not a solve -- and it neutralises random curve
   shocks to second order.

    python code/proxy_curve_jacobian.py

Reads the pinned Treasury snapshot in `data/`, like `bootstrap_curve.py`.
"""

from __future__ import annotations

import math

import numpy as np

from mortgagekit import Curve, bootstrap_par_curve
from mortgagekit.marketdata import par_curve

# Benny's hedge set: bills, then the on-the-run notes and the long bond. The
# 20-year is left out on purpose -- it is the least liquid point on the
# Treasury curve, so the 12-year point falls between the 10- and 30-year knots.
HEDGE_TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0)
TARGET_TENOR = 12.0
POOL_DV01 = 100.0  # dollars per basis point at the 12-year point
BASIS = 0.0166  # the mortgage-Treasury spread, held constant in forward space
FREQUENCY = 2
BP = 1e-4


def coupon_times(tenor: float) -> list[float]:
    period = 1.0 / FREQUENCY
    n = int(round(tenor * FREQUENCY))
    return [tenor - period * k for k in range(n - 1, -1, -1)]


def annuity(curve: Curve, tenor: float) -> float:
    """Sum of coupon-date discount factors over the frequency: dPrice/dCoupon."""
    return sum(curve.discount_factor(t) for t in coupon_times(tenor)) / FREQUENCY


def par_residuals(zeros, tenors, quotes) -> np.ndarray:
    """G(z, q): each instrument's price minus par, on the curve with knots z.

    The bootstrap is exactly the statement G(z, q) = 0. Bills follow the same
    money-market convention `bootstrap_par_curve` uses.
    """
    curve = Curve(times=tuple(tenors), zeros=tuple(zeros))
    out = []
    for tenor, quote in zip(tenors, quotes):
        if tenor <= 1.0 / FREQUENCY + 1e-12:
            out.append(curve.discount_factor(tenor) * (1.0 + quote * tenor) - 1.0)
        else:
            out.append(quote * annuity(curve, tenor) + curve.discount_factor(tenor) - 1.0)
    return np.array(out)


def build(quotes) -> Curve:
    return bootstrap_par_curve(list(zip(HEDGE_TENORS, quotes)), tolerance=1e-15)


def forward(curve_zero, t: float, h: float = 1e-5) -> float:
    """Instantaneous forward f(t) = d(t z(t))/dt, by a central difference."""
    return ((t + h) * curve_zero(t + h) - (t - h) * curve_zero(t - h)) / (2.0 * h)


def level_spread(t: float) -> float:
    """An illustrative proxy spread known at three pillars, linear between them."""
    return float(np.interp(t, [2.0, 10.0, 30.0], [0.0100, 0.0180, 0.0160]))


def main() -> None:
    snapshot = dict(par_curve().as_pairs())
    quotes = np.array([snapshot[t] for t in HEDGE_TENORS])
    bench = build(quotes)
    n = len(HEDGE_TENORS)
    k10, k30 = HEDGE_TENORS.index(10.0), HEDGE_TENORS.index(30.0)

    # --- 1. a level spread, seen on the forwards ------------------------------
    print("Direct method: a spread on the levels, read on the forwards (bp)\n")
    print(f"{'t':>7} {'level s':>9} {'fwd spread':>11} {"s + T*s'":>10}")
    for t in (3.0, 6.0, 9.99, 10.01, 20.0, 29.99):
        target_zero = lambda u: bench.zero(u) + level_spread(u)
        fwd_spread = forward(target_zero, t) - forward(bench.zero, t)
        slope = (level_spread(t + 1e-4) - level_spread(t - 1e-4)) / 2e-4
        print(f"{t:7.2f} {level_spread(t) / BP:9.0f} {fwd_spread / BP:11.0f} "
              f"{(level_spread(t) + t * slope) / BP:10.0f}")

    # --- 2. pool risk at the 12-year point, by key-rate bucket ---------------
    # A constant basis in forward space is a constant on every zero rate, so
    # the target curve's 12-year zero is the benchmark's plus BASIS, and the
    # map from benchmark knots to target point has Jacobian = identity.
    t12 = TARGET_TENOR
    face = POOL_DV01 / (t12 * math.exp(-(bench.zero(t12) + BASIS) * t12) * BP)

    def pool_value(curve: Curve) -> float:
        return face * math.exp(-(curve.zero(t12) + BASIS) * t12)

    v0 = pool_value(bench)
    # Central bumps, so the reported DV01 carries no convexity of its own.
    buckets = np.array([
        (pool_value(bench.bumped(k, -1.0)) - pool_value(bench.bumped(k, 1.0))) / 2.0
        for k in range(n)
    ])
    print(f"\nPool DV01 at the {t12:.0f}-year point, by key-rate bucket ($ per bp)\n")
    print(f"  10y knot {buckets[k10]:7.2f}   30y knot {buckets[k30]:7.2f}"
          f"   all others {np.abs(np.delete(buckets, [k10, k30])).max():.2f}"
          f"   total {buckets.sum():7.2f}")

    # --- 3. the build Jacobian, from the implicit function theorem -----------
    # G(z, q) = 0  =>  dz/dq = -(dG/dz)^-1 (dG/dq). dG/dq is diagonal, because
    # each quote sits in one instrument's price; dG/dz is lower triangular,
    # because an instrument only sees the curve out to its own maturity.
    z0 = np.array(bench.zeros)
    h = 1e-7
    dG_dz = np.column_stack([
        (par_residuals(z0 + h * np.eye(n)[j], HEDGE_TENORS, quotes)
         - par_residuals(z0 - h * np.eye(n)[j], HEDGE_TENORS, quotes)) / (2 * h)
        for j in range(n)
    ])
    dG_dq = np.diag([
        bench.discount_factor(t) * t if t <= 1.0 / FREQUENCY + 1e-12 else annuity(bench, t)
        for t in HEDGE_TENORS
    ])
    J_ift = -np.linalg.solve(dG_dz, dG_dq)
    J_bump = np.column_stack([
        (np.array(build(quotes + BP * np.eye(n)[j]).zeros)
         - np.array(build(quotes - BP * np.eye(n)[j]).zeros)) / (2 * BP)
        for j in range(n)
    ])
    upper = np.abs(dG_dz[np.triu_indices(n, 1)]).max()
    print("\nBuild Jacobian d(knot)/d(par quote)\n")
    print(f"  dG/dz above the diagonal (should be 0)        : {upper:.1e}")
    print(f"  IFT vs bump-and-rebuild, worst entry          : {np.abs(J_ift - J_bump).max():.1e}")
    print(f"  10y knot row: 10y {J_ift[k10, k10]:.3f}, 7y {J_ift[k10, k10 - 1]:+.3f}, "
          f"5y {J_ift[k10, k10 - 2]:+.3f}")
    print(f"  30y knot row: 30y {J_ift[k30, k30]:.3f}, 10y {J_ift[k30, k10]:+.3f}")

    # --- chain rule: bucket risk -> quote risk, checked against brute force ---
    chain = buckets @ J_ift
    brute = np.array([
        (pool_value(build(quotes - BP * np.eye(n)[j]))
         - pool_value(build(quotes + BP * np.eye(n)[j]))) / 2.0
        for j in range(n)
    ])
    print("\n  Pool DV01 by par quote ($ per bp of quote), chain rule vs rebuild\n")
    print(f"{'quote':>9} {'chain':>9} {'rebuild':>9}")
    for j, tenor in enumerate(HEDGE_TENORS):
        if abs(chain[j]) > 0.05:
            print(f"{tenor:8.2f}y {chain[j]:9.2f} {brute[j]:9.2f}")
    print(f"  worst gap {np.abs(chain - brute).max():.1e}")

    # --- 4. the hedge is read off, not solved ---------------------------------
    # A par bond struck at today's quote keeps pricing at par under any move of
    # the *other* quotes, because the rebuilt curve still reprices it. So its
    # sensitivity matrix in quote space is diagonal, with -annuity on the diagonal.
    def bond_price(curve: Curve, tenor: float, coupon: float) -> float:
        if tenor <= 1.0 / FREQUENCY + 1e-12:
            return curve.discount_factor(tenor) * (1.0 + coupon * tenor)
        return coupon * annuity(curve, tenor) + curve.discount_factor(tenor)

    bond_sens = np.array([
        [bond_price(build(quotes + BP * np.eye(n)[j]), t, q) - 1.0
         for j in range(n)]
        for t, q in zip(HEDGE_TENORS, quotes)
    ]) / BP
    off = np.abs(bond_sens - np.diag(np.diag(bond_sens))).max()
    print("\nHedge with par bonds\n")
    print(f"  bond price sensitivity to other bonds' quotes : {off:.1e}")
    bond_dv01 = -np.diag(bond_sens) * BP  # price change per bp, per 1 of face
    hedge_face = chain / bond_dv01  # face to sell of each bond
    print(f"\n{'bond':>9} {'pool DV01':>10} {'DV01/100':>9} {'face':>10}")
    for j, tenor in enumerate(HEDGE_TENORS):
        if abs(chain[j]) > 0.05:
            side = "sell" if hedge_face[j] > 0 else "buy"
            print(f"{tenor:8.2f}y {chain[j]:10.2f} {100 * bond_dv01[j]:9.4f} "
                  f"{abs(hedge_face[j]):10,.0f} {side}")

    naive = np.zeros(n)
    naive[k10], naive[k30] = buckets[k10] / bond_dv01[k10], buckets[k30] / bond_dv01[k30]

    rng = np.random.default_rng(7)
    pnl = {"unhedged": [], "90/10 on bonds": [], "Jacobian hedge": []}
    for _ in range(400):
        shock = rng.normal(0.0, 5.0, n) * BP
        moved = build(quotes + shock)
        pool = pool_value(moved) - v0
        bonds = np.array([bond_price(moved, t, q) - 1.0 for t, q in zip(HEDGE_TENORS, quotes)])
        pnl["unhedged"].append(pool)
        pnl["90/10 on bonds"].append(pool - naive @ bonds)
        pnl["Jacobian hedge"].append(pool - hedge_face @ bonds)
    print("\n  P&L under 400 random quote shocks (5 bp each, independent)")
    for name, values in pnl.items():
        print(f"  {name:16} rms ${np.sqrt(np.mean(np.square(values))):8.2f}")


if __name__ == "__main__":
    main()
