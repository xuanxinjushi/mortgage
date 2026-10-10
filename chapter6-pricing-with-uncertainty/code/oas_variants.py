"""The derived OAS runs Chapter 6 quotes beyond the main valuation.

Run from this chapter's folder:

    python code/oas_variants.py

`oas_engine.py` prices the pool once, the way the chapter first builds it. The
chapter then asks three follow-up questions of the same engine, and the
answers are tables and sentences in the text. They are computed here so they
can be rerun and checked rather than copied by hand:

1. The full chain against the shortcut: refinancing off the primary mortgage
   rate a borrower is actually quoted (the engine's default), versus off the
   short rate r_t.
2. No prepayment at all: what Z - OAS measures when there is no option.
3. Negative rates: how much of the simulation goes below zero, and what
   flooring the paths at zero does to the OAS across volatility levels.

Every run uses the engine's own calibration, path count and seed, so the
full-chain row reproduces the main run exactly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "oas_engine", Path(__file__).resolve().parent / "oas_engine.py")
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

# The engine's own run (see oas_engine.py's __main__ block).
BALANCE, NOTE_RATE, COUPON, MONTHS, MARKET_PRICE = 100.0, 0.065, 0.060, 360, 100.0
R0, A, THETA, SIGMA_R = engine.R0, engine.A, engine.THETA, engine.SIGMA_R
N_PATHS, SEED = 2000, 7


def spreads(paths, market_rate=None):
    """(Z-spread, OAS) in basis points: Z off the zero-volatility path, OAS off `paths`."""
    baseline = engine.zero_volatility_path(R0, A, THETA, MONTHS)
    z = engine.solve_oas([baseline], NOTE_RATE, BALANCE, MONTHS, MARKET_PRICE,
                         market_rate=market_rate, pass_through_coupon=COUPON)
    oas = engine.solve_oas(paths, NOTE_RATE, BALANCE, MONTHS, MARKET_PRICE,
                           market_rate=market_rate, pass_through_coupon=COUPON)
    return z * 1e4, oas * 1e4


def bp(x: float) -> str:
    """Signed basis points the way the chapter's tables print them."""
    return f"{x:+.1f}" if round(x, 1) != 0 else "-0.0"


if __name__ == "__main__":
    paths = engine.simulate_short_rate_paths(R0, A, THETA, SIGMA_R, MONTHS, N_PATHS, seed=SEED)

    # 1. Where the borrower refinances: the primary rate, or the short rate.
    print(f"{'':<34}{'Z-spread':>10}{'OAS':>9}{'Z - OAS':>10}")
    for label, market_rate in (("full chain, off the primary rate", None),
                               ("shortcut, off r_t", engine.short_rate_shortcut)):
        z, oas = spreads(paths, market_rate)
        print(f"{label:<34}{z:9.1f} {oas:8.1f} {bp(z - oas):>9}")

    # 2. Switch prepayment off entirely: no option left in the pool.
    real_refi = engine.refi_smm
    engine.refi_smm = lambda note_rate, market_rate, **kw: 0.0
    try:
        z, oas = spreads(paths)
    finally:
        engine.refi_smm = real_refi
    print(f"\nno prepayment at all: Z - OAS = {bp(z - oas)} bp "
          f"(Z {z:.1f}, OAS {oas:.1f}) -- the bond's own convexity")

    # 3. The paths that go below zero, at the chapter's volatility.
    months = paths[:, 1:]
    deepest = months.min()
    quoted = engine.primary_mortgage_rate(deepest)
    print(f"\nat sigma_r = {SIGMA_R:.1%}: {(months < 0).mean():.1%} of path-months negative, "
          f"{(months < 0).any(axis=1).mean():.0%} of paths dip below zero, deepest {deepest:.1%}")
    print(f"at that deepest point Jack is quoted {quoted:.2%}: incentive "
          f"{(NOTE_RATE - quoted) * 1e4:.0f} bp (off r_t it would be {(NOTE_RATE - deepest) * 1e4:.0f} bp); "
          f"monthly speed {engine.refi_smm(NOTE_RATE, quoted):.1%} "
          f"against {engine.refi_smm(NOTE_RATE, engine.primary_mortgage_rate(R0)):.1%} today")

    # ... and what flooring them at zero does to the OAS.
    print(f"\n{'sigma_r':>8}{'negative':>10}{'unfloored':>11}{'floored':>9}{'gap':>8}")
    for sigma in (0.015, 0.025, 0.040, 0.060):
        p = engine.simulate_short_rate_paths(R0, A, THETA, sigma, MONTHS, N_PATHS, seed=SEED)
        quoted = lambda r, s=sigma: engine.primary_mortgage_rate(r, A, THETA, s)
        _, raw = spreads(p, quoted)
        _, floored = spreads(np.maximum(p, 0.0), quoted)
        print(f"{sigma:8.1%}{(p[:, 1:] < 0).mean():10.1%}{raw:11.1f}{floored:9.1f}{bp(floored - raw):>8}")
