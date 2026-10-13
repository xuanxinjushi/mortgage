"""Asset allocation across sectors: how much of a bond portfolio should be mortgages?

A core bond manager is measured against an aggregate index that holds
Treasuries, agency MBS and investment-grade corporates. The decision above the
coupon stack is how much of each sector to hold, and the same tools answer it:
expected excess return over duration-matched Treasuries, a factor model for
the risk, and a tracking-error budget.

The MBS sector is the chapter's index portfolio, with its OAS, effective and
spread duration, and prepayment exposure taken from coupon_allocation.py. The
Treasury and corporate figures, the sector weights, and the spread
volatilities and their correlation are illustrative. The corporate excess
return is its spread net of an expected default loss.

    python code/asset_allocation.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from coupon_allocation import BENCHMARK, IDIO_VOL, PREPAY_VOL, RATE_VOL, exposures

SECTORS = ["Treasury", "MBS", "Corporate"]
INDEX_WEIGHTS = np.array([0.45, 0.27, 0.28])        # illustrative aggregate-index mix
TE_BUDGET = 0.0050                                   # 50 bp a year: a core bond mandate
CORP_SPREAD, CORP_LOSS = 0.0095, 0.0015              # spread and expected default loss
CORP_DURATION, TSY_DURATION = 7.0, 6.0
SPREAD_VOL = {"MBS": 0.0025, "Corporate": 0.0035}    # annual spread volatility
CORRELATION = 0.6                                    # MBS and corporate spreads move together


def sector_inputs() -> dict[str, np.ndarray]:
    x = exposures()
    mbs_oas = BENCHMARK @ x["oas"]
    return {
        "excess": np.array([0.0, mbs_oas, CORP_SPREAD - CORP_LOSS]),
        "duration": np.array([TSY_DURATION, BENCHMARK @ x["duration"], CORP_DURATION]),
        "spread_duration": np.array([0.0, BENCHMARK @ x["spread_duration"], CORP_DURATION]),
        "prepay": np.array([0.0, BENCHMARK @ x["prepay"], 0.0]),
    }


def risk_parts(active: np.ndarray, s: dict[str, np.ndarray],
               correlation: float = CORRELATION) -> dict[str, float]:
    rates = (active @ s["duration"] * RATE_VOL) ** 2
    mbs = active[1] * s["spread_duration"][1] * SPREAD_VOL["MBS"]
    corp = active[2] * s["spread_duration"][2] * SPREAD_VOL["Corporate"]
    spread = mbs**2 + corp**2 + 2.0 * correlation * mbs * corp
    prepay = (active[1] * s["prepay"][1]) ** 2
    idio = float(np.sum((active * IDIO_VOL) ** 2))
    return {"rates": rates, "spreads": spread, "prepayment": prepay, "idiosyncratic": idio}


def best_weights(s: dict[str, np.ndarray], correlation: float = CORRELATION) -> np.ndarray:
    te = lambda w: np.sqrt(sum(risk_parts(w - INDEX_WEIGHTS, s, correlation).values()))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w: (w - INDEX_WEIGHTS) @ s["duration"]},
            {"type": "ineq", "fun": lambda w: TE_BUDGET - te(w)}]
    res = minimize(lambda w: -(w - INDEX_WEIGHTS) @ s["excess"], INDEX_WEIGHTS, method="SLSQP",
                   bounds=[(0.0, 0.7)] * 3, constraints=cons)
    return res.x


def main() -> None:
    s = sector_inputs()
    print(f"{'sector':>10} {'index':>6} {'excess bp':>10} {'eff dur':>8} {'spr dur':>8}")
    for name, w, e, d, sd in zip(SECTORS, INDEX_WEIGHTS, s["excess"], s["duration"], s["spread_duration"]):
        print(f"{name:>10} {w:6.0%} {e * 1e4:10.1f} {d:8.2f} {sd:8.2f}")

    for corr in (CORRELATION, 0.9):
        w = best_weights(s, corr)
        active = w - INDEX_WEIGHTS
        parts = risk_parts(active, s, corr)
        total = sum(parts.values())
        print(f"\nSpread correlation {corr:.1f}: " + ", ".join(f"{n} {wi:.1%} ({a:+.1%})"
                                                         for n, wi, a in zip(SECTORS, w, active)))
        print(f"  active excess return {active @ s['excess'] * 1e4:+.1f} bp a year, "
              f"tracking error {np.sqrt(total) * 1e4:.0f} bp, information ratio "
              f"{active @ s['excess'] / np.sqrt(total):.2f}")
        print("  tracking error from " + ", ".join(f"{k} {v / total:.0%}" for k, v in parts.items()))


if __name__ == "__main__":
    main()
