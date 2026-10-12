"""Default: how often loans go bad, how it happens, and why bad loans arrive together.

Every number in Chapter 2's "Default: The Other Exit" section comes from this
script. The loan-level counts are read from the vintage summaries pinned in
data/ (see scripts/build_default_vintage.py), which reduce Fannie Mae's 2000Q1,
2007Q1 and 2007Q3 acquisition quarters to totals.
"""

from statistics import NormalDist

from mortgagekit import credit, tape

STATES = ["current", "30", "60", "90+"]
FICO = ["<620", "620-679", "680-739", "740+"]
LTV = ["<=70", "71-80", "81-90", ">90"]


def expected_loss_example() -> None:
    print("Jack's expected loss: PD x LGD x EAD")
    ead, lgd = 400_000, 0.30
    for pd in (0.005, 0.02, 0.10):
        print(f"  PD {pd:5.1%}  LGD {lgd:.0%}  EAD ${ead:,}  ->  EL ${credit.expected_loss(pd, lgd, ead):>7,.0f}")


def roll_rate_table() -> None:
    print("\nMonthly roll rates (share of loans in a state that are in each state next month)")
    print(f"  {'from':<9}{'to':<13}{'2000Q1':>8}{'2007Q1':>8}")
    for a, b in [("current", "30"), ("current", "prepay"), ("30", "current"), ("30", "60"),
                 ("60", "current"), ("60", "90+"), ("90+", "current"), ("90+", "credit exit")]:
        old, new = tape.roll_rates("2000Q1")[a].get(b, 0.0), tape.roll_rates("2007Q1")[a].get(b, 0.0)
        print(f"  {a:<9}{b:<13}{old:8.1%}{new:8.1%}")


def risk_grid() -> None:
    grid = tape.default_by_risk("2007Q1")
    print("\n2007Q1 loans 90+ days late within five years, by origination FICO and LTV")
    print(f"  {'FICO':<9}" + "".join(f"{'LTV ' + c:>11}" for c in LTV))
    for f in FICO:
        cells = []
        for c in LTV:
            loans, bad = grid[(f, c)]
            cells.append(f"{bad / loans:11.1%}")
        print(f"  {f:<9}" + "".join(cells))
    loans = sum(n for n, _ in grid.values())
    bad = sum(b for _, b in grid.values())
    print(f"  all {loans:,} loans: {bad / loans:.1%}")


def vintage_table() -> dict:
    curves = tape.default_vintage()
    print("\nShare of each vintage ever 90+ days late, by loan age")
    print(f"  {'vintage':<9}{'loans':>9}" + "".join(f"{f'{m // 12}y':>8}" for m in (24, 60, 120, 180)))
    for name, v in curves.items():
        print(f"  {name:<9}{v['loans']:>9,}" + "".join(f"{v['d90'][m]:8.1%}" for m in (24, 60, 120, 180)))
    print("\nShare ending in foreclosure or short sale vs. prepaid, by 15 years")
    for name, v in curves.items():
        print(f"  {name:<9}credit exit {v['credit'][180]:6.1%}   prepaid {v['prepay'][180]:6.1%}")
    return curves


def one_factor(curves: dict) -> None:
    # The 2000 vintage's five-year rate stands in for an ordinary PD.
    pd, seen = curves["2000Q1"]["d90"][60], curves["2007Q1"]["d90"][60]
    print(f"\nOne-factor model: five-year pool default rate when the average is {pd:.1%}")
    print(f"  {'rho':<6}{'median year':>13}{'1-in-100':>10}{'1-in-1000':>11}")
    for rho in (0.0, 0.05, credit.BASEL_MORTGAGE_RHO, 0.30):
        cells = [credit.default_rate_quantile(pd, rho, q) for q in (0.5, 0.99, 0.999)]
        print(f"  {rho:<6.2f}" + f"{cells[0]:13.1%}{cells[1]:10.1%}{cells[2]:11.1%}")

    # How bad a year was 2007, read through the model? Ask what draw of the
    # common factor turns the ordinary rate into the 2007 vintage's.
    rho = credit.BASEL_MORTGAGE_RHO
    lo, hi = -6.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if credit.conditional_pd(pd, rho, mid) > seen else (lo, mid)
    odds = 1 / NormalDist().cdf(lo)
    print(f"\n  2007Q1 five-year 90+ rate {seen:.1%}")
    print(f"  at rho = {rho:.2f} that is a common-factor draw of z = {lo:.2f}, about 1 year in {odds:.0f}")
    print(f"  at rho = 0.01 the same rate needs z = {_solve(pd, 0.01, seen):.2f}")


def _solve(pd: float, rho: float, target: float) -> float:
    lo, hi = -40.0, 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if credit.conditional_pd(pd, rho, mid) > target else (lo, mid)
    return lo


if __name__ == "__main__":
    expected_loss_example()
    roll_rate_table()
    risk_grid()
    one_factor(vintage_table())
