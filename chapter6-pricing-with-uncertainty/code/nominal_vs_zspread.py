"""Nominal spread vs. Z-spread on an upward-sloping curve.

The nominal spread compares the pool's own yield with the benchmark curve at
one point, the pool's average life. The Z-spread adds one constant spread to
the whole curve and discounts each cash flow at the rate for the year it
arrives. On a sloped curve the two disagree, before any prepayment option is
involved. The chapter's figure is drawn from this file.

Run from this chapter's folder:

    python code/nominal_vs_zspread.py
"""

from mortgagekit.rates import present_value

YEARS = range(1, 11)
# Illustrative, mortgage-like: cash every year, heaviest in the middle years,
# and the remaining balance paid off in year 10.
CASHFLOWS = [10.4, 11.7, 13.0, 14.3, 15.6, 15.6, 14.3, 13.0, 11.7, 33.8]
PRICE = 100.0


def benchmark(t):
    """Illustrative upward-sloping benchmark: 3.0% at zero, +28 bp a year."""
    return 0.030 + 0.0028 * t


def pv(rate_at):
    """Present value when the cash flow in year t is discounted at rate_at(t)."""
    return sum(present_value(cf, rate_at(t), t) for cf, t in zip(CASHFLOWS, YEARS))


def solve(price_at, lo=-0.05, hi=0.20):
    """Bisection: the rate or spread x at which price_at(x) equals PRICE."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if price_at(mid) > PRICE else (lo, mid)
    return (lo + hi) / 2


AVG_LIFE = sum(t * cf for t, cf in zip(YEARS, CASHFLOWS)) / sum(CASHFLOWS)
OWN_YIELD = solve(lambda y: pv(lambda t: y))                      # one flat rate
NOMINAL_SPREAD = OWN_YIELD - benchmark(AVG_LIFE)                  # one point
Z_SPREAD = solve(lambda s: pv(lambda t: benchmark(t) + s))       # every point

if __name__ == "__main__":
    print(f"Average life:            {AVG_LIFE:.1f} years")
    print(f"Pool's own yield:        {OWN_YIELD:.2%}")
    print(f"Benchmark at {AVG_LIFE:.1f} years:  {benchmark(AVG_LIFE):.2%}")
    print(f"Nominal spread:          {NOMINAL_SPREAD * 1e4:.0f} bp")
    print(f"Z-spread:                {Z_SPREAD * 1e4:.0f} bp")
    print(f"Difference:              {(NOMINAL_SPREAD - Z_SPREAD) * 1e4:.0f} bp")
