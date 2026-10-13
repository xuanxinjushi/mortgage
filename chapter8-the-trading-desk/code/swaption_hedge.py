"""swaption_hedge.py -- sizing the swaption hedge for the pool's negative convexity.

Takes the pool's effective convexity and the rate volatility from
duration_hedging.py and this chapter, and works out three numbers the desk
cares about: the annual convexity cost it bleeds, the notional and premium of
an at-the-money swaption straddle that would gamma-match that convexity, and
the straddle's running theta -- which, priced on the same volatility, comes
back as the convexity cost itself. That identity is why a full convexity hedge
is a wash and the desk buys back only part.

The notional and premium depend on the illustrative instrument chosen here --
a one-year-expiry straddle on a ten-year swap, priced in the normal (Bachelier)
model on a flat 4.5% curve. The theta-equals-cost identity does not depend on
that choice; it holds for any gamma-matched hedge priced on the hedged rate's
own volatility.
"""

from __future__ import annotations

import math

# --- the pool, from duration_hedging.py and this chapter ---
FACE = 100.0            # work per 100 face of pool
PRICE = 97.29           # pool price at the 6.5% base market rate
CONVEXITY = -288.0      # effective (option-adjusted) convexity
SIGMA = 0.0070          # annualized normal rate volatility, 70 bp

# --- illustrative hedge instrument: 1y-expiry straddle on a 10y swap ---
EXPIRY = 1.0            # option expiry, years
TENOR = 10             # underlying swap tenor, years
CURVE = 0.045          # flat continuously-compounded discount rate


def annuity(rate: float, tenor: int) -> float:
    """Underlying-swap annuity (PV01): discounted annual year-fractions."""
    return sum(math.exp(-rate * i) for i in range(1, tenor + 1))


def main() -> None:
    bp = 1.0e4

    # Pool: convexity cost is the second-order bleed, 1/2 * C * sigma^2,
    # as a fraction of price. Its dollar form is d2P/dy2 = C * P.
    cost_bp = 0.5 * abs(CONVEXITY) * SIGMA**2 * bp
    pool_gamma = abs(CONVEXITY) * PRICE                 # |d2P/dy2| per 100 face

    # ATM straddle in the normal model, per 1 unit of notional:
    #   premium fraction = A * sigma * sqrt(T) / sqrt(2 pi)   (x2 for the straddle)
    #   gamma d2V/dr2    = A / (sigma * sqrt(T) * sqrt(2 pi)) (x2 for the straddle)
    phi0 = 1.0 / math.sqrt(2.0 * math.pi)
    ann = annuity(CURVE, TENOR)
    premium_frac = 2.0 * ann * SIGMA * math.sqrt(EXPIRY) * phi0
    gamma_per_notional = 2.0 * ann * phi0 / (SIGMA * math.sqrt(EXPIRY))

    # Gamma-match: pick the notional whose straddle gamma offsets the pool's.
    notional = pool_gamma / gamma_per_notional          # per 100 face of pool
    premium_bp = notional * premium_frac / PRICE * bp    # of pool value
    theta_bp = 0.5 * pool_gamma * SIGMA**2 / PRICE * bp  # of pool value

    print(f"pool effective convexity          {CONVEXITY:8.0f}")
    print(f"rate volatility (normal)          {SIGMA * bp:6.0f} bp")
    print(f"convexity cost  1/2 C sigma^2     {cost_bp:6.1f} bp/yr")
    print("gamma-matched 1y-into-10y ATM straddle:")
    print(f"  notional                        {notional / FACE:6.2f} x pool face")
    print(f"  up-front premium (1y cover)     {premium_bp:6.1f} bp of pool value")
    print(f"  running theta                   {theta_bp:6.1f} bp/yr")
    print(f"premium / convexity cost          {premium_bp / cost_bp:6.2f} x")


if __name__ == "__main__":
    main()
