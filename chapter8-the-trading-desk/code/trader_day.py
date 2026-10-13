"""A day on the mortgage desk: one inventory, three events, and the P&L they leave.

Benny opens the day long a pool of Jack's 6.5% loans (6.0% pass-through) and
short ten-year Treasuries against it, duration-flat. Three things then happen
-- a payrolls surprise, a customer selling him more of the same pool, and an
afternoon selloff in which mortgages also cheapen against Treasuries -- and he
re-hedges after each. The pool is priced with the chapter's own pricer from
duration_hedging.py, and the Treasury hedge is measured the same way the drift
table measures it, so the opening hedge is the table's 57.90 per 100 of pool.

At the close the day's P&L is split into the pieces a desk reports: the
customer edge, the convexity cost of hedging something whose duration moves,
and the basis. Rebalancing trades are assumed to cost nothing, which flatters
the convexity line; the customer's 2/32 discount is illustrative.

    python code/trader_day.py
"""

from __future__ import annotations

import numpy as np

from duration_hedging import SPREAD, pool_cash_flows

BUMP = 0.0025        # the same 25 bp bump effective_risk uses
TICK = 1.0 / 32.0    # MBS prices are quoted in 32nds


def pool_price(rate: float, spread: float = SPREAD) -> float:
    """Price per 100 face: the chapter's cash flows, discounted at rate + spread."""
    times, flows = pool_cash_flows(rate)
    return float(np.sum(flows * np.exp(-(rate + spread) * times)))


def pool_dv01(rate: float, spread: float = SPREAD) -> float:
    """Effective DV01 per 100 face: reprice with the borrower re-deciding."""
    up, down = pool_price(rate + BUMP, spread), pool_price(rate - BUMP, spread)
    return (down - up) / (2.0 * BUMP) * 0.0001


def pool_duration(rate: float, spread: float = SPREAD) -> float:
    return pool_dv01(rate, spread) / (pool_price(rate, spread) * 0.0001)


def treasury_price(y: float, coupon: float, maturity: float = 10.0) -> float:
    """Price per 100 face of the ten-year hedge bond at yield y."""
    times = np.arange(0.5, maturity + 0.01, 0.5)
    flows = np.full_like(times, 100.0 * coupon / 2.0)
    flows[-1] += 100.0
    return float(np.sum(flows * np.exp(-y * times)))


def treasury_dv01(y: float, coupon: float) -> float:
    return (treasury_price(y - 0.0001, coupon) - treasury_price(y + 0.0001, coupon)) / 2.0


class Book:
    """Pool face and Treasury short, both in $ millions of face."""

    def __init__(self, rate: float, pool_face: float):
        self.rate, self.spread = rate, SPREAD
        self.coupon = rate                  # the hedge bond is struck at par at the open
        self.pool_face = pool_face
        self.short = 0.0
        self.rehedge()

    def value(self) -> float:
        """Mark to market in $ millions: long pool minus short Treasuries."""
        return (self.pool_face * pool_price(self.rate, self.spread)
                - self.short * treasury_price(self.rate, self.coupon)) / 100.0

    def net_dv01(self) -> float:
        """Dollar DV01 of the whole book, in $ thousands per bp."""
        pool = self.pool_face * pool_dv01(self.rate, self.spread)
        hedge = self.short * treasury_dv01(self.rate, self.coupon)
        return (pool - hedge) * 10.0        # $mm face x (price/100 per bp) -> $k

    def rehedge(self) -> float:
        """Resize the Treasury short to flatten DV01; return Treasuries sold."""
        target = (self.pool_face * pool_dv01(self.rate, self.spread)
                  / treasury_dv01(self.rate, self.coupon))
        sold, self.short = target - self.short, target
        return sold


def main() -> None:
    book = Book(rate=0.065, pool_face=100.0)
    opening_short = book.short
    pnl = {"customer edge": 0.0, "convexity (rates, hedged)": 0.0, "basis": 0.0}

    def status(clock: str, event: str, sold: float, move_pnl: float) -> None:
        print(f"{clock}  {event:<27} {book.rate:6.2%} {pool_price(book.rate, book.spread):7.2f}"
              f" {pool_duration(book.rate, book.spread):5.2f} {book.pool_face:6.0f}"
              f" {book.short:7.2f} {sold:+7.2f} {move_pnl * 1000:+8.0f}")

    print("time   event                        rate   price   dur   pool   short    sold   P&L $k")
    status("07:00", "open, hedged flat", 0.0, 0.0)

    # 08:30 -- payrolls beat; the whole curve sells off 12 bp, mortgages with it.
    before = book.value()
    book.rate += 0.0012
    move = book.value() - before
    pnl["convexity (rates, hedged)"] += move
    status("08:30", "payrolls: rates +12 bp", book.rehedge(), move)

    # 10:15 -- a money manager sells Benny 50mm of the same pool, 2/32 under mid.
    edge = 50.0 * 2 * TICK / 100.0
    book.pool_face += 50.0
    pnl["customer edge"] += edge
    status("10:15", "buys 50mm from customer", book.rehedge(), edge)

    # 13:00 -- originators hit the TBA bid: rates +8 bp, and mortgages cheapen
    # 3 bp of spread against Treasuries on top of it.
    before = book.value()
    book.rate += 0.0008
    rates_only = book.value() - before
    book.spread += 0.0003
    basis = book.value() - before - rates_only
    pnl["convexity (rates, hedged)"] += rates_only
    pnl["basis"] += basis
    status("13:00", "selloff +8 bp, spread +3 bp", book.rehedge(), rates_only + basis)

    print(f"\nTreasuries sold since the open: {book.short - opening_short:.2f}mm"
          f" on top of the opening {opening_short:.2f}mm; net DV01 {book.net_dv01():+.2f} $k/bp")
    print("\nP&L attribution, $ thousands")
    for name, value in pnl.items():
        print(f"  {name:<27} {value * 1000:+8.0f}")
    print(f"  {'total':<27} {sum(pnl.values()) * 1000:+8.0f}")


if __name__ == "__main__":
    main()
