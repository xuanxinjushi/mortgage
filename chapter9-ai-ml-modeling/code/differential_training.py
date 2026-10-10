"""Differential training: Chapter 6's pricing surrogate, retrained on its Greeks.

Chapter 6's neural surrogate prices pools to within the noise of its own labels
and still gets DV01 badly wrong on one of twelve test pools -- the pool sitting
on the bend of the prepayment S-curve, where a price is easy to fit and its
slope is not. This file keeps everything Chapter 6 used -- the pricer, the
training pools, the network, the optimizer, the twelve-pool DV01 check -- and
changes one thing: every 200-path price label now carries its pathwise
derivative with respect to the rate level, and the second network is trained
on both.

The derivative labels come from running the same Monte Carlo pricer in PyTorch
and asking autograd for d(price)/d(r0). That is the differential machine
learning of Huge and Savine (2020): adjoint differentiation hands back each
label's gradient for a small multiple of the pricing cost, with no bumping.

Run from this chapter's folder:

    python code/differential_training.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

_SOURCE = (Path(__file__).resolve().parents[2]
           / "chapter6-pricing-with-uncertainty" / "code" / "neural_pricing_surrogate.py")
_spec = importlib.util.spec_from_file_location("ch6_surrogate", _SOURCE)
ch6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch6)

LAMBDA = 1.0          # weight on the derivative term, after both are standardized
H = 1e-4              # the one-basis-point bump Chapter 6's DV01 check uses


def price_and_delta(r0, wac, vol, prepay_mult, n_paths=200, n_steps=60, seed=None):
    """Chapter 6's pricer, line for line, in PyTorch: (price, d price / d r0).

    The random draws come from the same numpy generator in the same order, so
    the price matches ch6.monte_carlo_pool_price to rounding; autograd then
    returns the derivative of exactly that number.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0
    draws = np.stack([rng.normal(0.0, math.sqrt(dt), size=n_paths) for _ in range(n_steps)])
    dw = torch.tensor(draws, dtype=torch.float64)

    r_start = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
    a, theta = 0.40, r_start + ch6.TERM_PREMIUM
    rates = [r_start.expand(n_paths)]
    for t in range(n_steps):
        r = rates[-1]
        rates.append(r + a * (theta - r) * dt + vol * dw[t])

    m_rate = wac / 12.0
    balance = torch.ones(n_paths, dtype=torch.float64)
    pv = torch.zeros(n_paths, dtype=torch.float64)
    discount = torch.ones(n_paths, dtype=torch.float64)
    for t in range(1, n_steps + 1):
        discount = discount * torch.exp(-rates[t - 1] * dt)
        rem = n_steps - t + 1
        if rem > 1:
            pmt = balance * (m_rate * (1.0 + m_rate) ** rem) / (((1.0 + m_rate) ** rem) - 1.0)
        else:
            pmt = balance * (1.0 + m_rate)
        interest = balance * m_rate
        scheduled = pmt - interest
        incentive_bp = (wac - ch6.quoted_rate(rates[t], a, theta, vol)) * 10000.0
        smm = torch.clamp(0.02 / (1.0 + torch.exp(-(incentive_bp - 75.0) / 40.0)) * prepay_mult, 0.001, 0.15)
        prepaid = (balance - scheduled) * smm
        pv = pv + (interest + scheduled + prepaid) * discount
        balance = torch.clamp(balance - scheduled - prepaid, min=0.0)

    price = pv.mean() * 100.0
    (delta,) = torch.autograd.grad(price, r_start)
    return float(price.detach()), float(delta)


def differential_labels(X, seed):
    """Price and pathwise delta for every pool, on the seeds Chapter 6 used."""
    out = np.array([price_and_delta(*x, n_paths=200, seed=seed + i) for i, x in enumerate(X)])
    return out[:, 0], out[:, 1]


def fit_differential(X, y, dy_dr, epochs=300, lr=0.005, seed=ch6.SEED):
    """Chapter 6's network and optimizer, with a second term on d(price)/d(r0)."""
    torch.manual_seed(seed)
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-7
    y_mean, y_std = float(y.mean()), float(y.std()) + 1e-7

    xn = torch.tensor((X - x_mean) / x_std, dtype=torch.float32, requires_grad=True)
    yn = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    # The same derivative in standardized units: d(yn)/d(xn_0) = dy/dr0 * std(r0) / std(y).
    gn = torch.tensor(dy_dr * x_std[0] / y_std, dtype=torch.float32)
    g_scale = float(gn.std())

    model = ch6.PricingSurrogateNet(in_features=4, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(xn)
        (grad,) = torch.autograd.grad(pred.sum(), xn, create_graph=True)
        loss = torch.mean((pred - yn) ** 2) + LAMBDA * torch.mean(((grad[:, 0] - gn) / g_scale) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model, (x_mean, x_std, y_mean, y_std)


def predictor(model, scaler):
    x_mean, x_std, y_mean, y_std = scaler
    model.eval()

    def f(x):
        with torch.no_grad():
            z = torch.tensor((np.atleast_2d(x) - x_mean) / x_std, dtype=torch.float32)
            return float(model(z)[0]) * y_std + y_mean
    return f


def dv01_errors(f):
    """Chapter 6's check: surrogate DV01 against a 20,000-path Monte Carlo bump."""
    rows = []
    for r in (0.03, 0.04, 0.05, 0.06):
        for w in (0.055, 0.065, 0.075):
            mc = (ch6.monte_carlo_pool_price(r - H, w, 0.012, 1.0, n_paths=20_000, seed=3)
                  - ch6.monte_carlo_pool_price(r + H, w, 0.012, 1.0, n_paths=20_000, seed=3)) / 2.0
            nn = (f((r - H, w, 0.012, 1.0)) - f((r + H, w, 0.012, 1.0))) / 2.0
            rows.append((r, w, mc, nn, abs(nn - mc) / abs(mc)))
    return rows


def main():
    X_train, y_train = ch6.generate_market_dataset(ch6.N_TRAIN_SAMPLES, seed=42)
    X_test, _ = ch6.generate_market_dataset(ch6.N_TEST_SAMPLES, seed=999)

    prices, deltas = differential_labels(X_train, seed=42)
    print(f"PyTorch pricer vs Chapter 6's labels, worst gap: ${np.max(np.abs(prices - y_train)):.1e}")
    print(f"pathwise delta per label, mean {np.mean(deltas):.1f} per unit of r0 "
          f"(DV01 {-np.mean(deltas) * H:.4f} per 100 face)")

    plain = predictor(*ch6.fit_pricing_surrogate(X_train, y_train, epochs=300))
    differential = predictor(*fit_differential(X_train, y_train, deltas))

    n_ref = 100
    ref = np.array([ch6.monte_carlo_pool_price(*x, n_paths=20_000, seed=7) for x in X_test[:n_ref]])
    print(f"\n{'':28}{'price MAE':>11}{'DV01 median':>13}{'DV01 worst':>12}")
    results = {}
    for label, f in (("prices only (Chapter 6)", plain), ("prices + pathwise deltas", differential)):
        mae = float(np.mean(np.abs(np.array([f(x) for x in X_test[:n_ref]]) - ref)))
        rows = dv01_errors(f)
        rel = [row[-1] for row in rows]
        results[label] = rows
        print(f"{label:28}{'$' + format(mae, '.3f'):>11}{np.median(rel):>13.1%}{max(rel):>12.1%}")

    worst = max(results["prices only (Chapter 6)"], key=lambda row: row[-1])
    r, w, mc, _, _ = worst
    after = next(row for row in results["prices + pathwise deltas"] if row[:2] == (r, w))
    print(f"\nChapter 6's worst pool (rate level {r:.0%}, WAC {w:.1%}): Monte Carlo DV01 {mc:.4f}, "
          f"prices only {worst[3]:.4f}, with deltas {after[3]:.4f}")


if __name__ == "__main__":
    main()
