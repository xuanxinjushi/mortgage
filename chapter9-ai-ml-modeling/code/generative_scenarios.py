"""A diffusion model that generates joint rate/spread scenarios -- and what it cannot do.

A mortgage scenario needs the rate a borrower is quoted, not just the Treasury
yield: the 30-year mortgage rate is the 10-year yield plus a spread, and that
spread does not move one-for-one with the yield. This file learns the joint
monthly move of the two from 1987 to 2026 and compares three generators:

    history      the 472 months actually observed
    Gaussian     a normal distribution with the same means and correlation --
                 the "correlation matrix" model a desk would start from
    diffusion    a small denoising diffusion model (DDPM) trained on the same
                 472 months

and then checks four things:

1. The asymmetry. In a sharp rally the spread widens much more than it narrows
   in a sharp sell-off, because lenders do not pass the full fall in rates on to
   borrowers when refinance volume swamps them. One correlation number cannot
   say that; the diffusion model should learn it.
2. Conditional generation. Fix the 10-year move and let the model fill in the
   spread -- the stress-testing use. Inside the range history covered, three
   independently trained models agree; outside it, they no longer do, and that
   disagreement is the only warning there is.
3. Memorisation. A model trained on 472 points can simply hand them back.
4. P versus Q. The generator learned history's drift. A pricing path has to
   match today's forward curve instead, and the two are not the same number.

    python code/generative_scenarios.py

Reads the pinned FRED snapshots in `data/`, so it runs offline. Needs a CUDA GPU.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
from torch import nn

from mortgagekit import bootstrap_par_curve
from mortgagekit.marketdata import mortgage_rate_30y, par_curve, treasury_10y

# This lab trains three diffusion models and samples hundreds of thousands of
# paths; it is written for a CUDA GPU and does not fall back to the CPU.
if not torch.cuda.is_available():
    sys.exit("generative_scenarios.py needs a CUDA GPU; none was found.")
DEVICE = torch.device("cuda")

N_STEPS = 200          # diffusion steps
TRAIN_ITERS = 4000
N_SAMPLES = 50_000
SEEDS = (0, 1, 2)      # three independently trained models, for the agreement check


# --- data ------------------------------------------------------------------

def monthly_moves():
    """Month-end 10-year yield and PMMS-minus-10-year spread; monthly changes in bp."""
    def month_end(series):
        out = {}
        for date, value in series:
            out[date[:7]] = value      # series are ascending, so the last one wins
        return out

    t10, mort = month_end(treasury_10y()), month_end(mortgage_rate_30y())
    months = sorted(m for m in set(t10) & set(mort) if m >= "1987-01")
    rate = np.array([t10[m] for m in months])
    spread = np.array([mort[m] - t10[m] for m in months])
    moves = np.column_stack([np.diff(rate), np.diff(spread)]) * 1e4
    return months[1:], moves


# --- a denoising diffusion model, small enough to read ----------------------

class Denoiser(nn.Module):
    """Predicts the noise that was added to a 2-vector at diffusion step t."""

    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 16, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, x, t):
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), 8, device=x.device))
        angle = t.float()[:, None] / N_STEPS * freqs[None, :]
        return self.net(torch.cat([x, torch.sin(angle), torch.cos(angle)], dim=1))


BETAS = torch.linspace(1e-4, 0.02, N_STEPS, device=DEVICE)
ALPHAS = 1.0 - BETAS
ALPHA_BAR = torch.cumprod(ALPHAS, dim=0)


def train(data: torch.Tensor, seed: int) -> Denoiser:
    """Standard DDPM training: noise a sample to step t, learn to predict the noise."""
    torch.manual_seed(seed)
    model = Denoiser().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_ITERS)
    for _ in range(TRAIN_ITERS):
        x0 = data[torch.randint(len(data), (256,), device=DEVICE)]
        t = torch.randint(N_STEPS, (len(x0),), device=DEVICE)
        eps = torch.randn_like(x0)
        ab = ALPHA_BAR[t][:, None]
        loss = ((model(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def sample(model: Denoiser, n: int, seed: int, fix_rate: float | None = None) -> torch.Tensor:
    """Reverse diffusion from pure noise. With ``fix_rate``, condition on the rate move.

    Conditioning is done by replacement: at every step the rate coordinate is
    overwritten with the fixed value, noised to that step's level, so the model
    only ever generates the spread consistent with it.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(n, 2, generator=gen, device=DEVICE)
    for t in reversed(range(N_STEPS)):
        if fix_rate is not None:
            ab = ALPHA_BAR[t]
            x[:, 0] = ab.sqrt() * fix_rate + (1 - ab).sqrt() * torch.randn(n, generator=gen, device=DEVICE)
        eps = model(x, torch.full((n,), t, device=DEVICE))
        mean = (x - BETAS[t] / (1 - ALPHA_BAR[t]).sqrt() * eps) / ALPHAS[t].sqrt()
        x = mean + (BETAS[t].sqrt() * torch.randn(n, 2, generator=gen, device=DEVICE) if t > 0 else 0.0)
    if fix_rate is not None:
        x[:, 0] = fix_rate
    return x


# --- checks -----------------------------------------------------------------

def spread_after(moves: np.ndarray, lo: float, hi: float) -> float:
    """Mean spread move in the months whose 10-year move falls in [lo, hi)."""
    sel = (moves[:, 0] >= lo) & (moves[:, 0] < hi)
    return float(moves[sel, 1].mean()), int(sel.sum())


def main() -> None:
    months, hist = monthly_moves()
    mu, sd = hist.mean(0), hist.std(0)
    z = torch.tensor((hist - mu) / sd, dtype=torch.float32, device=DEVICE)

    rng = np.random.default_rng(0)
    gauss = rng.multivariate_normal(mu, np.cov(hist.T), N_SAMPLES)
    models = [train(z, s) for s in SEEDS]
    diff = sample(models[0], N_SAMPLES, seed=10).cpu().numpy() * sd + mu

    print(f"Monthly moves, {months[0]} to {months[-1]}: {len(hist)} months (bp)")
    print(f"  10-year yield  sd {sd[0]:5.1f}   mortgage spread  sd {sd[1]:5.1f}"
          f"   correlation {np.corrcoef(hist.T)[0, 1]:+.2f}")

    # 1. the asymmetry a correlation matrix cannot hold
    print("\n1. Spread move after a large 10-year move (bp)")
    print(f"{'':28} {'history':>9} {'Gaussian':>9} {'diffusion':>10}")
    for label, lo, hi in [("10-year falls 50+ bp", -1e9, -50.0), ("10-year rises 50+ bp", 50.0, 1e9)]:
        h, n = spread_after(hist, lo, hi)
        g, _ = spread_after(gauss, lo, hi)
        d, _ = spread_after(diff, lo, hi)
        print(f"  {label:26} {h:+9.1f} {g:+9.1f} {d:+10.1f}   ({n} months)")

    print("\n   Tails of the spread move (bp)")
    print(f"{'':28} {'history':>9} {'Gaussian':>9} {'diffusion':>10}")
    for q, name in ((1, "1st percentile"), (99, "99th percentile")):
        print(f"  {name:26} {np.percentile(hist[:, 1], q):+9.1f} "
              f"{np.percentile(gauss[:, 1], q):+9.1f} {np.percentile(diff[:, 1], q):+10.1f}")

    # 2. conditional generation, and where three models stop agreeing
    lo_hist, hi_hist = hist[:, 0].min(), hist[:, 0].max()
    print(f"\n2. Conditional scenarios: fix the 10-year move, generate the spread"
          f"\n   (history's 10-year moves span {lo_hist:+.0f} to {hi_hist:+.0f} bp)")
    print(f"{'10-year move':>14} {'mean spread move, three models':>34} {'disagreement':>13}")
    for move in (-75.0, -50.0, 50.0, 75.0, -200.0, 200.0):
        target = (move - mu[0]) / sd[0]
        means = [float(sample(m, 20_000, seed=20, fix_rate=target)[:, 1].mean()) * sd[1] + mu[1]
                 for m in models]
        flag = "" if lo_hist <= move <= hi_hist else "  outside history"
        print(f"{move:+13.0f}  " + "  ".join(f"{v:+8.1f}" for v in means)
              + f"     {max(means) - min(means):8.1f}{flag}")

    # 3. memorisation: are generated months just training months handed back?
    dz = (torch.tensor((diff[:5000] - mu) / sd, dtype=torch.float32, device=DEVICE)[:, None, :] - z[None]).norm(dim=2)
    gen_nn = dz.min(1).values.median().item()
    zz = (z[:, None, :] - z[None]).norm(dim=2) + torch.eye(len(z), device=DEVICE) * 1e9
    hist_nn = zz.min(1).values.median().item()
    print("\n3. Memorisation check (distance to the nearest training month, in sd units)")
    print(f"  generated month -> training : median {gen_nn:.3f}")
    print(f"  training month  -> another  : median {hist_nn:.3f}")

    # 4. P versus Q: history's drift against today's forward curve
    curve = bootstrap_par_curve(par_curve().as_pairs())
    fwd_10y_in_1y = (curve.zero(11.0) * 11.0 - curve.zero(1.0) * 1.0) / 10.0
    q_change = (fwd_10y_in_1y - curve.zero(10.0)) * 1e4
    p_change = 12 * diff[:, 0].mean()
    print("\n4. Expected change in the 10-year rate over the next 12 months (bp)")
    print(f"  generator (history's drift, real-world P)     : {p_change:+6.1f}")
    print(f"  today's curve (1y-forward 10y zero, pricing Q) : {q_change:+6.1f}")


if __name__ == "__main__":
    main()
