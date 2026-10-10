"""neural_pricing_surrogate.py -- Neural Pricing Surrogate for Fast MBS Monte Carlo Valuation in PyTorch.

Demonstrates:
1. Simulating Monte Carlo pricing labels for an MBS pool with prepayment options.
2. Training a PyTorch Neural Surrogate Emulator to learn the non-linear pricing map: (r0, WAC, vol, prepay_mult) -> Price.
3. Benchmarking inference latency: neural evaluation vs Monte Carlo simulation (thousands of times faster).
4. Evaluating pricing accuracy (R^2 about 0.997, MAE under 0.10 per $100 face value) on held-out market scenarios.
5. Showing what the surrogate still needs Monte Carlo for: its labels, a precise reference
   to measure it against, a DV01 check, and the inputs outside its training box.

The trained model is saved next to this file (`neural_pricing_surrogate.pt`), with
the scaling it was trained on. Later runs load it instead of rebuilding the
2,500-pool training set and training again; pass --retrain to start from scratch.

    python code/neural_pricing_surrogate.py
    python code/neural_pricing_surrogate.py --retrain
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Cap CPU threads for deterministic performance
torch.set_num_threads(4)

# Parameters
N_TRAIN_SAMPLES = 2500
N_TEST_SAMPLES = 500
SEED = 42


MORTGAGE_SPREAD = 0.0177   # PMMS over the 10-year Treasury, 1971-2026 (see oas_engine.py)
TERM_PREMIUM = 0.008       # long-run short rate above today's: the engine's 4.5% theta vs 3.66% SOFR

CHECKPOINT = Path(__file__).with_name("neural_pricing_surrogate.pt")
# Everything the saved model depends on. A checkpoint trained under a different
# setup is ignored rather than silently reused.
TRAINING_SETUP = {"n_train": N_TRAIN_SAMPLES, "train_seed": 42, "epochs": 300, "hidden_dim": 64,
                  "mortgage_spread": MORTGAGE_SPREAD, "term_premium": TERM_PREMIUM}


def quoted_rate(r, a, theta, vol, tenor=10.0, spread=MORTGAGE_SPREAD):
    """Mortgage rate quoted when the short rate is r: Vasicek zero rate + spread."""
    B = (1.0 - math.exp(-a * tenor)) / a
    log_A = (B - tenor) * (a * a * theta - vol ** 2 / 2.0) / (a * a) - vol ** 2 * B * B / (4.0 * a)
    return (-log_A + B * r) / tenor + spread


def monte_carlo_pool_price(
    r0: float,
    wac: float,
    vol: float,
    prepay_mult: float,
    n_paths: int = 250,
    n_steps: int = 60,
    dt: float = 1.0 / 12.0,
    seed: int | None = None,
) -> float:
    """Vectorized Monte Carlo pricer for an MBS passthrough pool with prepayment S-curve.
    
    Price = E[sum_{t=1}^T CF_t * exp(-int_0^t r_s ds)]
    """
    rng = np.random.default_rng(seed)
    
    # 1. Simulate rate paths under mean-reverting Vasicek. r0 sets the level of
    # the whole curve: the short rate starts there and reverts toward r0 plus a
    # fixed term premium, so moving r0 moves the quoted mortgage rate one for one
    # -- a parallel shift, which is what a desk means by "rates moved".
    a_rev, theta_rev = 0.40, r0 + TERM_PREMIUM
    r_paths = np.zeros((n_paths, n_steps + 1))
    r_paths[:, 0] = r0
    for t in range(n_steps):
        r_curr = r_paths[:, t]
        drift = a_rev * (theta_rev - r_curr)
        dw = rng.normal(0.0, math.sqrt(dt), size=n_paths)
        r_paths[:, t + 1] = r_curr + drift * dt + vol * dw

    # 2. Monthly cash flow generation along each path
    # Pool parameters: 5-year remaining term, monthly coupon = WAC / 12
    m_rate = wac / 12.0
    balance = np.ones(n_paths)  # Normalized to $1 initial balance
    pv_paths = np.zeros(n_paths)
    cum_discount = np.ones(n_paths)

    for t in range(1, n_steps + 1):
        # Discount factor increment
        r_step = r_paths[:, t - 1]
        cum_discount *= np.exp(-r_step * dt)

        # Scheduled payment
        rem_steps = n_steps - t + 1
        if rem_steps > 1:
            pmt = balance * (m_rate * (1.0 + m_rate) ** rem_steps) / (((1.0 + m_rate) ** rem_steps) - 1.0)
        else:
            pmt = balance * (1.0 + m_rate)

        interest_pmt = balance * m_rate
        sched_prin = pmt - interest_pmt

        # Prepayment S-curve on the refinance incentive: WAC against the
        # mortgage rate the borrower is quoted -- the Vasicek 10-year zero
        # implied by today's short rate plus the 177 bp mortgage spread, the
        # same chain oas_engine.py uses -- never the short rate itself.
        incentive_bp = (wac - quoted_rate(r_paths[:, t], a_rev, theta_rev, vol)) * 10000.0  # bp
        # Monthly SMM via logistic curve
        base_smm = 0.02 / (1.0 + np.exp(-(incentive_bp - 75.0) / 40.0))
        smm = np.clip(base_smm * prepay_mult, 0.001, 0.15)

        prepay_prin = (balance - sched_prin) * smm
        total_cf = interest_pmt + sched_prin + prepay_prin

        pv_paths += total_cf * cum_discount
        balance = np.maximum(balance - sched_prin - prepay_prin, 0.0)

    # Scale to $100 face value
    price = float(np.mean(pv_paths) * 100.0)
    return price


def generate_market_dataset(n_samples: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Generate diverse macro market states and evaluate ground-truth Monte Carlo prices."""
    rng = np.random.default_rng(seed)
    
    # Inputs:
    # 0: Rate level r0, today's short rate (2.5% to 7.0%); the curve moves with it
    # 1: Pool WAC (5.0% to 8.0%), spanning the money against a quoted rate near 6.3%
    # 2: Rate Volatility (0.8% to 2.0%)
    # 3: Prepayment Speed Multiplier (0.6 to 1.8)
    r0 = rng.uniform(0.025, 0.070, size=n_samples)
    wac = rng.uniform(0.050, 0.080, size=n_samples)
    vol = rng.uniform(0.008, 0.020, size=n_samples)
    prepay_mult = rng.uniform(0.6, 1.8, size=n_samples)

    X = np.column_stack([r0, wac, vol, prepay_mult])
    prices = np.zeros(n_samples)

    for i in range(n_samples):
        prices[i] = monte_carlo_pool_price(
            r0=X[i, 0],
            wac=X[i, 1],
            vol=X[i, 2],
            prepay_mult=X[i, 3],
            n_paths=200,
            seed=seed + i,
        )

    return X, prices


class PricingSurrogateNet(nn.Module):
    """Deep Neural Surrogate Emulator for instantaneous MBS pricing."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_pricing_surrogate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 300,
    lr: float = 0.005,
    seed: int = SEED,
) -> tuple[PricingSurrogateNet, tuple[np.ndarray, np.ndarray, float, float]]:
    """Train the deep neural surrogate pricing network in PyTorch."""
    torch.manual_seed(seed)

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-7
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) + 1e-7

    X_train_norm = (X_train - x_mean) / x_std
    y_train_norm = (y_train - y_mean) / y_std

    model = PricingSurrogateNet(in_features=4, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_norm, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tr_t)
        loss = criterion(preds, y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    scaler = (x_mean, x_std, y_mean, y_std)
    return model, scaler


def save_surrogate(model: PricingSurrogateNet, scaler, path: Path = CHECKPOINT) -> None:
    """Save the weights together with the scaling they were trained on.

    A model reloaded without its scaling still runs, but on inputs scaled
    differently from training: it returns wrong prices and raises no error.
    """
    x_mean, x_std, y_mean, y_std = scaler
    torch.save({"setup": TRAINING_SETUP, "state_dict": model.state_dict(),
                "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                "y_mean": y_mean, "y_std": y_std}, path)


def load_surrogate(path: Path = CHECKPOINT):
    """The saved model and its scaling, or None if there is none for this setup."""
    if not path.exists():
        return None
    saved = torch.load(path, weights_only=True)
    if saved.get("setup") != TRAINING_SETUP:
        return None
    model = PricingSurrogateNet(in_features=4, hidden_dim=TRAINING_SETUP["hidden_dim"])
    model.load_state_dict(saved["state_dict"])
    scaler = (np.array(saved["x_mean"]), np.array(saved["x_std"]), saved["y_mean"], saved["y_std"])
    return model, scaler


def benchmark_latency(
    model: PricingSurrogateNet, scaler: tuple[np.ndarray, np.ndarray, float, float], n_evals: int = 100
) -> tuple[float, float]:
    """Measure inference latency per pool: Monte Carlo vs Neural Surrogate."""
    x_mean, x_std, y_mean, y_std = scaler
    sample_input = np.array([0.045, 0.065, 0.012, 1.0])

    # 1. Monte Carlo runtime
    t0 = time.perf_counter()
    for _ in range(n_evals):
        _ = monte_carlo_pool_price(0.045, 0.065, 0.012, 1.0, n_paths=200)
    t_mc = (time.perf_counter() - t0) / n_evals * 1000.0  # ms per pool

    # 2. Neural Surrogate runtime (vectorized batch of 1,000 evaluations)
    batch_input = np.tile(sample_input, (1000, 1))
    batch_norm = torch.tensor((batch_input - x_mean) / x_std, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # Warmup
        _ = model(batch_norm)
        t0 = time.perf_counter()
        _ = model(batch_norm)
        t_nn = (time.perf_counter() - t0) / 1000.0 * 1000.0  # ms per pool

    return t_mc, t_nn


def plot_results(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    rates_mc: np.ndarray,
    prices_mc: np.ndarray,
    rates_nn: np.ndarray,
    prices_nn: np.ndarray,
    t_mc: float,
    t_nn: float,
    output_path: str = "neural_pricing_surrogate.png",
    lang: str = "en",
) -> None:
    """Generate comparative visualization of pricing accuracy, latency, and convexity."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel 1: Parity Plot (Monte Carlo vs Neural Surrogate)
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, color="#1b4965", alpha=0.5, s=18, edgecolors="none")
    lo, hi = min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5
    ax1.plot([lo, hi], [lo, hi], "r--", lw=1.8, label="Ideal 45° Parity" if lang == "en" else "理论完美吻合线 (45°)")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Monte Carlo Ground Truth Price ($)" if lang == "en" else "蒙特卡洛高精真实价格 ($)", fontsize=11)
    ax1.set_ylabel("Neural Surrogate Predicted Price ($)" if lang == "en" else "神经网络代理预测价格 ($)", fontsize=11)
    r2 = 1.0 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    ax1.set_title(f"Pricing Accuracy (R² = {r2:.4f}, MAE = ${mae:.3f})" if lang == "en" else f"定价准确度 (R² = {r2:.4f}, MAE = ${mae:.3f})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Panel 2: Latency Speedup
    ax2 = axes[1]
    speedup = t_mc / t_nn if t_nn > 0 else 1.0
    bars = ax2.bar(["Monte Carlo Engine", "Neural Surrogate"], [t_mc, t_nn], color=["#e76f51", "#2a9d8f"], width=0.5)
    ax2.set_ylabel("Inference Time per Pool (ms)" if lang == "en" else "单池定价延迟 (毫秒/池)", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_title(f"Valuation Latency: {speedup:,.0f}x Speedup" if lang == "en" else f"计算延迟基准：加速 {speedup:,.0f} 倍", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, ls="--", which="both")
    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h * 1.3, f"{h:.4f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 3: Negative Convexity Curve Captured by Surrogate
    ax3 = axes[2]
    ax3.plot(rates_mc * 100, prices_mc, "o", color="#e76f51", markersize=6, label="Monte Carlo Evaluations" if lang == "en" else "蒙特卡洛计算点")
    ax3.plot(rates_nn * 100, prices_nn, color="#1b4965", lw=2.2, label="Neural Surrogate Curve" if lang == "en" else "神经代理平滑定价曲线")
    ax3.set_xlabel("Market Rate $r$ (%)" if lang == "en" else "市场利率 $r$ (%)", fontsize=11)
    ax3.set_ylabel("MBS Price (per $100 face)" if lang == "en" else "MBS 价格 (面值 100)", fontsize=11)
    ax3.set_title("MBS Negative Convexity Reproduction" if lang == "en" else "房贷负凸性特征精确还原", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, ls="--")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {output_path}")


def where_monte_carlo_is_still_needed(
    model: PricingSurrogateNet,
    scaler: tuple[np.ndarray, np.ndarray, float, float],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """The surrogate replaces Monte Carlo in the intraday loop, not in the firm.

    Every check below needs a Monte Carlo run the surrogate cannot supply itself:
    the noise in its own labels, a precise reference to score it against, a
    finite-difference DV01 to check autograd with, and the prices of inputs that
    lie outside the box it was trained on.
    """
    x_mean, x_std, y_mean, y_std = scaler
    model.eval()

    def surrogate(x) -> float:
        with torch.no_grad():
            z = torch.tensor((np.atleast_2d(x) - x_mean) / x_std, dtype=torch.float32)
            return float(model(z)[0]) * y_std + y_mean

    base = (0.045, 0.065, 0.012, 1.0)

    # a. How noisy is one 200-path label? Reprice the same pool on 200 seeds.
    label_sd = np.std([monte_carlo_pool_price(*base, n_paths=200, seed=s) for s in range(200)])

    # b. Score labels and surrogate against a 20,000-path reference.
    n_ref = 100
    ref = np.array([monte_carlo_pool_price(*x, n_paths=20_000, seed=7) for x in X_test[:n_ref]])
    label_err = float(np.mean(np.abs(y_test[:n_ref] - ref)))
    nn_err = float(np.mean(np.abs(np.array([surrogate(x) for x in X_test[:n_ref]]) - ref)))

    print("\n5. What the surrogate still needs Monte Carlo for:")
    print(f"   One 200-path label, std across seeds:      ${label_sd:.3f}")
    print(f"   200-path labels vs 20,000-path reference:  ${label_err:.3f} MAE")
    print(f"   Surrogate vs 20,000-path reference:        ${nn_err:.3f} MAE")

    # c. DV01: surrogate vs a two-sided Monte Carlo bump on common random numbers.
    h = 1e-4
    rel = []
    for r in (0.03, 0.04, 0.05, 0.06):
        for w in (0.055, 0.065, 0.075):
            mc = (monte_carlo_pool_price(r - h, w, 0.012, 1.0, n_paths=20_000, seed=3)
                  - monte_carlo_pool_price(r + h, w, 0.012, 1.0, n_paths=20_000, seed=3)) / 2.0
            nn_dv01 = (surrogate((r - h, w, 0.012, 1.0)) - surrogate((r + h, w, 0.012, 1.0))) / 2.0
            rel.append(abs(nn_dv01 - mc) / abs(mc))
    print(f"   DV01 vs Monte Carlo bump, 12 pools:        median {np.median(rel):.1%}, worst {max(rel):.1%}")

    # d. Inputs outside the training box. The surrogate still answers.
    print("\n   Outside the training box (Monte Carlo on 50,000 paths):")
    print(f"   {'input':34} {'MC':>8} {'surrogate':>10} {'error':>7}")
    for label, x in [
        ("short rate 1.0%  (trained 2.5-7%)", (0.010, 0.065, 0.012, 1.0)),
        ("short rate 9.0%  (trained 2.5-7%)", (0.090, 0.065, 0.012, 1.0)),
        ("volatility 3.5%  (trained 0.8-2%)", (0.045, 0.065, 0.035, 1.0)),
        ("prepay mult 3.0  (trained 0.6-1.8)", (0.045, 0.065, 0.012, 3.0)),
    ]:
        mc = monte_carlo_pool_price(*x, n_paths=50_000, seed=5)
        nn_price = surrogate(x)
        print(f"   {label:34} {mc:8.3f} {nn_price:10.3f} {abs(nn_price - mc):7.3f}")


def main():
    print("=" * 72)
    print("Chapter 6: Neural Pricing Surrogate for Fast MBS Monte Carlo Valuation")
    print("=" * 72)

    # 1-2. Load the saved surrogate if there is one; otherwise build the training
    # set (the slow part: 2,500 Monte Carlo prices), train, and save.
    loaded = None if "--retrain" in sys.argv[1:] else load_surrogate()
    t0 = time.time()
    X_test, y_test = generate_market_dataset(N_TEST_SAMPLES, seed=999)
    if loaded is None:
        print("1. Generating offline training data using high-fidelity Monte Carlo...")
        X_train, y_train = generate_market_dataset(N_TRAIN_SAMPLES, seed=TRAINING_SETUP["train_seed"])
        print(f"   Generated {len(X_train)} train and {len(X_test)} test pools in {time.time() - t0:.1f}s.")
        print(f"   Average Pool Price: ${y_train.mean():.2f} (Std: ${y_train.std():.2f})")

        print("\n2. Training Deep Neural Surrogate Network in PyTorch...")
        model, scaler = fit_pricing_surrogate(X_train, y_train, epochs=TRAINING_SETUP["epochs"])
        save_surrogate(model, scaler)
        print(f"   Saved the model and its scaling to {CHECKPOINT.name}")
    else:
        model, scaler = loaded
        print(f"1-2. Loaded the trained surrogate from {CHECKPOINT.name} (--retrain to train again).")
        print(f"   Generated {len(X_test)} test pools in {time.time() - t0:.1f}s.")

    # 3. Evaluate out-of-sample accuracy
    x_mean, x_std, y_mean, y_std = scaler
    X_test_norm = torch.tensor((X_test - x_mean) / x_std, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test_norm).numpy()
    y_pred = y_pred_norm * y_std + y_mean

    r2 = 1.0 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    print("\n3. Out-of-Sample Accuracy Evaluation:")
    print(f"   R-squared (R²):               {r2:.5f}")
    print(f"   Mean Absolute Error (MAE):     ${mae:.4f} per $100 face")
    print(f"   Root Mean Square Error (RMSE): ${rmse:.4f} per $100 face")

    # 4. Latency benchmark
    print("\n4. Latency Benchmark:")
    t_mc, t_nn = benchmark_latency(model, scaler)
    speedup = t_mc / t_nn if t_nn > 0 else 1.0
    print(f"   Monte Carlo Engine:  {t_mc:8.3f} ms / pool")
    print(f"   Neural Surrogate:    {t_nn:8.5f} ms / pool")
    print(f"   Execution Speedup:   {speedup:8.0f}x Faster!")

    # 5. What Monte Carlo is still needed for
    where_monte_carlo_is_still_needed(model, scaler, X_test, y_test)

    # 6. Negative convexity curve reconstruction
    rate_grid = np.linspace(0.025, 0.075, 15)
    wac_fixed, vol_fixed, mult_fixed = 0.065, 0.012, 1.0
    mc_curve = [monte_carlo_pool_price(r, wac_fixed, vol_fixed, mult_fixed, n_paths=400, seed=123) for r in rate_grid]

    fine_rates = np.linspace(0.025, 0.075, 60)
    fine_X = np.column_stack([fine_rates, np.full_like(fine_rates, wac_fixed), np.full_like(fine_rates, vol_fixed), np.full_like(fine_rates, mult_fixed)])
    with torch.no_grad():
        nn_curve = model(torch.tensor((fine_X - x_mean) / x_std, dtype=torch.float32)).numpy() * y_std + y_mean

    # 7. Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, "neural_pricing_surrogate.png")
    plot_results(y_test, y_pred, rate_grid, np.array(mc_curve), fine_rates, nn_curve, t_mc, t_nn, output_png)
    print("\nCompleted successfully!")


if __name__ == "__main__":
    main()
