"""Vectorized Monte Carlo pool pricing, for the GPU section of Chapter 9.

Chapter 6 prices one rate path at a time: a Python loop over months, inside a
Python loop over paths. That is the clearest way to read the recursion and the
slowest way to run it.

The month loop cannot be removed -- the outstanding balance at month k depends
on month k-1, so time is genuinely sequential. The *path* loop can: every path
does the same arithmetic on different numbers, which is the shape a GPU wants.
So the transformation is not "vectorize the pricer", it is "keep the time loop,
make each step operate on all paths at once".

Run directly to compare the two against each other on this machine:

    python code/vectorized_pricing.py
"""

from __future__ import annotations

import math
import time

import torch

# Chapter 6's prepayment model: an S-curve in Jack's note rate minus the rate
# he is quoted, which is the Vasicek 10-year zero implied by r_t plus a 177 bp
# spread. Under Vasicek that quote is linear in r_t, so it is two constants.
BASE_CPR, MAX_CPR, STEEPNESS, MIDPOINT = 0.06, 0.45, 100.0, 0.0125
A, THETA, SIGMA, TENOR, MORTGAGE_SPREAD = 0.5, 0.045, 0.025, 10.0, 0.0177
_B = (1.0 - math.exp(-A * TENOR)) / A
_LOG_A = (_B - TENOR) * (A * A * THETA - SIGMA ** 2 / 2.0) / (A * A) - SIGMA ** 2 * _B * _B / (4.0 * A)
QUOTE_0, QUOTE_1 = -_LOG_A / TENOR + MORTGAGE_SPREAD, _B / TENOR

# The CPU timings are for four cores, as elsewhere in the book. Left at its
# default, PyTorch spreads these small element-wise steps over every core and
# spends more time synchronising the threads than computing.
torch.set_num_threads(4)


def price_paths_scalar(rate_paths, note_rate, balance, months, spread=0.0):
    """One path at a time, as Chapter 6 writes it. The reference answer."""
    dt = 1.0 / 12.0
    i = note_rate / 12.0
    pmt = balance * i / (1.0 - (1.0 + i) ** (-months))
    out = []
    # Plain Python floats: indexing a tensor one element at a time costs about
    # a microsecond per read, which would time PyTorch's dispatch, not the loop.
    for path in rate_paths.tolist():
        outstanding, price, discount = balance, 0.0, 1.0
        for k in range(1, months + 1):
            r_t = path[k]
            discount *= math.exp(-(r_t + spread) * dt)
            interest = outstanding * i
            scheduled = min(pmt - interest, outstanding)
            incentive = note_rate - (QUOTE_0 + QUOTE_1 * r_t)
            cpr = BASE_CPR + (MAX_CPR - BASE_CPR) / (1.0 + math.exp(-STEEPNESS * (incentive - MIDPOINT)))
            smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)
            prepay = max(outstanding - scheduled, 0.0) * smm
            principal = scheduled + prepay
            price += (interest + principal) * discount
            outstanding -= principal
            if outstanding <= 1e-8:
                break
        out.append(price)
    return torch.tensor(out, dtype=rate_paths.dtype)


def price_paths_vectorized(rate_paths, note_rate, balance, months, spread=0.0):
    """Every path advanced together: one month per iteration, all paths at once.

    The early-exit on a paid-off pool disappears, because paths pay down at
    different speeds and there is no longer a single loop to break out of. The
    clamp does that job instead: once a path's balance reaches zero its cash
    flows are zero, so the remaining iterations add nothing. Doing arithmetic
    that is known to contribute nothing is the price of vectorizing, and it is
    almost always worth paying.
    """
    dt = 1.0 / 12.0
    i = note_rate / 12.0
    pmt = balance * i / (1.0 - (1.0 + i) ** (-months))

    outstanding = torch.full_like(rate_paths[:, 0], balance)
    price = torch.zeros_like(outstanding)
    log_discount = torch.zeros_like(outstanding)

    for k in range(1, months + 1):
        r_t = rate_paths[:, k]
        log_discount = log_discount - (r_t + spread) * dt
        discount = torch.exp(log_discount)

        interest = outstanding * i
        scheduled = torch.clamp(torch.minimum(pmt - interest, outstanding), min=0.0)
        incentive = note_rate - (QUOTE_0 + QUOTE_1 * r_t)
        cpr = BASE_CPR + (MAX_CPR - BASE_CPR) * torch.sigmoid(STEEPNESS * (incentive - MIDPOINT))
        smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)
        prepay = torch.clamp(outstanding - scheduled, min=0.0) * smm
        principal = scheduled + prepay

        price = price + (interest + principal) * discount
        outstanding = torch.clamp(outstanding - principal, min=0.0)

    return price


def simulate_paths(r0, a, theta, sigma, months, n_paths, seed=0, device="cpu",
                   dtype=torch.float64):
    """Vasicek paths, generated on whichever device will price them."""
    g = torch.Generator(device=device).manual_seed(seed)
    dt = 1.0 / 12.0
    rates = torch.empty((n_paths, months + 1), device=device, dtype=dtype)
    rates[:, 0] = r0
    for k in range(months):
        dW = torch.normal(0.0, math.sqrt(dt), (n_paths,), generator=g,
                          device=device, dtype=dtype)
        rates[:, k + 1] = rates[:, k] + a * (theta - rates[:, k]) * dt + sigma * dW
    return rates


def _time(fn, *args, device="cpu", repeats=1, warmup=1, **kwargs):
    """Time `fn`, discarding a warmup call first.

    The warmup is not politeness, it is the difference between measuring the
    steady state and measuring a one-off. The first CUDA call in a process pays
    for kernel loading and start-up: on this workload the first call takes
    about 120 ms and every later one about 30 ms. Average three calls without
    warming up and the result lands near 60 ms, which is not what the device
    does on the second pool of the day and not a number worth quoting.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    best = float("inf")
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    # The fastest repeat, not the mean: on a laptop with mixed performance and
    # efficiency cores one repeat landing on the slow cores can multiply the
    # mean several-fold, and the minimum is what the code can actually do.
    return result, best


if __name__ == "__main__":
    R0 = 0.045
    NOTE_RATE, BALANCE, MONTHS = 0.065, 100.0, 360
    N_PATHS = 4_000

    paths = simulate_paths(R0, A, THETA, SIGMA, MONTHS, N_PATHS, seed=0)

    # Repeat: a single un-repeated timing of this varies by several fold run
    # to run, which is enough to invent a speedup that is not there.
    # No warmup for the scalar loop: it has no device to initialise.
    scalar, t_scalar = _time(price_paths_scalar, paths, NOTE_RATE, BALANCE,
                             MONTHS, repeats=3, warmup=0)
    vector, t_vector = _time(price_paths_vectorized, paths, NOTE_RATE, BALANCE,
                             MONTHS, repeats=20, warmup=3)

    # The vectorized version must be the same pricer, not merely a faster one.
    gap = float((scalar - vector).abs().max())

    print(f"{N_PATHS} paths x {MONTHS} months, float64")
    print(f"  scalar loop (CPU)   {t_scalar * 1e3:9.1f} ms")
    print(f"  vectorized  (CPU)   {t_vector * 1e3:9.1f} ms   {t_scalar / t_vector:5.1f}x")

    if torch.cuda.is_available():
        gpu_paths = paths.to("cuda")
        # Several warmups: an idle laptop GPU sits at a low clock and takes a
        # few calls to ramp up, and the first one also pays for CUDA start-up.
        gpu, t_gpu = _time(price_paths_vectorized, gpu_paths, NOTE_RATE, BALANCE,
                           MONTHS, device="cuda", repeats=10, warmup=5)
        print(f"  vectorized  (GPU)   {t_gpu * 1e3:9.1f} ms   {t_scalar / t_gpu:5.1f}x")
        gap = max(gap, float((scalar - gpu.cpu()).abs().max()))

    print(f"  max price difference vs the scalar reference: {gap:.2e}")

    if torch.cuda.is_available():
        # Where the GPU starts paying. The month loop issues 360 sequential
        # kernels whatever the path count, so at small widths that launch
        # overhead is the whole cost and the GPU loses to the CPU outright.
        print("\nGPU crossover:")
        print(f"  {'paths':>9} {'CPU':>9} {'GPU':>9} {'speedup':>9}")
        # Each side generates its own paths on its own device; the timing is
        # of the pricer, and the paths never cross the bus. The CPU stops at
        # 100,000 paths -- by then the trend is plain, and 400,000 on the CPU
        # would take longer than everything else in this file together.
        for n in (1_000, 4_000, 20_000, 100_000, 400_000):
            p_gpu = simulate_paths(R0, A, THETA, SIGMA, MONTHS, n, seed=0, device="cuda")
            _, t_g = _time(price_paths_vectorized, p_gpu, NOTE_RATE, BALANCE,
                           MONTHS, device="cuda", repeats=10, warmup=3)
            if n <= 100_000:
                p_cpu = simulate_paths(R0, A, THETA, SIGMA, MONTHS, n, seed=0)
                _, t_c = _time(price_paths_vectorized, p_cpu, NOTE_RATE, BALANCE,
                               MONTHS, repeats=10, warmup=2)
                print(f"  {n:9,} {t_c * 1e3:8.1f}ms {t_g * 1e3:8.1f}ms {t_c / t_g:8.2f}x")
                del p_cpu
            else:
                print(f"  {n:9,} {'--':>10} {t_g * 1e3:8.1f}ms {'--':>9}")
            del p_gpu
            torch.cuda.empty_cache()
