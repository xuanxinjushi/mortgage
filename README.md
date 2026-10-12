# Mortgage Markets in the Age of AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Package: mortgagekit](https://img.shields.io/pypi/v/mortgagekit?color=brightgreen&label=mortgagekit)](https://pypi.org/project/mortgagekit/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Amazon](https://img.shields.io/badge/Amazon-Available%20on%20Amazon-FF9900?logo=amazon&logoColor=white)](https://www.amazon.com/dp/B0HKYDTK8C/)

<p align="left">
  <a href="https://www.amazon.com/dp/B0HKYDTK8C/">
    <img src="https://m.media-amazon.com/images/I/71sseRBOoVL._SL1500_.jpg" alt="Mortgage Markets in the Age of AI" width="300" />
  </a>
</p>

Companion open-source codebase, Jupyter notebooks, and empirical pricing models for the book:

> **Mortgage Markets in the Age of AI**  
> *Financial Principles, Pricing, Risk, and Machine Learning*  
> **Authors:** Chiu Yan & Xuan Xin  
> **Get the Book:** [Available on Amazon](https://www.amazon.com/dp/B0HKYDTK8C/)  
> **Website & Issues:** [https://github.com/xuanxinjushi/mortgage](https://github.com/xuanxinjushi/mortgage)

---

## 📖 About the Book & Codebase

The US residential mortgage market is one of the largest debt markets in the world: American households owed \$13.9 trillion on one-to-four-family homes at the end of the first quarter of 2026. The book starts from the first cash flow of a single home loan and follows it to what an institutional desk does with it: pricing it under uncertainty, quoting it as an option-adjusted spread, hedging a position whose duration will not hold still, building the platforms that compute all of this, and layering machine-learning models on top.

Every technical term arrives first in plain language, from the point of view of the person who uses it, and only then as a formula. Every chapter closes with a **retail window** that restates the material for the reader who holds mortgages through a bond fund: roughly a fifth to a quarter of a total bond index fund is agency MBS.

This repository provides reproducible implementations of the quantitative models, valuation engines, risk sensitivities, and machine learning architectures developed throughout the book, plus the source of **mortgagekit**, the book's Python library.

### The Cast of Market Participants

Rather than treating formulas in an academic vacuum, the book ties every algorithm to the person who uses it. Eight characters follow one 30-year payment stream:

- **Jack (The Borrower)** — Takes out the 30-year loan and can repay it early at any time — by refinancing or moving — the option that makes a mortgage different from a bond.
- **Maya (The Loan Officer)** — Turns Jack's promise into a loan, and has to defend her underwriting under the CFPB's ability-to-repay / QM rules.
- **Priya (The Servicer)** — Keeps the monthly payments moving, earns the servicing strip (MSR) that prepayment can cut short, and supplies the loan-level data the prepayment models learn from.
- **Benny (The Dealer)** — Turns many loans into a security that can trade, and makes markets in it.
- **Elena (The Investor)** — Holds pass-throughs and CMO tranches, values them by OAS, and hedges the negative convexity on the desk.
- **Peter (The Quant)** — Calibrates the rate models, builds and maintains the pricing engine, and grafts machine-learning layers onto it.
- **Melissa (The Examiner)** — Challenges the models independently, and asks how bad a year could get.
- **Simon (The Retail Investor)** — Reaches the same mortgage cash flows through a bond fund or a mortgage REIT; his view is each chapter's retail window.

---

## 🗂️ Chapter Code Directory

Each chapter in this repository is self-contained with Python scripts and dual-language (English and Chinese `_zh.ipynb`) interactive Jupyter Notebooks:

| Chapter | Focus & Mechanics | Key Scripts & Notebooks |
| :--- | :--- | :--- |
| **[Chapter 1](chapter1-us-mortgage-market/code/)**<br>US Mortgage Market | Amortization mechanics, loan-to-pool aggregation, institutional holders. | `amortization.py`<br>`loan_to_pool_representation.py` (`.ipynb`)<br>`mbs_holders.py` |
| **[Chapter 2](chapter2-chance-and-prepayment/code/)**<br>Chance & Prepayment | Prepayment rate conventions (CPR vs SMM), non-prepayment survival probability, empirical hazard curves. | `cpr_smm.py`<br>`survival_analysis.py` (`.ipynb`) |
| **[Chapter 3](chapter3-how-rates-move/code/)**<br>How Rates Move | Par curve bootstrapping, continuous discounting, stochastic rate simulation, Neural SDE paths, HPI correlation, proxy curves and the hedge Jacobian. | `bootstrap_curve.py`<br>`discounting_and_paths.py`<br>`neural_sde_rates.py` (`.ipynb`)<br>`rate_hpi_correlation.py`<br>`proxy_curve_jacobian.py` |
| **[Chapter 4](chapter4-turning-change-into-math/code/)**<br>Turning Change into Math | Itô's lemma path verification, PyTorch autograd Greeks (DV01, CV01, cross-gamma), parameter calibration identifiability. | `autograd_greeks.py` (`.ipynb`)<br>`calibration_identifiability.py`<br>`ito_verification.py` |
| **[Chapter 5](chapter5-mortgage-optionality/code/)**<br>Mortgage Optionality | Black-Scholes embedded call option pricing, behavioral exercise network capturing non-rational borrower friction. | `black_scholes.py`<br>`behavioral_exercise_net.py` (`.ipynb`) |
| **[Chapter 6](chapter6-pricing-with-uncertainty/code/)**<br>Pricing with Uncertainty | Monte Carlo Option-Adjusted Spread (OAS) solver, current coupon models, nominal spread vs Z-spread, primary-secondary spread decomposition, cashflow waterfall, neural pricing surrogate. | `oas_engine.py`<br>`oas_variants.py`<br>`current_coupon.py`<br>`nominal_vs_zspread.py`<br>`mortgage_treasury_spread.py`<br>`waterfall.py`<br>`neural_pricing_surrogate.py` (`.ipynb`, `_demo.ipynb`) |
| **[Chapter 7](chapter7-why-prepayment-models-matter/code/)**<br>Prepayment Modeling | Fannie Mae loan-level performance tape streaming reader, hybrid residual prepayment modeling (parametric S-curve + deep residual). | `fannie_tape.py`<br>`hybrid_residual_model.py` (`.ipynb`) |
| **[Chapter 8](chapter8-the-desk/code/)**<br>The MBS Trading Desk | Duration and swaption hedging against negative convexity, MSR valuation and hedging, TBA dollar roll analytics, coupon-stack allocation, the refinance S-curve on real loan data, one pricing kernel shared by desk/risk/accounting, scenario reports, desk drift & PSI monitoring. | `duration_hedging.py`<br>`swaption_hedge.py`<br>`msr_hedge.py`<br>`dollar_roll.py`<br>`coupon_allocation.py`<br>`scurve_from_tape.py`<br>`desk_drift_psi_monitor.py` (`.ipynb`)<br>`mini_desk.py`<br>`one_kernel_many_books.py`<br>`scenario_catalog.py`<br>`jacobian_cache_demo.py` |
| **[Chapter 9](chapter9-ai-ml-modeling/code/)**<br>AI & Machine Learning | Feature engineering on a real loan tape, hybrid residual models, ML vs classical logistic prepayment RMSE benchmarks, Cox vs Random Survival Forests, black-box failure modes and monotonic fixes, Neural SDE Vasicek, diffusion-model scenarios, differential training on Greeks, CUDA pricing kernel. | `feature_engineering.py`<br>`hybrid_residual_model.py`<br>`ml_vs_classical_rmse.py` (`.ipynb`)<br>`cox_vs_survival_trees.py`<br>`blackbox_failure.py`<br>`monotone_fix.py`<br>`time_split.py`<br>`neural_sde_vasicek.py` (`.ipynb`)<br>`generative_scenarios.py`<br>`differential_training.py` (`differential_training_explained.ipynb`)<br>`vectorized_pricing.py`<br>`cuda_pricing_kernel.cu` |
| **[Chapter X](chapterx/code/)**<br>Appendix & Reference | Single-family loan performance tape inspection, data dictionary, and summary statistics. | `describe_tape.py` |

---

## ⚡ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/xuanxinjushi/mortgage.git
cd mortgage-markets-in-the-age-of-ai
```

### 2. Set Up Python Environment

Python 3.10 or higher is recommended. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Companion Package

The computational engine of the book is packaged as `mortgagekit`:

```bash
pip install mortgagekit
```

To run the chapter code, including the machine-learning models (PyTorch):

```bash
pip install "mortgagekit[ml]"
```

Or install dependencies directly:

```bash
pip install numpy scipy matplotlib torch scikit-learn xgboost jupyterlab
```

---

## 🚀 Running Code & Notebooks

### Running Standalone Python Scripts

As explained in the book's preface, each chapter keeps its own directory. Commands in the text (such as `python code/something.py`) are meant to be executed from within that chapter's directory:

```bash
# Chapter 1: Monthly payment and loan amortization
cd chapter1-us-mortgage-market
python code/amortization.py

# Chapter 3: Curve bootstrapping and stochastic rate paths
cd ../chapter3-how-rates-move
python code/discounting_and_paths.py

# Chapter 6: Monte Carlo OAS engine
cd ../chapter6-pricing-with-uncertainty
python code/oas_engine.py

# Chapter 8: Desk duration hedging and negative convexity
cd ../chapter8-the-desk
python code/duration_hedging.py
```

### Running Interactive Jupyter Notebooks

Launch JupyterLab to interactively explore each model and its visualizations:

```bash
jupyter lab
```

Each chapter notebook provides both English (`*.ipynb`) and Chinese (`*_zh.ipynb`) editions with step-by-step mathematical explanations and rendered plots.

---

## 📊 Data Sources & Reproducibility

1. **Public Market Data Snapshots**:
   - Zero-coupon Treasury yield curves (Federal Reserve H.15 / Treasury par curves)
   - Secured Overnight Financing Rate (SOFR)
   - Freddie Mac Primary Mortgage Market Survey (PMMS 30-year conforming fixed rate)
   - S&P CoreLogic Case-Shiller Home Price Index (HPI)

2. **Loan-Level Performance Data**:
   - Fannie Mae Single-Family Loan Performance dataset.
   - Chapter 7 and Chapter X provide high-performance streaming readers (`fannie_tape.py` and `describe_tape.py`) that process millions of records without requiring expensive database infrastructure.

3. **Synthetic Data**: some results are run on data a script generates with the answer built in, to show that a method works. The book's data appendix (Chapter X) says which numbers rest on real data and which on synthetic, and what the real data cannot tell you.

---

## 📝 Errata and Contributions

If you find an error in the text, code, or notebooks, we welcome issues and pull requests!

- **Issue Tracker:** [https://github.com/xuanxinjushi/mortgage/issues](https://github.com/xuanxinjushi/mortgage/issues)
- Please mention the relevant chapter, section, or script name to help us fix it quickly.

---

## ⚖️ License & Disclaimer

- **Code License:** The code in this repository is licensed under the [Apache 2.0 License](LICENSE).
- **Book Content:** Book text, structural materials, and book illustrations are Copyright © 2026 Chiu Yan & Xuan Xin. All rights reserved.
- **Disclaimer:** All code and models in this repository are developed for pedagogical and educational purposes. They do not constitute financial, investment, legal, tax, or credit advice.
