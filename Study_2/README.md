# What Price Presupposes: Source Code

This repository contains the computational pipeline for the paper:

> **Nguyen, H. P. (2026). What Price Presupposes: Quantifying Labour Obligations in Speculative Market Regimes.**
> DOI: [10.5281/zenodo.19366083](https://doi.org/10.5281/zenodo.19366083)

The pipeline operationalises a five-component price decomposition derived from the M-C-M' circuit. It extracts the verified productive base (V_Prod_base), the Expectation Index (E*), and the uninitiated obligation stock (KΠ') from quarterly SEC filings and market price data. The A/B decomposition of the obligation ratio E₃ separates the productive growth component (A = 1 + PGR_t) from the surplus extraction component (B = E₃ − A), providing empirical instruments for the Expectation Chimera lifecycle analysis across six structural configurations (C1–C6).

---

## Repository Structure

```
what-price-presupposes-code/
├── Study_1/              Purposive sample (21 companies, Technology/Retail/Services)
├── Study_2/
│   ├── Main_Pipeline/    Data collection and framework calculation
│   └── Statistis/        Statistical analysis, ML instruments, robustness tests
└── Out_of_sample/        Independent validation dataset pipeline
```

---

## System Requirements

- Python 3.8 or higher
- Active internet access (SEC EDGAR company facts API and Yahoo Finance)
- Required packages: see `requirements.txt`

---

## Study 1

Purposive sample of 21 companies across Technology, Retail, and Services sectors. 709 company-quarter observations, 2015–2025. Study 1 establishes that the framework is operational and that findings are directionally consistent with theoretical predictions in productive sectors where the M-C-M' revenue proxy holds.

**Pipeline** (single folder, 11 steps):

| Step | Script | Responsibility |
|------|--------|----------------|
| 1 | `01_Crawl_SEC_Data.py` | Fetches quarterly financial data from SEC EDGAR via company facts API with XBRL fallback chains |
| 2 | `02_Clean_raw_Data.py` | Applies data quality filters; removes companies failing minimum criteria |
| 3 | `03_Crawl_benchmark_Data.py` | Fetches sector benchmark cohort data |
| 4 | `04_Benchmark_calculate.py` | Computes rolling 12-quarter sector median margins |
| 5 | `05_Crawl_market_cap.py` | Fetches historical closing prices from Yahoo Finance; merges with shares outstanding |
| 6 | `06_Clean_market_cap.py` | Applies continuity filter; updates config status flags |
| 7 | `07_Kbrand_calculate.py` | Estimates K_Brand using brand scores and sector multipliers (ISO 10668/20671) |
| 8 | `08_Framework_calculate.py` | Computes V_Prod_base, E*, KΠ', E₀–E₃, R_t, PDI_t, PGR_t, and gate conditions |
| 9 | `09_Classify_configurations.py` | Assigns structural configurations C1–C6 or Normal based on R_t and ΔKΠ' |
| 10 | `10_Merge_cycles.py` | Combines company outputs into unified panel; computes kinematic variables for C2 analysis |
| 11 | `11_Statistical_analysis.py` | Runs full statistical battery |

**Config:** `survey_config.yaml` (company tickers, sectors, CIK identifiers, status flags)

---

## Study 2

Quasi-random sample starting from 1,603 US-listed companies. After applying four data quality filters, 274 companies were retained (17.1% retention rate) across Technology, Healthcare, and Financials and Real Estate. 2,573 company-quarter observations.

### Main Pipeline

Three config files are generated sequentially before data collection begins.

| Step | Script | Responsibility |
|------|--------|----------------|
| 0 | `00_Create_SEC_Ticker_config.py` | Converts company list to SEC_Ticker_config.yaml |
| 1 | `01_Create_benchmark_ticker_config.py` | Creates Benchmark_config.yaml for sector cohorts |
| 2 | `02_Create_survey_config.py` | Generates Survey_config.yaml from filtered ticker list |
| 3 | `03_Crawl_benchmark.py` | Fetches sector benchmark data from SEC EDGAR |
| 4 | `04_Crawl_all_sample.py` | Fetches financial data for all sample companies |
| 5 | `05_Benchmark_calculate.py` | Computes rolling 12-quarter sector median margins |
| 6 | `06_Clean_raw_Data.py` | Applies four data quality filters |
| 7 | `07_Crawl_margin.py` | Fetches and computes operating margins |
| 8 | `08_Clean_market_cap.py` | Fetches and cleans market capitalisation data |
| 9 | `09_KBrand_calculate.py` | Estimates K_Brand using brand scores and sector multipliers |
| 10 | `10_Framework_calculate.py` | Computes all framework variables |
| 11 | `11_Classify_configurations.py` | Assigns structural configurations C1–C6 or Normal |
| 12 | `12_Merge_cycles.py` | Merges company outputs into unified panel; adds kinematic variables |

**Config files:** `SEC_Ticker_config.yaml`, `Benchmark_config.yaml`, `Survey_config.yaml`

### Statistis

All statistical analysis, ML instruments, robustness tests, placebo tests, and diagnostic tests.

| Script | Purpose |
|--------|---------|
| `01_Statistical_analysis.py` | Core statistical battery: Markov transition matrix with company-level bootstrap CIs, Kruskal-Wallis H-tests, pairwise Mann-Whitney U with Bonferroni correction, PDI directional tests |
| `02_C2_linear_test.py` | C2 gestation pathway analysis; B vs E₃ threshold comparison |
| `03_Advance_statistical_analysis.py` | Panel fixed effects (within-firm transformation), Granger causality, temporal stability (pre/post 2020) |
| `04_Financials_structer_test.py` | Sector boundary condition tests; permutation test for Financials contamination |
| `05_C3C4_ML_Pipeline.py` | Maturity risk stratification instrument: feature selection, logistic regression with undersampling, bootstrap robustness |
| `06_OutOfSampleValidation_test.py` | Out-of-sample evaluation of maturity instrument; 694 duplicate observations removed; risk bucket analysis |
| `07_Chimera_Discription.py` | Expectation Chimera lifecycle descriptive analysis; configuration entry/exit rates by sector |
| `08_C2_Twostage_Casade_pipeline.py` | Two-stage cascade model for C2 bifurcation: Stage 1 evolution gate, Stage 2 collapse discriminator |
| `09_C2_Twostage_OOS_Validation.py` | Out-of-sample evaluation of C2 cascade; 149 duplicate observations removed |
| `11_KBrand_Robustness.py` | K_Brand descriptive statistics, Mann-Whitney tests between configurations, bootstrap CIs |
| `12_K_Brand_Robustness_Report.py` | End-to-end K_Brand scaling robustness: recomputes full framework pipeline under factors 0.25x–1.00x |
| `13_Identity_Check_Report.py` | Verifies algebraic identity E* = E₀+E₁+E₂+E₃ across all observations. Supports `--exclude_financials` flag. Identity holds by arithmetic construction in all sectors; for Financials it confirms pipeline integrity, not theoretical validity of the proxy |
| `14_Placebo_Report.py` | Three placebo tests: (1) random KΠ' from sector-year distribution recomputed through full pipeline, (2) shuffled configuration labels within firm, (3) PDI at lags 1–12 |
| `15_PDI_Placebo_Report.py` | PDI structural state placebo tests: Approach A (within-firm temporal shuffle, 200 iterations), Approach B (cross-firm shuffle within sector, 200 iterations) |
| `16_Diagnostics_Report.py` | Three diagnostic tests: (1) PDI level vs change variants, (2) logistic regression with interaction term PDI_roll3 × dK_Pi_prime_pct, (3) Markov path dependency permutation test (1,000 iterations) |
| `17_Plot_figures.py` | Generates four main paper figures: Markov transition diagram, price decomposition, C2 bifurcation trajectories, A/B decomposition |

---

## Out-of-Sample Dataset

The out-of-sample dataset uses the same pipeline as Study 2 Main Pipeline applied to an independently sourced company list (`sec_companies.xlsx`). The resulting dataset overlaps with Study 2 in calendar time. Duplicate company-quarter observations are removed at the evaluation stage before performance metrics are computed: 694 duplicates removed for the maturity instrument and 149 for the C2 cascade. The deduplicated output is passed to Study_2/Statistis scripts 06 and 09 for validation.

| Step | Script | Note |
|------|--------|------|
| 0–2 | Config generation | Same as Study 2; input from `sec_companies.xlsx` |
| 3–12 | Data collection and framework | Identical logic to Study 2 Main Pipeline; `09_KBrand_calculate.py` not included |

**Config files:** `SEC_Ticker_config.yaml`, `Benchmark_config.yaml`, `Survey_config.yaml`

---

## Key Variables

| Variable | Definition |
|----------|-----------|
| `V_Prod_base` | Revenue × (1 − operating margin); GAAP proxy for c + v |
| `K_Pi_prime` | market_cap − V_Prod_base − s_baseline − S_Surplus − K_Brand |
| `E_star` | (market_cap − V_Prod_base) / V_Prod_base — Expectation Index |
| `E_3` | K_Pi_prime / V_Prod_base — uninitiated obligation ratio |
| `A` | 1 + PGR_t — productive growth component |
| `B` | E₃ − A — surplus extraction component |
| `R_t` | s_t / K_Pi_prime(t−1) — absorption ratio |
| `PDI_t` | s_t / (|ΔK_Pi_prime| + s_t) — Productive Discharge Index |

---

## Sector Boundary Conditions

The M-C-M' revenue proxy holds for Technology, Healthcare, Retail, and Services where revenue reflects discrete commodity production cycles. It is structurally weaker for Financials and Real Estate where revenue reflects financial intermediation. Financials are retained in Study 2 because permutation tests indicate the Markov transition structure is not contaminated (matrix distance = 0.000, p = 1.00). E* and E₃ do not carry the same theoretical interpretation for Financials as for productive sectors.

---

## Reproducibility

- All random seeds set to 42
- Bootstrap confidence intervals use company-level resampling to account for within-company autocorrelation
- SEC EDGAR API requests are throttled to respect the 10 requests/second rate limit
- Full pipeline from raw data to final results requires approximately 4–6 hours depending on API response times

---

## Citation

```bibtex
@software{nguyen2026whatprice,
  author    = {Nguyen, H. P.},
  title     = {Nomatter2021/what-price-presupposes-code: 
               Source Code for What Price Presupposes V2.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19366083},
  url       = {https://doi.org/10.5281/zenodo.19366083}
}
```
