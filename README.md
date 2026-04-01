# Source Code: What Price Presupposes

This repository contains the 11-step Python computational pipeline for the paper "What Price Presupposes: Quantifying Labour Obligations in Speculative Market Regimes". 

The pipeline processes financial data to decompose market price, extracting the portion corresponding to validated productive activity (VProd_base) and the portion corresponding to claims on uninitiated production cycles (KΠ′)[span_2](end_span). 

## System Requirements
To run this pipeline and reproduce the results, you will need:
* A Python 3.x environment.
* Active internet access to fetch data from the SEC EDGAR company facts API and Yahoo Finance APIs.
* Required Python packages (please refer to `requirements.txt`).

## Configuration
[span_4](start_span)The entire pipeline is config-driven through a `survey_config.yaml` file[span_4](end_span). This file specifies:
* [span_5](start_span)Company tickers[span_5](end_span)
* [span_6](start_span)Sectors[span_6](end_span)
* [span_7](start_span)CIK identifiers[span_7](end_span)
* [span_8](start_span)Status flags (active/delete)[span_8](end_span)

## The 11-Step Pipeline
[span_9](start_span)The pipeline consists of 11 Python scripts executed in sequence[span_9](end_span). [span_10](start_span)Each step has a single responsibility and communicates with subsequent steps through file-system outputs[span_10](end_span).

* **Step 1 (`01_Crawl_SEC_Data.py`)**: Fetches quarterly financial data from the SEC EDGAR company facts API, applies XBRL fallback chains, and filters to discrete quarterly periods[span_11](end_span).
* **Step 2 (`02_Clean_raw_Data.py`)**: Applies data quality rules, flags, and removes companies with missing CSVs or anomalous margin sums[span_12](end_span).
* **Step 3 (`03_Crawl_benchmark_Data.py`)**: Fetches financial data for sector benchmark cohorts using the same XBRL methodology[span_13](end_span).
* **Step 4 (`04_Benchmark_calculate.py`)**: Computes rolling 12-quarter sector median margins with Tier A/B quality flagging[span_14](end_span).
* **Step 5 (`05_Crawl_market_cap.py`)**: Fetches historical prices from Yahoo Finance and merges them with SEC shares outstanding[span_15](end_span).
* **Step 6 (`06_Clean_market_cap.py`)**: Applies a continuity filter (minimum 6 consecutive quarters) and updates the config status[span_16](end_span).
* **Step 7 (`07_Kbrand_calculate.py`)**: Estimates KBrand using brand scores and sector multipliers[span_17](end_span).
* **Step 8 (`08_Framework_calculate.py`)**: Computes core framework variables: VProd_base, E*, KΠ′, Rt, PDIt, and formal gate variables[span_18](end_span).
* **Step 9 (`09_Classify_configurations.py`)**: Applies Proposition 7 classification rules to assign structural configurations (C1-C6 or Normal) to each period[span_19](end_span).
* **Step 10 (`10_Merge_cycles.py`)**: Combines individual company outputs into a unified panel dataset[span_20](end_span).
* **Step 11 (`11_Statistical_analysis.py`)**: Runs the full statistical battery, including the Markov matrix, Kruskal-Wallis, post-hoc tests, PDI dynamics, and robustness tests[span_21](end_span).

## Citation & DOI
https://doi.org/10.5281/zenodo.19272790
