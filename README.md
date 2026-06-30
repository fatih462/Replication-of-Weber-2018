# Replication of Weber (2018)

This repository contains code and selected outputs for a replication and sample
extension of Michael Weber's paper "Cash flow duration and the term structure of
equity returns" (Journal of Financial Economics, 2018). The replication uses
Weber's data setup and extends the analysis through 2025. The project rebuilds
firm-level cash-flow duration from CRSP/Compustat data, forms duration-sorted
portfolios, replicates several tables and figures from the paper, and scrapes
SEC 13F filings to do the institutional ownership analysis.

## Repository Structure

- `code/`: Python scripts for data cleaning, variable construction, portfolio
  formation, tables, figures, and additional subsample checks.
- `tables/`: selected table outputs, mainly PDFs and small diagnostic outputs.
- `figures/`: generated figures.
- `data/`: local data directory. Large or licensed data files are not included
  in the repository.

Run scripts from the repository root, for example:

```bash
python3 code/table5.py
```

## Data Pipeline

The replication starts from merged CRSP/Compustat extracts downloaded from WRDS.
These files are expected in `data/raw/`, but the licensed CRSP/Compustat files
are not included in this repository.

1. Clean CRSP/Compustat data:

```bash
python3 code/data_cleaning_crsp_compustat.py
```

This applies the main Weber sample screens: common U.S. stocks, exchange and
security filters, financials/utilities exclusions, delisting-return treatment,
and accounting variable construction. It writes cleaned monthly CRSP and annual
Compustat parquet files.

2. Build cash-flow duration:

```bash
python3 code/build_cash_flow_duration.py
```

This constructs firm-year cash-flow duration following Weber's duration
methodology using book equity, profitability, growth, market equity, and the
forecasting assumptions described in the paper.

3. Form duration portfolios:

```bash
python3 code/build_duration_portfolios.py
```

This sorts stocks into duration deciles at the end of June and computes
equal-weighted July-to-June portfolio returns.

4. Build institutional ownership and residual institutional ownership:

```bash
python3 code/scrape_13f_holdings.py
python3 code/build_residual_institutional_ownership.py
```

The 13F scripts download and process public SEC 13F holdings data. Residual
institutional ownership is computed by residualizing logit institutional
ownership on log market equity and squared log market equity, following Weber's
Table 10 construction. The local 13F data cover the extended sample after the
original Thomson Reuters 13F sample, so Table 10 is reproduced for the
2015-2025 extension using public SEC filings.

5. Produce tables and figures:

```bash
python3 code/table1.py
python3 code/table2.py
python3 code/table3.py
python3 code/table5.py
python3 code/table7.py
python3 code/table10.py
python3 code/figure1.py
```

The scripts write outputs to `tables/` and `figures/`.

## Additional Checks

The file `code/analyze_duration_premium_subsamples.py` performs simple
subsample checks for interpreting the extended-sample results. It compares the
short-duration premium across periods such as the dotcom run-up, dotcom bust,
post-GFC period, 2015-2025 extension, growth-led months, and value-led months.
These checks are descriptive and are not meant as a full causal explanation.

## Data Availability

The repository does not include licensed CRSP/Compustat data or large parquet
intermediate files.
Large SEC 13F holdings files are also omitted from GitHub because of size, but
the scraping and processing code is provided.
