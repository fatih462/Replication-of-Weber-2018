#!/usr/bin/env python3
"""Build duration-sorted decile portfolios following Weber (2018), Section 3.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MONTHLY_COLUMNS = [
    "PERMNO",
    "PERMCO",
    "gvkey",
    "YYYYMM",
    "year",
    "month",
    "ret",
    "me_firm_millions",
]

DURATION_COLUMNS = [
    "gvkey",
    "conm",
    "datadate",
    "formation_year",
    "dur",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Weber duration deciles and equal-weight portfolio returns."
    )
    parser.add_argument(
        "--duration-parquet",
        type=Path,
        default=Path("data/cash_flow_duration.parquet"),
        help="Parquet file produced by build_cash_flow_duration.py.",
    )
    parser.add_argument(
        "--monthly-parquet",
        type=Path,
        default=Path("data/crsp_monthly_clean.parquet"),
        help="Cleaned monthly CRSP parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for portfolio output files.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1963,
        help="First June portfolio formation year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=(
            "Last June portfolio formation year. Defaults to the latest complete "
            "July-June holding period available in the local monthly data."
        ),
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def clean_gvkey(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def assign_deciles(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    if out["dur"].notna().sum() < 10:
        out["duration_decile"] = pd.NA
        return out
    ranks = out["dur"].rank(method="first")
    out["duration_decile"] = pd.qcut(ranks, 10, labels=range(1, 11)).astype("Int64")
    return out


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    duration = pd.read_parquet(args.duration_parquet, columns=DURATION_COLUMNS)
    monthly = pd.read_parquet(args.monthly_parquet, columns=MONTHLY_COLUMNS)
    require_columns(duration, set(DURATION_COLUMNS), args.duration_parquet)
    require_columns(monthly, set(MONTHLY_COLUMNS), args.monthly_parquet)

    duration["gvkey"] = clean_gvkey(duration["gvkey"])
    duration["formation_year"] = pd.to_numeric(duration["formation_year"], errors="coerce")
    duration["dur"] = pd.to_numeric(duration["dur"], errors="coerce")

    monthly["gvkey"] = clean_gvkey(monthly["gvkey"])
    for col in ["PERMNO", "PERMCO", "YYYYMM", "year", "month", "ret", "me_firm_millions"]:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    return duration, monthly


def latest_complete_end_year(duration: pd.DataFrame, monthly: pd.DataFrame) -> int:
    max_duration_year = int(duration.loc[duration["dur"].notna(), "formation_year"].max())
    june_returns = monthly.loc[monthly["month"].eq(6), "year"].dropna()
    if june_returns.empty:
        raise ValueError("Monthly CRSP data contain no June observations.")
    max_return_complete_sort_year = int(june_returns.max()) - 1
    max_june_formation_year = int(
        monthly.loc[monthly["month"].eq(6) & monthly["gvkey"].notna(), "year"].max()
    )
    return min(max_duration_year, max_june_formation_year, max_return_complete_sort_year)


def build_assignments(
    duration: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    june = monthly.loc[
        monthly["month"].eq(6)
        & monthly["year"].between(start_year, end_year)
        & monthly["PERMNO"].notna()
        & monthly["gvkey"].notna(),
        ["PERMNO", "PERMCO", "gvkey", "year", "YYYYMM", "me_firm_millions"],
    ].copy()
    june = june.rename(columns={"year": "sort_year", "YYYYMM": "june_yyyymm"})
    june = june.drop_duplicates(["sort_year", "PERMNO"])

    sort_duration = duration.loc[
        duration["formation_year"].between(start_year, end_year)
        & duration["gvkey"].notna()
        & duration["dur"].notna(),
        ["gvkey", "conm", "datadate", "formation_year", "dur"],
    ].copy()
    sort_duration = sort_duration.rename(columns={"formation_year": "sort_year"})

    assignments = june.merge(sort_duration, on=["gvkey", "sort_year"], how="inner")
    assignments = assignments.dropna(subset=["dur"])
    assignments = pd.concat(
        [assign_deciles(group) for _, group in assignments.groupby("sort_year", sort=True)],
        ignore_index=True,
    )
    assignments = assignments.dropna(subset=["duration_decile"])
    assignments["duration_decile"] = assignments["duration_decile"].astype(int)
    return assignments.sort_values(["sort_year", "duration_decile", "PERMNO"]).reset_index(
        drop=True
    )


def add_holding_period_sort_year(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    out["sort_year"] = np.where(out["month"].ge(7), out["year"], out["year"] - 1)
    out["holding_month"] = np.where(out["month"].ge(7), out["month"] - 6, out["month"] + 6)
    return out


def build_monthly_returns(assignments: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    holding_returns = add_holding_period_sort_year(monthly)
    holding_returns = holding_returns.loc[
        holding_returns["ret"].notna(),
        ["PERMNO", "YYYYMM", "year", "month", "sort_year", "holding_month", "ret"],
    ].copy()

    stock_months = assignments.loc[
        :,
        ["sort_year", "PERMNO", "gvkey", "duration_decile", "dur"],
    ].merge(holding_returns, on=["sort_year", "PERMNO"], how="inner")
    stock_months = stock_months.loc[stock_months["holding_month"].between(1, 12)].copy()

    monthly_returns = (
        stock_months.groupby(["sort_year", "YYYYMM", "holding_month", "duration_decile"])
        .agg(
            ew_ret=("ret", "mean"),
            n_stocks=("ret", "count"),
            median_dur=("dur", "median"),
        )
        .reset_index()
        .sort_values(["sort_year", "holding_month", "duration_decile"])
    )
    return monthly_returns


def build_annual_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    annual = (
        monthly_returns.sort_values(["sort_year", "duration_decile", "holding_month"])
        .groupby(["sort_year", "duration_decile"])
        .agg(
            annual_return=("ew_ret", lambda returns: (1.0 + returns).prod() - 1.0),
            months=("ew_ret", "count"),
            avg_monthly_stocks=("n_stocks", "mean"),
        )
        .reset_index()
    )
    return annual.loc[annual["months"].eq(12)].copy()


def build_figure1_data(assignments: pd.DataFrame, annual_returns: pd.DataFrame) -> pd.DataFrame:
    median_duration = (
        assignments.groupby(["sort_year", "duration_decile"])["dur"].median().reset_index()
    )
    figure_data = annual_returns.merge(
        median_duration,
        on=["sort_year", "duration_decile"],
        how="inner",
    )
    figure_data = (
        figure_data.groupby("duration_decile")
        .agg(
            avg_median_dur=("dur", "mean"),
            avg_annual_return=("annual_return", "mean"),
            years=("annual_return", "count"),
            avg_monthly_stocks=("avg_monthly_stocks", "mean"),
        )
        .reset_index()
        .sort_values("duration_decile")
    )
    return figure_data


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    assignments: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    annual_returns: pd.DataFrame,
    figure_data: pd.DataFrame,
) -> None:
    report = {
        "method": {
            "source": "Weber (2018), Section 3.1 and Figure 1 caption",
            "sort": (
                "At each June formation year t, assign stocks to duration deciles "
                "using Dur from fiscal years ending in calendar year t-1."
            ),
            "returns": (
                "Equal-weight monthly portfolio returns from July t through June t+1, "
                "compounded to annual holding-period returns."
            ),
        },
        "inputs": {
            "duration_parquet": str(args.duration_parquet),
            "monthly_parquet": str(args.monthly_parquet),
        },
        "outputs": {
            "assignments": str(args.output_dir / "duration_decile_assignments.parquet"),
            "monthly_returns": str(args.output_dir / "duration_decile_monthly_returns.parquet"),
            "figure1_data": str(args.output_dir / "figure1_duration_term_structure.parquet"),
        },
        "coverage": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "assigned_stock_years": int(len(assignments)),
            "formation_years": int(assignments["sort_year"].nunique()),
            "monthly_portfolio_rows": int(len(monthly_returns)),
            "annual_portfolio_rows_with_12_months": int(len(annual_returns)),
            "figure_deciles": int(len(figure_data)),
        },
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stale_annual_path = args.output_dir / "duration_decile_annual_returns.parquet"
    if stale_annual_path.exists():
        stale_annual_path.unlink()

    duration, monthly = load_inputs(args)
    if args.end_year is None:
        args.end_year = latest_complete_end_year(duration, monthly)
    assignments = build_assignments(
        duration,
        monthly,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    monthly_returns = build_monthly_returns(assignments, monthly)
    annual_returns = build_annual_returns(monthly_returns)
    figure_data = build_figure1_data(assignments, annual_returns)

    assignments.to_parquet(args.output_dir / "duration_decile_assignments.parquet", index=False)
    monthly_returns.to_parquet(
        args.output_dir / "duration_decile_monthly_returns.parquet", index=False
    )
    figure_data.to_parquet(
        args.output_dir / "figure1_duration_term_structure.parquet", index=False
    )
    write_report(
        args.output_dir / "duration_decile_portfolios_report.json",
        args=args,
        assignments=assignments,
        monthly_returns=monthly_returns,
        annual_returns=annual_returns,
        figure_data=figure_data,
    )

    print(f"Wrote duration decile outputs to {args.output_dir}")
    print(
        figure_data.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )


if __name__ == "__main__":
    main()
