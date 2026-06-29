#!/usr/bin/env python3
"""Print institutional ownership descriptives and plot mean IOR by year."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


YEAR_CANDIDATES = [
    "ior_report_year",
    "report_year",
    "formation_year",
    "fyear",
    "year",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print mean and standard errors of institutional ownership."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cash_flow_duration_with_ior.parquet"),
        help="Input parquet containing institutional ownership.",
    )
    parser.add_argument(
        "--ior-column",
        default="ior",
        help="Column containing institutional ownership ratios.",
    )
    parser.add_argument(
        "--year-column",
        default=None,
        help=(
            "Year column for the trend plot. Defaults to the first available of "
            f"{', '.join(YEAR_CANDIDATES)}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/ior_mean_trend.png"),
        help="Path for the saved mean IOR trend plot.",
    )
    parser.add_argument(
        "--clip-ior",
        action="store_true",
        help="Clip IOR to [0, 1] before computing descriptives.",
    )
    return parser.parse_args()


def choose_year_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"{requested} is not a column in the input file.")
        return requested

    for column in YEAR_CANDIDATES:
        if column in frame.columns:
            return column
    raise ValueError(
        "Could not infer a year column. Pass --year-column with the column to use."
    )


def standard_error(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    return valid.std(ddof=1) / np.sqrt(len(valid))


def summarize_by_year(data: pd.DataFrame, year_column: str, ior_column: str) -> pd.DataFrame:
    yearly = (
        data.groupby(year_column, sort=True)[ior_column]
        .agg(count="count", mean="mean", std="std")
        .reset_index()
    )
    yearly["se"] = yearly["std"] / np.sqrt(yearly["count"])
    return yearly.rename(columns={year_column: "year"})


def plot_trend(yearly: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    years = yearly["year"].to_numpy(dtype=float)
    means = yearly["mean"].to_numpy(dtype=float)
    standard_errors = yearly["se"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(years, means, color="#1f77b4", linewidth=2.0)
    ax.scatter(years, means, color="#1f77b4", s=36, zorder=3)

    lower = means - standard_errors
    upper = means + standard_errors
    ax.fill_between(
        years,
        lower,
        upper,
        color="#1f77b4",
        alpha=0.16,
        linewidth=0,
    )

    ax.set_title("Mean Institutional Ownership Ratio by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean IOR")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.03, y=0.12)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def format_number(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}f}"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} does not exist.")

    frame = pd.read_parquet(args.input)
    if args.ior_column not in frame.columns:
        raise ValueError(f"{args.ior_column} is not a column in {args.input}.")

    year_column = choose_year_column(frame, args.year_column)
    data = frame.loc[:, [year_column, args.ior_column]].copy()
    data[year_column] = pd.to_numeric(data[year_column], errors="coerce")
    data[args.ior_column] = pd.to_numeric(data[args.ior_column], errors="coerce")
    data = data.dropna(subset=[year_column, args.ior_column])
    data[year_column] = data[year_column].astype(int)

    if data.empty:
        raise ValueError(f"No nonmissing {args.ior_column} observations found in {args.input}.")
    if args.clip_ior:
        data[args.ior_column] = data[args.ior_column].clip(0.0, 1.0)

    overall_mean = data[args.ior_column].mean()
    overall_std = data[args.ior_column].std(ddof=1)
    overall_se = standard_error(data[args.ior_column])
    yearly = summarize_by_year(data, year_column, args.ior_column)
    plot_trend(yearly, args.output)

    print("Institutional ownership ratio descriptives")
    print(f"Input file: {args.input}")
    print(f"IOR column: {args.ior_column}")
    print(f"Year column: {year_column}")
    print(f"Observations: {len(data):,}")
    print(f"Years: {int(yearly['year'].min())}-{int(yearly['year'].max())}")
    print(f"Mean IOR: {format_number(overall_mean)}")
    print(f"Standard error: {format_number(overall_se)}")
    print(f"Standard deviation: {format_number(overall_std)}")
    print()
    print("Yearly mean IOR")
    print(
        yearly.loc[:, ["year", "count", "mean", "se"]].to_string(
            index=False,
            formatters={
                "mean": lambda value: format_number(value),
                "se": lambda value: format_number(value),
            },
        )
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
