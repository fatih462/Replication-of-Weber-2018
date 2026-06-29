#!/usr/bin/env python3
"""Recreate Weber (2018) Figure 1 from duration decile portfolio outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average annual returns against average median duration."
    )
    parser.add_argument(
        "--figure-data",
        type=Path,
        default=Path("data/figure1_duration_term_structure.parquet"),
        help="Parquet output from build_duration_portfolios.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/figure1_duration_term_structure.png"),
        help="Path for the saved Figure 1 image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.figure_data.exists():
        raise FileNotFoundError(
            f"{args.figure_data} does not exist. Run python3 build_duration_portfolios.py first."
        )

    figure_data = pd.read_parquet(args.figure_data)
    required = {"duration_decile", "avg_median_dur", "avg_annual_return"}
    missing = sorted(required - set(figure_data.columns))
    if missing:
        raise ValueError(f"{args.figure_data} is missing required columns: {missing}")

    figure_data = figure_data.sort_values("duration_decile")
    x = figure_data["avg_median_dur"].to_numpy()
    y = (figure_data["avg_annual_return"] * 100.0).to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, y, color="#1f77b4", linewidth=2.0)
    ax.scatter(x, y, color="#1f77b4", s=38, zorder=3)

    for _, row in figure_data.iterrows():
        ax.annotate(
            f"D{int(row['duration_decile'])}",
            (row["avg_median_dur"], row["avg_annual_return"] * 100.0),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )

    ax.set_title("Average Term Structure of Equity")
    ax.set_xlabel("Average median cash-flow duration")
    ax.set_ylabel("Average annual portfolio return (%)")
    ax.margins(x=0.05, y=0.12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Wrote {args.output}")
    print(
        figure_data.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )


if __name__ == "__main__":
    main()
