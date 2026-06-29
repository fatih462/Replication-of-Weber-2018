#!/usr/bin/env python3
"""Lightweight subsample checks for the short-duration premium.

This script is deliberately diagnostic rather than paper-table machinery. It
uses the existing equal-weight duration decile return file and summarizes the
low-minus-high duration spread in periods that are useful for explaining why a
post-2014 Table 10 replication may look different from Weber's 1981-2013
sample.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from table_utils import yyyymm_range_label


OUTPUT_COLUMNS = [
    "Sample",
    "Condition",
    "Months",
    "Start",
    "End",
    "Low Dur",
    "High Dur",
    "D1-D10",
    "SE",
    "t-stat",
    "Avg RF",
    "Avg HML",
    "Avg Mkt-RF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze duration-premium subsamples using existing decile returns."
    )
    parser.add_argument(
        "--monthly-returns",
        type=Path,
        default=Path("data/duration_decile_monthly_returns.parquet"),
        help="Equal-weight duration decile monthly returns parquet.",
    )
    parser.add_argument(
        "--fama-french",
        type=Path,
        default=Path("data/raw/F-F_Research_Data_5_Factors_2x3.csv"),
        help="Fama-French five-factor monthly CSV containing RF, HML, and Mkt-RF.",
    )
    parser.add_argument(
        "--table10-monthly-returns",
        type=Path,
        default=Path("data/table10_duration_rior_monthly_returns.parquet"),
        help="Optional Table 10 duration x RIOR monthly returns parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Directory for human-readable outputs.",
    )
    parser.add_argument(
        "--data-output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for machine-readable outputs.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path

    data_path = Path("data/raw") / path.name
    if not path.is_absolute() and data_path.exists():
        return data_path

    raise FileNotFoundError(
        f"Could not find {path} or {data_path}. "
        "Use the matching command-line option to point to the file."
    )


def read_factor_csv(path: Path, wanted_columns: list[str]) -> pd.DataFrame:
    path = resolve_input_path(path)
    rows: list[dict[str, float | int]] = []
    columns: list[str] | None = None

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if parts[0] == "" and any(column in parts for column in wanted_columns):
                columns = ["YYYYMM", *parts[1:]]
                continue
            if columns is None:
                continue
            date = parts[0]
            if not (date.isdigit() and len(date) == 6):
                continue

            values: list[float] = []
            for value in parts[1 : len(columns)]:
                parsed = float(value)
                values.append(np.nan if parsed in {-99.99, -999.0} else parsed / 100.0)
            rows.append(dict(zip(columns, [int(date), *values])))

    if not rows:
        raise ValueError(f"No monthly factor rows found in {path}.")
    factors = pd.DataFrame(rows)
    require_columns(factors, {"YYYYMM", *wanted_columns}, path)
    return factors.loc[:, ["YYYYMM", *wanted_columns]].sort_values("YYYYMM")


def duration_decile_returns(path: Path) -> pd.DataFrame:
    monthly = pd.read_parquet(path)
    require_columns(monthly, {"YYYYMM", "duration_decile", "ew_ret"}, path)
    portfolio = monthly.pivot_table(
        index="YYYYMM",
        columns="duration_decile",
        values="ew_ret",
        aggfunc="mean",
    )
    portfolio = portfolio.reindex(columns=range(1, 11)).sort_index()
    portfolio["D1-D10"] = portfolio[1] - portfolio[10]
    return portfolio.reset_index()


def build_dataset(monthly_returns: Path, fama_french: Path) -> pd.DataFrame:
    returns = duration_decile_returns(monthly_returns)
    factors = read_factor_csv(fama_french, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    data = returns.merge(factors, on="YYYYMM", how="inner").sort_values("YYYYMM")
    if data.empty:
        raise ValueError("No overlapping duration return and factor months.")
    data["low_excess"] = data[1] - data["RF"]
    data["high_excess"] = data[10] - data["RF"]
    data["spread"] = data["D1-D10"]
    return data.reset_index(drop=True)


def mean_and_se(series: pd.Series) -> tuple[float, float, float]:
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan, np.nan, np.nan
    mean = float(clean.mean())
    se = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    t_stat = np.nan if se == 0 else mean / se
    return mean, se, t_stat


def yyyymm_label(value: int) -> str:
    value = int(value)
    return f"{value // 100}-{value % 100:02d}"


def summarize(
    data: pd.DataFrame,
    sample: str,
    condition: str,
    *,
    start_yyyymm: int | None = None,
    end_yyyymm: int | None = None,
    mask: pd.Series | None = None,
) -> dict[str, object]:
    panel = data.copy()
    if start_yyyymm is not None:
        panel = panel.loc[panel["YYYYMM"].ge(start_yyyymm)]
    if end_yyyymm is not None:
        panel = panel.loc[panel["YYYYMM"].le(end_yyyymm)]
    if mask is not None:
        panel = panel.loc[mask.reindex(panel.index).fillna(False)]

    if panel.empty:
        return {
            "Sample": sample,
            "Condition": condition,
            "Months": 0,
            "Start": "",
            "End": "",
            "Low Dur": np.nan,
            "High Dur": np.nan,
            "D1-D10": np.nan,
            "SE": np.nan,
            "t-stat": np.nan,
            "Avg RF": np.nan,
            "Avg HML": np.nan,
            "Avg Mkt-RF": np.nan,
        }

    spread_mean, spread_se, spread_t = mean_and_se(panel["spread"])
    return {
        "Sample": sample,
        "Condition": condition,
        "Months": int(panel["YYYYMM"].nunique()),
        "Start": yyyymm_label(int(panel["YYYYMM"].min())),
        "End": yyyymm_label(int(panel["YYYYMM"].max())),
        "Low Dur": float(panel["low_excess"].mean()),
        "High Dur": float(panel["high_excess"].mean()),
        "D1-D10": spread_mean,
        "SE": spread_se,
        "t-stat": spread_t,
        "Avg RF": float(panel["RF"].mean()),
        "Avg HML": float(panel["HML"].mean()),
        "Avg Mkt-RF": float(panel["Mkt-RF"].mean()),
    }


def build_named_periods(data: pd.DataFrame) -> pd.DataFrame:
    rows = [
        summarize(data, "Full duration-return sample", "Calendar", start_yyyymm=196407, end_yyyymm=202506),
        summarize(data, "Weber institutional-ownership era", "Calendar", start_yyyymm=198107, end_yyyymm=201406),
        summarize(data, "Dotcom run-up", "Calendar", start_yyyymm=199507, end_yyyymm=200003),
        summarize(data, "Dotcom bust", "Calendar", start_yyyymm=200004, end_yyyymm=200306),
        summarize(data, "Full dotcom cycle", "Calendar", start_yyyymm=199507, end_yyyymm=200306),
        summarize(data, "Post-GFC, pre-post-2014 sample", "Calendar", start_yyyymm=200907, end_yyyymm=201406),
        summarize(data, "Post-2014 replication window", "Calendar", start_yyyymm=201407, end_yyyymm=202506),
        summarize(data, "Post-2014 pre-Covid", "Calendar", start_yyyymm=201407, end_yyyymm=201912),
        summarize(data, "Covid/ZLB rebound", "Calendar", start_yyyymm=202001, end_yyyymm=202112),
        summarize(data, "Rate-hike and AI-boom years", "Calendar", start_yyyymm=202201, end_yyyymm=202506),
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def build_condition_periods(data: pd.DataFrame) -> pd.DataFrame:
    post_2014 = data["YYYYMM"].between(201407, 202506)
    full = pd.Series(True, index=data.index)

    low_rf_cutoff = data["RF"].quantile(1 / 3)
    high_rf_cutoff = data["RF"].quantile(2 / 3)

    rows = [
        summarize(
            data,
            "Full sample, lowest RF tercile",
            f"RF <= {100 * low_rf_cutoff:.2f}% per month",
            mask=full & data["RF"].le(low_rf_cutoff),
        ),
        summarize(
            data,
            "Full sample, highest RF tercile",
            f"RF >= {100 * high_rf_cutoff:.2f}% per month",
            mask=full & data["RF"].ge(high_rf_cutoff),
        ),
        summarize(
            data,
            "Full sample, growth beats value",
            "HML < 0",
            mask=full & data["HML"].lt(0),
        ),
        summarize(
            data,
            "Full sample, value beats growth",
            "HML >= 0",
            mask=full & data["HML"].ge(0),
        ),
        summarize(
            data,
            "Post-2014, growth beats value",
            "HML < 0",
            mask=post_2014 & data["HML"].lt(0),
        ),
        summarize(
            data,
            "Post-2014, value beats growth",
            "HML >= 0",
            mask=post_2014 & data["HML"].ge(0),
        ),
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def table10_dataset(path: Path, factors: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    monthly = pd.read_parquet(path)
    require_columns(
        monthly,
        {"YYYYMM", "rior_quintile", "duration_quintile", "excess_ret"},
        path,
    )
    pivot = monthly.pivot_table(
        index="YYYYMM",
        columns=["rior_quintile", "duration_quintile"],
        values="excess_ret",
        aggfunc="mean",
    ).sort_index()

    def col(rior_quintile: int, duration_quintile: int) -> pd.Series:
        key = (rior_quintile, duration_quintile)
        if key in pivot.columns:
            return pivot[key]
        return pd.Series(np.nan, index=pivot.index)

    out = pd.DataFrame(
        {
            "YYYYMM": pivot.index,
            "low_rior_d1d5": col(1, 1) - col(1, 5),
            "high_rior_d1d5": col(5, 1) - col(5, 5),
            "low_dur_rior1_rior5": col(1, 1) - col(5, 1),
            "high_dur_rior1_rior5": col(1, 5) - col(5, 5),
        }
    ).reset_index(drop=True)
    out["interaction"] = out["low_rior_d1d5"] - out["high_rior_d1d5"]
    out = out.merge(factors.loc[:, ["YYYYMM", "HML", "RF", "Mkt-RF"]], on="YYYYMM", how="left")
    return out.sort_values("YYYYMM").reset_index(drop=True)


def summarize_table10(
    data: pd.DataFrame,
    sample: str,
    condition: str,
    *,
    start_yyyymm: int | None = None,
    end_yyyymm: int | None = None,
    mask: pd.Series | None = None,
) -> dict[str, object]:
    panel = data.copy()
    if start_yyyymm is not None:
        panel = panel.loc[panel["YYYYMM"].ge(start_yyyymm)]
    if end_yyyymm is not None:
        panel = panel.loc[panel["YYYYMM"].le(end_yyyymm)]
    if mask is not None:
        panel = panel.loc[mask.reindex(panel.index).fillna(False)]

    if panel.empty:
        return {
            "Sample": sample,
            "Condition": condition,
            "Months": 0,
            "Start": "",
            "End": "",
            "Low RIOR D1-D5": np.nan,
            "High RIOR D1-D5": np.nan,
            "Interaction": np.nan,
            "Interaction SE": np.nan,
            "Interaction t-stat": np.nan,
            "High Dur RIOR1-RIOR5": np.nan,
            "Avg RF": np.nan,
            "Avg HML": np.nan,
        }

    interaction_mean, interaction_se, interaction_t = mean_and_se(panel["interaction"])
    return {
        "Sample": sample,
        "Condition": condition,
        "Months": int(panel["YYYYMM"].nunique()),
        "Start": yyyymm_label(int(panel["YYYYMM"].min())),
        "End": yyyymm_label(int(panel["YYYYMM"].max())),
        "Low RIOR D1-D5": float(panel["low_rior_d1d5"].mean()),
        "High RIOR D1-D5": float(panel["high_rior_d1d5"].mean()),
        "Interaction": interaction_mean,
        "Interaction SE": interaction_se,
        "Interaction t-stat": interaction_t,
        "High Dur RIOR1-RIOR5": float(panel["high_dur_rior1_rior5"].mean()),
        "Avg RF": float(panel["RF"].mean()),
        "Avg HML": float(panel["HML"].mean()),
    }


def build_table10_periods(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    hml_negative = data["HML"].lt(0)
    rows = [
        summarize_table10(data, "Post-2014 Table 10 window", "Calendar", start_yyyymm=201507, end_yyyymm=202506),
        summarize_table10(data, "Post-2014 pre-Covid", "Calendar", start_yyyymm=201507, end_yyyymm=201912),
        summarize_table10(data, "Covid/ZLB rebound", "Calendar", start_yyyymm=202001, end_yyyymm=202112),
        summarize_table10(data, "Rate-hike and AI-boom years", "Calendar", start_yyyymm=202201, end_yyyymm=202506),
        summarize_table10(data, "Post-2014 Table 10, growth beats value", "HML < 0", mask=hml_negative),
        summarize_table10(data, "Post-2014 Table 10, value beats growth", "HML >= 0", mask=~hml_negative),
    ]
    columns = [
        "Sample",
        "Condition",
        "Months",
        "Start",
        "End",
        "Low RIOR D1-D5",
        "High RIOR D1-D5",
        "Interaction",
        "Interaction SE",
        "Interaction t-stat",
        "High Dur RIOR1-RIOR5",
        "Avg RF",
        "Avg HML",
    ]
    return pd.DataFrame(rows, columns=columns)


def pct(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.2f}"


def number(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def format_plain(table: pd.DataFrame) -> str:
    formatted = table.copy()
    for column in ["Low Dur", "High Dur", "D1-D10", "SE", "Avg RF", "Avg HML", "Avg Mkt-RF"]:
        formatted[column] = formatted[column].map(pct)
    formatted["t-stat"] = formatted["t-stat"].map(number)
    return formatted.to_string(index=False)


def format_table10_plain(table: pd.DataFrame) -> str:
    if table.empty:
        return "No Table 10 monthly-return file was found."

    formatted = table.copy()
    pct_columns = [
        "Low RIOR D1-D5",
        "High RIOR D1-D5",
        "Interaction",
        "Interaction SE",
        "High Dur RIOR1-RIOR5",
        "Avg RF",
        "Avg HML",
    ]
    for column in pct_columns:
        formatted[column] = formatted[column].map(pct)
    formatted["Interaction t-stat"] = formatted["Interaction t-stat"].map(number)
    return formatted.to_string(index=False)


def write_outputs(
    output_dir: Path,
    data_output_dir: Path,
    table: pd.DataFrame,
    table10: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_output_dir / "duration_premium_subsample_analysis.parquet"
    table10_data_path = data_output_dir / "table10_rior_subsample_analysis.parquet"
    text_path = output_dir / "duration_premium_subsample_analysis.txt"
    csv_path = output_dir / "duration_premium_subsample_analysis.csv"
    table10_csv_path = output_dir / "table10_rior_subsample_analysis.csv"

    table.to_parquet(data_path, index=False)
    table.to_csv(csv_path, index=False)
    if not table10.empty:
        table10.to_parquet(table10_data_path, index=False)
        table10.to_csv(table10_csv_path, index=False)

    notes = [
        "Returns are monthly percentages in the displayed table.",
        "Low Dur and High Dur are excess returns of duration deciles 1 and 10.",
        "D1-D10 is the low-minus-high duration spread; RF cancels in the spread.",
        "Standard errors and t-statistics are simple intercept-only time-series summaries.",
        "The condition rows are descriptive cuts, not causal tests.",
    ]
    table10_notes = [
        "The Table 10 companion panel uses the local post-2014 RIOR portfolio returns only.",
        "Interaction is Low RIOR D1-D5 minus High RIOR D1-D5.",
        "High Dur RIOR1-RIOR5 is the RIOR spread inside the high-duration quintile.",
    ]
    text = "\n\n".join(
        [
            "Short-Duration Premium: Simple Subsample Checks",
            format_plain(table),
            "Table 10 RIOR Companion: Post-2014 Subsamples",
            format_table10_plain(table10),
            "Notes:",
            *notes,
            *table10_notes,
        ]
    )
    text_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    factors = read_factor_csv(args.fama_french, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    data = build_dataset(args.monthly_returns, args.fama_french)
    named = build_named_periods(data)
    conditions = build_condition_periods(data)
    table = pd.concat([named, conditions], ignore_index=True)
    table10 = build_table10_periods(table10_dataset(args.table10_monthly_returns, factors))

    write_outputs(args.output_dir, args.data_output_dir, table, table10)

    print("Short-duration premium subsample analysis\n")
    print(format_plain(table))
    print("\nTable 10 RIOR companion: post-2014 subsamples\n")
    print(format_table10_plain(table10))
    print("\nCoverage:")
    print(f"- {yyyymm_range_label(data['YYYYMM'].min(), data['YYYYMM'].max())}")
    print(f"\nWrote tables to {args.output_dir.resolve()}")
    print(f"Wrote data to {args.data_output_dir.resolve()}")


if __name__ == "__main__":
    main()
