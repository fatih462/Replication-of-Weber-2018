#!/usr/bin/env python3
"""Recreate Weber (2018) Table 10 using local duration and SEC 13F data.

The original paper uses Thomson Reuters 13F data from the 1981-2013 sorting
years. The local SEC filings in this repository begin at report year 2014, so
the default output reproduces the Table 10 methodology over the available
post-2014 SEC sample.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from table_utils import yyyymm_range_label


DURATION_COLUMNS = [
    "gvkey",
    "conm",
    "datadate",
    "formation_year",
    "dur",
    "ior",
    "rior",
    "ior_size_filter_eligible",
]

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

COLUMN_LABELS = ["Low Dur", "D2", "D3", "D4", "High Dur", "D1-D5"]
ROW_LABELS = {
    1: "Low RIOR",
    2: "RIOR2",
    3: "RIOR3",
    4: "RIOR4",
    5: "High RIOR",
}
SPREAD_ROW_LABEL = "RIOR1-RIOR5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Weber Table 10 duration x residual institutional ownership returns."
    )
    parser.add_argument(
        "--duration-parquet",
        type=Path,
        default=Path("data/cash_flow_duration_with_ior.parquet"),
        help="Augmented duration parquet containing ior and rior columns.",
    )
    parser.add_argument(
        "--crsp-monthly",
        type=Path,
        default=Path("data/crsp_monthly_clean.parquet"),
        help="Cleaned CRSP monthly parquet.",
    )
    parser.add_argument(
        "--fama-french",
        type=Path,
        default=Path("data/raw/F-F_Research_Data_5_Factors_2x3.csv"),
        help="Fama-French monthly factors CSV containing RF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Directory for formatted table outputs.",
    )
    parser.add_argument(
        "--data-output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for assignment and monthly portfolio parquet outputs.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First June portfolio formation year. Defaults to first available RIOR year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last June portfolio formation year. Defaults to last full holding year.",
    )
    parser.add_argument(
        "--include-partial-years",
        action="store_true",
        help="Allow sorting years with fewer than 12 holding-period return months.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write TeX and TXT only; do not try to compile a PDF with pdflatex.",
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


def clean_gvkey(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def one_duration_row_per_year(duration: pd.DataFrame) -> pd.DataFrame:
    out = duration.copy()
    if "datadate" in out.columns:
        out["datadate"] = pd.to_datetime(out["datadate"], errors="coerce")
    else:
        out["datadate"] = pd.NaT
    out["_valid_dur"] = out["dur"].notna()
    out = out.sort_values(
        ["gvkey", "formation_year", "_valid_dur", "datadate"],
        ascending=[True, True, False, False],
    )
    out = out.drop_duplicates(["gvkey", "formation_year"])
    return out.drop(columns=["_valid_dur"])


def read_fama_french(path: Path) -> pd.DataFrame:
    path = resolve_input_path(path)
    rows: list[dict[str, float | int]] = []
    columns: list[str] | None = None

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if parts[0] == "" and "RF" in parts:
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
        raise ValueError(f"No monthly Fama-French rows found in {path}.")
    factors = pd.DataFrame(rows)
    require_columns(factors, {"YYYYMM", "RF"}, path)
    return factors.loc[:, ["YYYYMM", "RF"]].sort_values("YYYYMM")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duration = pd.read_parquet(args.duration_parquet, columns=DURATION_COLUMNS)
    monthly = pd.read_parquet(args.crsp_monthly, columns=MONTHLY_COLUMNS)
    factors = read_fama_french(args.fama_french)
    require_columns(duration, set(DURATION_COLUMNS), args.duration_parquet)
    require_columns(monthly, set(MONTHLY_COLUMNS), args.crsp_monthly)

    duration["gvkey"] = clean_gvkey(duration["gvkey"])
    for column in ["formation_year", "dur", "ior", "rior"]:
        duration[column] = pd.to_numeric(duration[column], errors="coerce")
    if "ior_size_filter_eligible" in duration.columns:
        duration["ior_size_filter_eligible"] = (
            duration["ior_size_filter_eligible"].astype("boolean").fillna(False).astype(bool)
        )

    monthly["gvkey"] = clean_gvkey(monthly["gvkey"])
    for column in ["PERMNO", "PERMCO", "YYYYMM", "year", "month", "ret", "me_firm_millions"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    return duration, monthly, factors


def full_holding_years(monthly: pd.DataFrame) -> set[int]:
    months = set(pd.to_numeric(monthly["YYYYMM"], errors="coerce").dropna().astype(int))
    years = set(pd.to_numeric(monthly["year"], errors="coerce").dropna().astype(int))
    full_years = set()
    for year in years:
        needed = {year * 100 + month for month in range(7, 13)}
        needed |= {(year + 1) * 100 + month for month in range(1, 7)}
        if needed.issubset(months):
            full_years.add(year)
    return full_years


def infer_sort_years(
    duration: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    start_year: int | None,
    end_year: int | None,
    include_partial_years: bool,
) -> list[int]:
    available = set(
        pd.to_numeric(
            duration.loc[duration["rior"].notna(), "formation_year"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    if not include_partial_years:
        available &= full_holding_years(monthly)

    if start_year is not None:
        available = {year for year in available if year >= start_year}
    if end_year is not None:
        available = {year for year in available if year <= end_year}

    if not available:
        raise ValueError(
            "No formation years have RIOR and matching monthly returns. "
            "Run build_residual_institutional_ownership.py first, or use "
            "--include-partial-years if you intentionally want partial years."
        )
    return sorted(available)


def assign_quintile(group: pd.DataFrame, column: str, out_column: str) -> pd.Series:
    if group[column].notna().sum() < 5:
        return pd.Series(pd.NA, index=group.index, dtype="Int64")
    ranks = group[column].rank(method="first")
    return pd.qcut(ranks, 5, labels=range(1, 6)).astype("Int64")


def assign_year_quintiles(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    out["duration_quintile"] = assign_quintile(out, "dur", "duration_quintile")
    out["rior_quintile"] = assign_quintile(out, "rior", "rior_quintile")
    return out


def build_assignments(
    duration: pd.DataFrame,
    monthly: pd.DataFrame,
    sort_years: list[int],
) -> pd.DataFrame:
    deduped_duration = one_duration_row_per_year(duration)
    sort_mask = (
        deduped_duration["formation_year"].isin(sort_years)
        & deduped_duration["dur"].notna()
        & deduped_duration["rior"].notna()
        & deduped_duration["ior_size_filter_eligible"].fillna(False)
    )
    sort_data = deduped_duration.loc[
        sort_mask,
        ["gvkey", "conm", "datadate", "formation_year", "dur", "ior", "rior"],
    ].copy()
    sort_data = sort_data.rename(columns={"formation_year": "sort_year"})

    june = monthly.loc[
        monthly["month"].eq(6)
        & monthly["year"].isin(sort_years)
        & monthly["PERMNO"].notna()
        & monthly["gvkey"].notna(),
        ["PERMNO", "PERMCO", "gvkey", "year", "YYYYMM", "me_firm_millions"],
    ].copy()
    june = june.rename(columns={"year": "sort_year", "YYYYMM": "june_yyyymm"})
    june = june.drop_duplicates(["sort_year", "PERMNO"])

    assignments = june.merge(sort_data, on=["gvkey", "sort_year"], how="inner")
    if assignments.empty:
        raise ValueError("No June stocks matched to duration and RIOR.")

    assignments = pd.concat(
        [assign_year_quintiles(group) for _, group in assignments.groupby("sort_year", sort=True)],
        ignore_index=True,
    )
    assignments = assignments.dropna(subset=["duration_quintile", "rior_quintile"])
    assignments["duration_quintile"] = assignments["duration_quintile"].astype(int)
    assignments["rior_quintile"] = assignments["rior_quintile"].astype(int)
    return assignments.sort_values(
        ["sort_year", "rior_quintile", "duration_quintile", "PERMNO"]
    ).reset_index(drop=True)


def add_holding_period_sort_year(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    out["sort_year"] = np.where(out["month"].ge(7), out["year"], out["year"] - 1)
    out["holding_month"] = np.where(out["month"].ge(7), out["month"] - 6, out["month"] + 6)
    return out


def build_monthly_portfolio_returns(
    assignments: pd.DataFrame,
    monthly: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    holding_returns = add_holding_period_sort_year(monthly)
    holding_returns = holding_returns.loc[
        holding_returns["holding_month"].between(1, 12)
        & holding_returns["ret"].notna(),
        ["PERMNO", "YYYYMM", "sort_year", "holding_month", "ret"],
    ].copy()

    stock_months = assignments.loc[
        :,
        [
            "sort_year",
            "PERMNO",
            "gvkey",
            "duration_quintile",
            "rior_quintile",
            "dur",
            "ior",
            "rior",
        ],
    ].merge(holding_returns, on=["PERMNO", "sort_year"], how="inner")

    portfolio = (
        stock_months.groupby(
            ["sort_year", "YYYYMM", "holding_month", "rior_quintile", "duration_quintile"],
            as_index=False,
        )
        .agg(
            ew_ret=("ret", "mean"),
            n_stocks=("ret", "count"),
            median_dur=("dur", "median"),
            median_rior=("rior", "median"),
        )
        .sort_values(["sort_year", "holding_month", "rior_quintile", "duration_quintile"])
    )
    portfolio = portfolio.merge(factors, on="YYYYMM", how="left")
    if portfolio["RF"].isna().any():
        missing = sorted(portfolio.loc[portfolio["RF"].isna(), "YYYYMM"].unique())
        raise ValueError(f"Missing Fama-French RF values for months: {missing[:10]}")
    portfolio["excess_ret"] = portfolio["ew_ret"] - portfolio["RF"]
    return portfolio


def pivot_series(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    return portfolio_returns.pivot_table(
        index="YYYYMM",
        columns=["rior_quintile", "duration_quintile"],
        values="excess_ret",
        aggfunc="mean",
    ).sort_index()


def safe_column(pivot: pd.DataFrame, rior_quintile: int, duration_quintile: int) -> pd.Series:
    key = (rior_quintile, duration_quintile)
    if key in pivot.columns:
        return pivot[key]
    return pd.Series(np.nan, index=pivot.index, dtype="float64")


def table_series(portfolio_returns: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    pivot = pivot_series(portfolio_returns)
    out: dict[tuple[str, str], pd.Series] = {}

    for rior_quintile, row_label in ROW_LABELS.items():
        low_duration = safe_column(pivot, rior_quintile, 1)
        high_duration = safe_column(pivot, rior_quintile, 5)
        for duration_quintile, column_label in enumerate(COLUMN_LABELS[:5], start=1):
            out[(row_label, column_label)] = safe_column(
                pivot, rior_quintile, duration_quintile
            )
        out[(row_label, "D1-D5")] = low_duration - high_duration

    for duration_quintile, column_label in enumerate(COLUMN_LABELS[:5], start=1):
        out[(SPREAD_ROW_LABEL, column_label)] = safe_column(pivot, 1, duration_quintile) - safe_column(
            pivot, 5, duration_quintile
        )
    out[(SPREAD_ROW_LABEL, "D1-D5")] = (
        out[("Low RIOR", "D1-D5")] - out[("High RIOR", "D1-D5")]
    )
    return out


def mean_and_se(series: pd.Series) -> tuple[float, float, int]:
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan, np.nan, int(len(clean))
    return clean.mean(), clean.std(ddof=1) / math.sqrt(len(clean)), int(len(clean))


def build_table(portfolio_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_map = table_series(portfolio_returns)
    rows = [*ROW_LABELS.values(), SPREAD_ROW_LABEL]
    means = pd.DataFrame(index=rows, columns=COLUMN_LABELS, dtype="float64")
    ses = pd.DataFrame(index=rows, columns=COLUMN_LABELS, dtype="float64")
    counts = pd.DataFrame(index=rows, columns=COLUMN_LABELS, dtype="float64")

    for row in rows:
        for column in COLUMN_LABELS:
            mean, se, count = mean_and_se(series_map[(row, column)])
            means.loc[row, column] = mean
            ses.loc[row, column] = se
            counts.loc[row, column] = count
    return means, ses, counts


def percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * value:.2f}"


def parenthesized_percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"({100.0 * value:.2f})"


def format_plain(means: pd.DataFrame, ses: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    index: list[str] = []
    for row in means.index:
        rows.append([percent_or_blank(value) for value in means.loc[row]])
        index.append(row)
        rows.append([parenthesized_percent_or_blank(value) for value in ses.loc[row]])
        index.append("")
    display = pd.DataFrame(rows, index=index, columns=means.columns)
    return display.to_string()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def table_to_latex(means: pd.DataFrame, ses: pd.DataFrame) -> str:
    lines = []
    for row in means.index:
        mean_values = [percent_or_blank(value) for value in means.loc[row]]
        se_values = [parenthesized_percent_or_blank(value) for value in ses.loc[row]]
        lines.append(latex_escape(row) + " & " + " & ".join(mean_values) + r" \\")
        lines.append(" & " + " & ".join(se_values) + r" \\")
    return "\n".join(lines)


def compile_pdf(tex_path: Path, output_dir: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("Note: pdflatex was not found, so no PDF was created.")
        return

    subprocess.run(
        [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output_dir),
            str(tex_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    stem = tex_path.stem
    for suffix in [".aux", ".log", ".out"]:
        aux_path = output_dir / f"{stem}{suffix}"
        if aux_path.exists():
            aux_path.unlink()


def write_outputs(
    output_dir: Path,
    means: pd.DataFrame,
    ses: pd.DataFrame,
    notes: list[str],
    *,
    build_pdf: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table_body = "\n\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{5pt}",
            r"\caption{Mean excess returns of 25 portfolios sorted on duration and residual institutional ownership}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            " & " + " & ".join(COLUMN_LABELS) + r" \\",
            r"\midrule",
            table_to_latex(means, ses),
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.92\linewidth}",
            r"\footnotesize Notes: " + " ".join(latex_escape(note) for note in notes),
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    latex_document = "\n\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[landscape,margin=0.6in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{caption}",
            r"\captionsetup{font=small}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\setcounter{table}{9}",
            table_body,
            r"\end{document}",
        ]
    )

    tex_path = output_dir / "table10_duration_rior.tex"
    tex_path.write_text(latex_document + "\n", encoding="utf-8")

    plain = "\n\n".join(
        [
            "Table 10: Mean excess returns of 25 portfolios sorted on duration and residual institutional ownership",
            format_plain(means, ses),
            "Notes:",
            *notes,
        ]
    )
    (output_dir / "table10_duration_rior.txt").write_text(plain + "\n", encoding="utf-8")
    if build_pdf:
        compile_pdf(tex_path, output_dir)


def write_data_outputs(
    output_dir: Path,
    assignments: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    means: pd.DataFrame,
    ses: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output_dir / "table10_duration_rior_assignments.parquet", index=False)
    portfolio_returns.to_parquet(
        output_dir / "table10_duration_rior_monthly_returns.parquet",
        index=False,
    )

    tidy_rows = []
    for row in means.index:
        for column in means.columns:
            tidy_rows.append(
                {
                    "rior_portfolio": row,
                    "duration_portfolio": column,
                    "mean": means.loc[row, column],
                    "se": ses.loc[row, column],
                    "months": counts.loc[row, column],
                }
            )
    pd.DataFrame(tidy_rows).to_parquet(
        output_dir / "table10_duration_rior_summary.parquet",
        index=False,
    )


def coverage_notes(
    sort_years: list[int],
    assignments: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
) -> list[str]:
    months = portfolio_returns["YYYYMM"]
    avg_stocks = (
        portfolio_returns.groupby("YYYYMM")["n_stocks"].sum().mean()
        if len(portfolio_returns)
        else np.nan
    )
    return [
        (
            f"Sort years are {min(sort_years)}-{max(sort_years)}; monthly returns "
            f"cover {yyyymm_range_label(months.min(), months.max())} "
            f"({months.nunique()} months)."
        ),
        (
            f"The sample has {len(assignments):,} stock-year assignments and about "
            f"{avg_stocks:,.0f} stock-month observations per return month across cells."
        ),
        (
            "Returns and standard errors are monthly percentages. Standard errors are "
            "intercept-only OLS standard errors."
        ),
        (
            "RIOR is computed from SEC 13F holdings with Weber Eq. (7): clipped logit "
            "IOR residualized on log market equity and squared log market equity by year."
        ),
        (
            "The local SEC 13F data cover report years 2014 onward, so this is a "
            "post-2014 methodology reproduction rather than Weber's 1981-2013 TR-13F sample."
        ),
    ]


def main() -> None:
    args = parse_args()
    duration, monthly, factors = load_inputs(args)
    sort_years = infer_sort_years(
        duration,
        monthly,
        start_year=args.start_year,
        end_year=args.end_year,
        include_partial_years=args.include_partial_years,
    )
    assignments = build_assignments(duration, monthly, sort_years)
    portfolio_returns = build_monthly_portfolio_returns(assignments, monthly, factors)
    means, ses, counts = build_table(portfolio_returns)

    notes = coverage_notes(sort_years, assignments, portfolio_returns)
    write_data_outputs(args.data_output_dir, assignments, portfolio_returns, means, ses, counts)
    write_outputs(args.output_dir, means, ses, notes, build_pdf=not args.no_pdf)

    print("Table 10: Mean excess returns sorted on duration and residual institutional ownership\n")
    print(format_plain(means, ses))
    print("\nNotes:")
    for note in notes:
        print(f"- {note}")
    print(f"\nWrote outputs to {args.output_dir.resolve()}")
    print(f"Wrote intermediate data to {args.data_output_dir.resolve()}")


if __name__ == "__main__":
    main()
