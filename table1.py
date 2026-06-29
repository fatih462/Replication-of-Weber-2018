#!/usr/bin/env python3
"""Recreate Weber (2018) Table 1 from local cleaned files.

The script computes annual cross-sectional means, standard deviations, and
correlations, then reports their time-series averages. This follows Weber
(2018), Table 1. IOR is merged from the augmented duration file when available.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = [
    ("Dur", "dur"),
    ("BM", "bm"),
    ("IOR", "ior"),
    ("PR", "payout_ratio"),
    ("ROE", "roe"),
    ("Sales_g", "sales_growth"),
    ("ME", "me_dec_millions"),
    ("Age", "age"),
]

TABLE1_NOTES = [
    (
        "This table reports time-series averages of annual cross-sectional "
        "means and standard deviations for firm characteristics and return "
        "predictors in Panel A and contemporaneous correlations of these "
        "variables in Panel B."
    ),
    (
        "Dur is cash-flow duration; BM is the book-to-market ratio; IOR is "
        "the fraction of shares institutions hold; PR is net payout over net "
        "income; ROE is return on equity; Sales_g is sales growth; ME is "
        "market capitalization in millions; and Age is the number of years a "
        "firm has been on Compustat."
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save Weber Table 1 summary statistics and correlations."
    )
    parser.add_argument(
        "--duration-parquet",
        type=Path,
        default=Path("data/cash_flow_duration.parquet"),
        help="Parquet file produced by build_cash_flow_duration.py.",
    )
    parser.add_argument(
        "--annual-parquet",
        type=Path,
        default=Path("data/compustat_annual_clean.parquet"),
        help="Cleaned annual Compustat parquet with payout ratio.",
    )
    parser.add_argument(
        "--ior-parquet",
        type=Path,
        default=Path("data/cash_flow_duration_with_ior.parquet"),
        help=(
            "Augmented duration parquet containing institutional ownership. "
            "If missing, IOR is left blank."
        ),
    )
    parser.add_argument(
        "--no-size-filter",
        action="store_true",
        help="Do not apply Weber's annual 20th size percentile filter.",
    )
    parser.add_argument(
        "--no-winsorize",
        action="store_true",
        help="Do not winsorize Table 1 variables at the 1st and 99th percentiles.",
    )
    parser.add_argument(
        "--winsorize-scope",
        choices=["global", "year"],
        default="global",
        help="Winsorize over the full sample or separately by formation year.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Directory for formatted table outputs.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1981,
        help="Fallback first annual cross-section if no IOR years are available.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2013,
        help=(
            "Fallback last annual cross-section if no IOR years are available. "
            "The default corresponds to the paper's July 1981-June 2014 return window."
        ),
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write TeX and TXT only; do not try to compile a PDF with pdflatex.",
    )
    return parser.parse_args()


def load_ior_data(ior_path: Path) -> pd.DataFrame:
    if not ior_path.exists():
        return pd.DataFrame(columns=["gvkey", "datadate", "formation_year", "ior"])

    ior = pd.read_parquet(ior_path)
    required = {"gvkey", "datadate", "formation_year", "ior"}
    missing = sorted(required - set(ior.columns))
    if missing:
        raise ValueError(f"{ior_path} is missing required columns: {missing}")

    ior = ior.loc[:, ["gvkey", "datadate", "formation_year", "ior"]].copy()
    ior["gvkey"] = ior["gvkey"].astype("string").str.strip()
    ior["datadate"] = pd.to_datetime(ior["datadate"], errors="coerce")
    ior["formation_year"] = pd.to_numeric(ior["formation_year"], errors="coerce")
    ior["ior"] = pd.to_numeric(ior["ior"], errors="coerce")
    ior = ior.sort_values(["gvkey", "formation_year", "datadate"]).drop_duplicates(
        ["gvkey", "datadate", "formation_year"],
        keep="last",
    )
    return ior


def load_data(duration_path: Path, annual_path: Path, ior_path: Path) -> pd.DataFrame:
    duration = pd.read_parquet(duration_path)
    annual = pd.read_parquet(
        annual_path,
        columns=["gvkey", "datadate", "payout_ratio"],
    )

    required_duration = {
        "gvkey",
        "datadate",
        "formation_year",
        "dur",
        "bm",
        "roe",
        "sales_growth",
        "me_dec_millions",
        "age",
    }
    missing = sorted(required_duration - set(duration.columns))
    if missing:
        raise ValueError(f"{duration_path} is missing required columns: {missing}")

    duration = duration.copy()
    duration["gvkey"] = duration["gvkey"].astype("string").str.strip()
    duration["datadate"] = pd.to_datetime(duration["datadate"], errors="coerce")
    duration["formation_year"] = pd.to_numeric(duration["formation_year"], errors="coerce")

    annual = annual.copy()
    annual["gvkey"] = annual["gvkey"].astype("string").str.strip()
    annual["datadate"] = pd.to_datetime(annual["datadate"], errors="coerce")

    data = duration.merge(annual, on=["gvkey", "datadate"], how="left")
    data = data.drop(columns=["ior"], errors="ignore")
    ior = load_ior_data(ior_path)
    if not ior.empty:
        data = data.merge(
            ior,
            on=["gvkey", "datadate", "formation_year"],
            how="left",
            validate="one_to_one",
        )
    else:
        data["ior"] = np.nan

    for _, column in VARIABLES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def apply_size_filter(data: pd.DataFrame) -> pd.DataFrame:
    cutoff = data.groupby("formation_year")["me_dec_millions"].transform(
        lambda series: series.quantile(0.20)
    )
    return data.loc[data["me_dec_millions"].ge(cutoff)].copy()


def winsorize_global(series: pd.Series) -> pd.Series:
    if series.notna().sum() < 20:
        return series
    lower = series.quantile(0.01)
    upper = series.quantile(0.99)
    return series.clip(lower, upper)


def winsorize_by_year(data: pd.DataFrame, column: str) -> pd.Series:
    def clip_one_year(series: pd.Series) -> pd.Series:
        if series.notna().sum() < 20:
            return series
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        return series.clip(lower, upper)

    return data.groupby("formation_year", group_keys=False)[column].transform(clip_one_year)


def apply_winsorization(data: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = data.copy()
    for _, column in VARIABLES:
        if out[column].notna().sum() == 0:
            continue
        if scope == "year":
            out[column] = winsorize_by_year(out, column)
        else:
            out[column] = winsorize_global(out[column])
    return out


def panel_a(data: pd.DataFrame) -> pd.DataFrame:
    columns = [column for _, column in VARIABLES]
    grouped = data.groupby("formation_year")[columns]
    annual_means = grouped.mean()
    annual_stds = grouped.std()
    return pd.DataFrame(
        [
            [annual_means[column].dropna().mean() for _, column in VARIABLES],
            [annual_stds[column].dropna().mean() for _, column in VARIABLES],
        ],
        index=["Mean", "Std"],
        columns=[name for name, _ in VARIABLES],
    )


def panel_b(data: pd.DataFrame) -> pd.DataFrame:
    names = [name for name, _ in VARIABLES]
    columns = [column for _, column in VARIABLES]
    yearly_correlations = []
    for _, group in data.groupby("formation_year", sort=True):
        corr = group.loc[:, columns].corr(min_periods=3)
        corr.index = names
        corr.columns = names
        yearly_correlations.append(corr)

    if not yearly_correlations:
        return pd.DataFrame(np.nan, index=names, columns=names)

    averaged = pd.concat(yearly_correlations, keys=range(len(yearly_correlations))).groupby(level=1).mean()
    averaged = averaged.reindex(index=names, columns=names)
    for row_index, row_name in enumerate(names):
        for column_index, column_name in enumerate(names):
            if column_index <= row_index:
                averaged.loc[row_name, column_name] = np.nan
    return averaged


def panel_b_sample(data: pd.DataFrame, fallback_start_year: int, fallback_end_year: int) -> pd.DataFrame:
    ior_years = data.loc[data["ior"].notna(), "formation_year"].dropna()
    if not ior_years.empty:
        years = set(ior_years.astype(int))
        return data.loc[data["formation_year"].astype("Int64").isin(years)].copy()
    return data.loc[data["formation_year"].between(fallback_start_year, fallback_end_year)].copy()


def format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return f"{value:.{decimals}f}"


def format_panel_a_plain(table: pd.DataFrame) -> str:
    formatted = pd.DataFrame(index=table.index, columns=table.columns, dtype="object")
    for row in table.index:
        formatted.loc[row] = [format_number(value) for value in table.loc[row]]
    return formatted.to_string()


def format_panel_b_plain(table: pd.DataFrame) -> str:
    formatted = pd.DataFrame(index=table.index, columns=table.columns, dtype="object")
    for row in table.index:
        formatted.loc[row] = [format_number(value) for value in table.loc[row]]
    return formatted.to_string()


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


def latex_label(label: str) -> str:
    label_map = {
        "Sales_g": r"Sales\_g",
    }
    return label_map.get(label, latex_escape(label))


def panel_a_to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\addlinespace",
        r"\multicolumn{9}{l}{\textit{Panel A: means and standard deviations}}\\",
    ]
    for row in table.index:
        values = [format_number(table.loc[row, column]) for column in table.columns]
        lines.append(latex_escape(row) + " & " + " & ".join(values) + r" \\")
    return "\n".join(lines)


def panel_b_to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\addlinespace",
        r"\multicolumn{9}{l}{\textit{Panel B: contemporaneous correlations}}\\",
    ]
    for row in table.index[:-1]:
        values = [format_number(table.loc[row, column]) for column in table.columns]
        lines.append(latex_label(row) + " & " + " & ".join(values) + r" \\")
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


def sample_note(data: pd.DataFrame, fallback_start_year: int, fallback_end_year: int) -> str:
    years = data["formation_year"].dropna().astype(int)
    if years.empty:
        return (
            "Panel B has no observations after filters. "
            f"Fallback annual cross-sections were {fallback_start_year}-{fallback_end_year}."
        )
    source = (
        "the years with nonmissing IOR"
        if data["ior"].notna().any()
        else f"fallback annual cross-sections {fallback_start_year}-{fallback_end_year}"
    )
    return (
        f"Panel B uses {source}: {int(years.min())}-{int(years.max())} "
        f"({years.nunique()} years, {len(data):,} firm-year observations after filters)."
    )


def panel_a_note(data: pd.DataFrame) -> str:
    years = data["formation_year"].dropna().astype(int)
    if years.empty:
        return "Panel A has no available observations after filters."
    return (
        "Panel A uses all available annual cross-sections after filters "
        f"({int(years.min())}-{int(years.max())}); each variable is averaged over "
        "the years in which that variable is nonmissing."
    )


def ior_note(data: pd.DataFrame, ior_path: Path) -> str | None:
    if data["ior"].notna().any():
        years = data.loc[data["ior"].notna(), "formation_year"].dropna().astype(int)
        return (
            f"IOR is merged from {ior_path} and populated for {years.nunique()} local sample years "
            f"({int(years.min())}-{int(years.max())})."
        )
    return (
        f"IOR is blank because {ior_path} has no observations matching the relevant sample."
    )


def build_notes(
    args: argparse.Namespace,
    panel_a_data: pd.DataFrame,
    panel_b_data: pd.DataFrame,
) -> list[str]:
    notes = [
        *TABLE1_NOTES,
        panel_a_note(panel_a_data),
        sample_note(panel_b_data, args.start_year, args.end_year),
    ]
    note = ior_note(panel_a_data, args.ior_parquet)
    if note is not None:
        notes.append(note)
    if not args.no_size_filter:
        notes.append("Applied the annual 20th percentile ME filter, matching Weber Table 1.")
    if not args.no_winsorize:
        notes.append(f"Winsorized Table 1 variables at 1%/99% using {args.winsorize_scope} scope.")
    return notes


def write_outputs(
    output_dir: Path,
    panel_a_df: pd.DataFrame,
    panel_b_df: pd.DataFrame,
    notes: list[str],
    *,
    build_pdf: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [name for name, _ in VARIABLES]
    table_body = "\n\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{6pt}",
            r"\caption{Summary statistics and correlations for firm characteristics and return predictors}",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            " & " + " & ".join(latex_label(column) for column in columns) + r" \\",
            r"\midrule",
            panel_a_to_latex(panel_a_df),
            panel_b_to_latex(panel_b_df),
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.95\linewidth}",
            r"\footnotesize Notes: " + " ".join(latex_escape(note) for note in notes),
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    latex_document = "\n\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[landscape,margin=0.5in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{caption}",
            r"\captionsetup{font=small}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\setcounter{table}{0}",
            table_body,
            r"\end{document}",
        ]
    )

    tex_path = output_dir / "table1_summary_statistics.tex"
    tex_path.write_text(latex_document + "\n", encoding="utf-8")

    plain = "\n\n".join(
        [
            "Table 1: Summary statistics and correlations for firm characteristics and return predictors",
            "Panel A: means and standard deviations",
            format_panel_a_plain(panel_a_df),
            "Panel B: contemporaneous correlations",
            format_panel_b_plain(panel_b_df),
            "Notes:",
            *notes,
        ]
    )
    (output_dir / "table1_summary_statistics.txt").write_text(
        plain + "\n",
        encoding="utf-8",
    )
    if build_pdf:
        compile_pdf(tex_path, output_dir)


def main() -> None:
    args = parse_args()
    data = load_data(args.duration_parquet, args.annual_parquet, args.ior_parquet)

    panel_a_data = data.copy()
    panel_b_data = panel_b_sample(data, args.start_year, args.end_year)
    if not args.no_size_filter:
        panel_a_data = apply_size_filter(panel_a_data)
        panel_b_data = apply_size_filter(panel_b_data)
    if not args.no_winsorize:
        panel_a_data = apply_winsorization(panel_a_data, args.winsorize_scope)
        panel_b_data = apply_winsorization(panel_b_data, args.winsorize_scope)

    panel_a_df = panel_a(panel_a_data)
    panel_b_df = panel_b(panel_b_data)
    notes = build_notes(args, panel_a_data, panel_b_data)

    write_outputs(
        args.output_dir,
        panel_a_df,
        panel_b_df,
        notes,
        build_pdf=not args.no_pdf,
    )

    print("Table 1: Summary statistics and correlations for firm characteristics and return predictors\n")
    print("Panel A: means and standard deviations")
    print(format_panel_a_plain(panel_a_df))
    print("\nPanel B: contemporaneous correlations")
    print(format_panel_b_plain(panel_b_df))
    print("\nNotes:")
    for note in notes:
        print(f"- {note}")
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
