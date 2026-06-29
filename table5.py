#!/usr/bin/env python3
"""Recreate Weber (2018) Table 5 from local cleaned files.

The table reports monthly mean excess returns for ten equal-weight duration
decile portfolios in subsamples. Unlike the paper, this script extends the
subsample panels after June 2014 when the local cleaned CRSP and Compustat
files contain enough data.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from build_duration_portfolios import (
    DURATION_COLUMNS,
    MONTHLY_COLUMNS,
    build_assignments,
    build_monthly_returns,
    clean_gvkey,
    require_columns,
)
from table_utils import yyyymm_range_label, yyyymm_to_month_label


COLUMN_LABELS = [
    "Low Dur",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "High Dur",
    "D1-D10",
]

PAPER_PANELS = [
    ("Panel A: July 1963-June 1973", 196307, 197306),
    ("Panel B: July 1973-June 1983", 197307, 198306),
    ("Panel C: July 1983-June 1993", 198307, 199306),
    ("Panel D: July 1993-June 2003", 199307, 200306),
    ("Panel E: July 2003-June 2014", 200307, 201406),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save Weber Table 5 subsample mean excess returns."
    )
    parser.add_argument(
        "--duration-parquet",
        type=Path,
        default=Path("data/cash_flow_duration.parquet"),
        help="Cash-flow duration parquet produced by build_cash_flow_duration.py.",
    )
    parser.add_argument(
        "--monthly-parquet",
        type=Path,
        default=Path("data/crsp_monthly_clean.parquet"),
        help="Cleaned monthly CRSP parquet.",
    )
    parser.add_argument(
        "--fama-french",
        type=Path,
        default=Path("data/raw/F-F_Research_Data_5_Factors_2x3.csv"),
        help="Fama-French five-factor CSV containing RF.",
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
    parser.add_argument(
        "--post-start-yyyymm",
        type=int,
        default=201407,
        help="First month for the added post-paper panel.",
    )
    parser.add_argument(
        "--post-end-yyyymm",
        type=int,
        default=None,
        help=(
            "Last month for the added post-paper panel. Defaults to June after "
            "the selected end-year."
        ),
    )
    parser.add_argument(
        "--exclude-corona-start-yyyymm",
        type=int,
        default=202001,
        help="First month to exclude from the added non-corona panel.",
    )
    parser.add_argument(
        "--exclude-corona-end-yyyymm",
        type=int,
        default=202112,
        help="Last month to exclude from the added non-corona panel.",
    )
    parser.add_argument(
        "--yearly-start-yyyymm",
        type=int,
        default=None,
        help=(
            "First July-June yearly subsample to write. Defaults to the added "
            "post-paper start month."
        ),
    )
    parser.add_argument(
        "--yearly-end-yyyymm",
        type=int,
        default=None,
        help=(
            "Last July-June yearly subsample to write. Defaults to the added "
            "post-paper end month."
        ),
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write TeX and TXT only; do not try to compile a PDF with pdflatex.",
    )
    return parser.parse_args()


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


def load_inputs(duration_path: Path, monthly_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    duration = pd.read_parquet(duration_path, columns=DURATION_COLUMNS)
    monthly = pd.read_parquet(monthly_path, columns=MONTHLY_COLUMNS)
    require_columns(duration, set(DURATION_COLUMNS), duration_path)
    require_columns(monthly, set(MONTHLY_COLUMNS), monthly_path)

    duration["gvkey"] = clean_gvkey(duration["gvkey"])
    duration["formation_year"] = pd.to_numeric(duration["formation_year"], errors="coerce")
    duration["dur"] = pd.to_numeric(duration["dur"], errors="coerce")

    monthly["gvkey"] = clean_gvkey(monthly["gvkey"])
    for column in ["PERMNO", "PERMCO", "YYYYMM", "year", "month", "ret", "me_firm_millions"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

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


def yyyymm_to_label(yyyymm: int) -> str:
    return yyyymm_to_month_label(yyyymm)


def add_spread(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[11] = out[1] - out[10]
    return out


def portfolio_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    portfolio = monthly_returns.pivot_table(
        index="YYYYMM",
        columns="duration_decile",
        values="ew_ret",
        aggfunc="mean",
    )
    portfolio = portfolio.reindex(columns=range(1, 11)).sort_index()
    return add_spread(portfolio).reset_index()


def align_with_rf(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    data = returns.merge(factors.loc[:, ["YYYYMM", "RF"]], on="YYYYMM", how="inner")
    if data.empty:
        raise ValueError("No overlapping return and risk-free-rate months.")
    return data.sort_values("YYYYMM").reset_index(drop=True)


def mean_and_se(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan, np.nan
    return clean.mean(), clean.std(ddof=1) / math.sqrt(len(clean))


def summarize_panel(
    data: pd.DataFrame,
    start_yyyymm: int,
    end_yyyymm: int,
    *,
    exclude_start_yyyymm: int | None = None,
    exclude_end_yyyymm: int | None = None,
) -> pd.DataFrame:
    panel = data.loc[data["YYYYMM"].between(start_yyyymm, end_yyyymm)].copy()
    if exclude_start_yyyymm is not None and exclude_end_yyyymm is not None:
        exclude = panel["YYYYMM"].between(exclude_start_yyyymm, exclude_end_yyyymm)
        panel = panel.loc[~exclude].copy()
    if panel.empty:
        return pd.DataFrame(index=["Mean", "SE"], columns=COLUMN_LABELS, dtype=float)

    excess = panel.loc[:, range(1, 11)].sub(panel["RF"], axis=0)
    excess[11] = panel[1] - panel[10]

    means = []
    ses = []
    for column in range(1, 12):
        mean, se = mean_and_se(excess[column])
        means.append(mean)
        ses.append(se)
    return pd.DataFrame([means, ses], index=["Mean", "SE"], columns=COLUMN_LABELS)


def build_panels(
    data: pd.DataFrame,
    post_start_yyyymm: int,
    post_end_yyyymm: int,
    exclude_corona_start_yyyymm: int,
    exclude_corona_end_yyyymm: int,
) -> list[tuple[str, pd.DataFrame]]:
    panels = [(title, summarize_panel(data, start, end)) for title, start, end in PAPER_PANELS]
    full_post_title = f"Panel F: {yyyymm_to_label(post_start_yyyymm)}-{yyyymm_to_label(post_end_yyyymm)}"
    panels.append((full_post_title, summarize_panel(data, post_start_yyyymm, post_end_yyyymm)))
    corona_title = (
        f"Panel G: {yyyymm_to_label(post_start_yyyymm)}-{yyyymm_to_label(post_end_yyyymm)}, "
        f"excluding {yyyymm_to_label(exclude_corona_start_yyyymm)}-"
        f"{yyyymm_to_label(exclude_corona_end_yyyymm)}"
    )
    panels.append(
        (
            corona_title,
            summarize_panel(
                data,
                post_start_yyyymm,
                post_end_yyyymm,
                exclude_start_yyyymm=exclude_corona_start_yyyymm,
                exclude_end_yyyymm=exclude_corona_end_yyyymm,
            ),
        )
    )
    return panels


def yearly_windows(start_yyyymm: int, end_yyyymm: int) -> list[tuple[str, int, int]]:
    start_year = start_yyyymm // 100
    if start_yyyymm % 100 > 7:
        start_year += 1
    end_year = end_yyyymm // 100
    if end_yyyymm % 100 < 6:
        end_year -= 1

    windows = []
    for year in range(start_year, end_year):
        start = year * 100 + 7
        end = (year + 1) * 100 + 6
        if start < start_yyyymm or end > end_yyyymm:
            continue
        windows.append((f"{year}-{year + 1}", start, end))
    return windows


def build_yearly_table(data: pd.DataFrame, start_yyyymm: int, end_yyyymm: int) -> pd.DataFrame:
    rows = []
    for label, start, end in yearly_windows(start_yyyymm, end_yyyymm):
        panel_months = data.loc[data["YYYYMM"].between(start, end), "YYYYMM"]
        if panel_months.empty:
            continue
        panel = summarize_panel(data, start, end)
        row = {"Year": label, "Months": int(panel_months.nunique())}
        row.update({column: panel.loc["Mean", column] for column in COLUMN_LABELS})
        row["SE D1-D10"] = panel.loc["SE", "D1-D10"]
        rows.append(row)

    columns = ["Year", "Months", *COLUMN_LABELS, "SE D1-D10"]
    return pd.DataFrame(rows, columns=columns)


def percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * value:.2f}"


def parenthesized_percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"({100.0 * value:.2f})"


def format_plain(panel: pd.DataFrame) -> str:
    formatted = pd.DataFrame(index=panel.index, columns=panel.columns, dtype="object")
    for row in panel.index:
        formatter = parenthesized_percent_or_blank if row == "SE" else percent_or_blank
        formatted.loc[row] = [formatter(value) for value in panel.loc[row]]
    return formatted.to_string()


def format_yearly_plain(table: pd.DataFrame) -> str:
    formatted = table.copy()
    for column in COLUMN_LABELS + ["SE D1-D10"]:
        formatted[column] = formatted[column].map(percent_or_blank)
    return formatted.to_string(index=False)


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


def latex_value(row: str, value: float) -> str:
    formatter = parenthesized_percent_or_blank if row == "SE" else percent_or_blank
    return formatter(value)


def panels_to_latex(panels: list[tuple[str, pd.DataFrame]]) -> str:
    lines = []
    for title, panel in panels:
        lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{12}}{{l}}{{\textit{{{latex_escape(title)}}}}}\\")
        for row in panel.index:
            values = [latex_value(row, panel.loc[row, column]) for column in panel.columns]
            lines.append(row + " & " + " & ".join(values) + r" \\")
    return "\n".join(lines)


def yearly_table_to_latex(table: pd.DataFrame) -> str:
    lines = []
    for _, row in table.iterrows():
        values = [latex_escape(str(row["Year"])), str(int(row["Months"]))]
        values.extend(percent_or_blank(row[column]) for column in COLUMN_LABELS)
        values.append(percent_or_blank(row["SE D1-D10"]))
        lines.append(" & ".join(values) + r" \\")
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


def coverage_note(data: pd.DataFrame) -> str:
    months = data["YYYYMM"]
    return (
        "Computed portfolio/factor overlap covers "
        f"{yyyymm_range_label(months.min(), months.max())} ({months.nunique()} months)."
    )


def panel_coverage_notes(
    data: pd.DataFrame,
    post_start_yyyymm: int,
    post_end_yyyymm: int,
    exclude_corona_start_yyyymm: int,
    exclude_corona_end_yyyymm: int,
) -> list[str]:
    notes = []
    post_panels = [("Panel F", post_start_yyyymm, post_end_yyyymm)]
    for title, start, end in [*PAPER_PANELS, *post_panels]:
        months = data.loc[data["YYYYMM"].between(start, end), "YYYYMM"]
        if months.empty:
            notes.append(f"{title} has no overlapping months in the local data.")
            continue
        notes.append(
            f"{title} uses {yyyymm_range_label(months.min(), months.max())} "
            f"({months.nunique()} months)."
        )
    panel_i_months = data.loc[
        data["YYYYMM"].between(post_start_yyyymm, post_end_yyyymm)
        & ~data["YYYYMM"].between(exclude_corona_start_yyyymm, exclude_corona_end_yyyymm),
        "YYYYMM",
    ]
    if panel_i_months.empty:
        notes.append("Panel G has no overlapping months after excluding the corona window.")
    else:
        notes.append(
            f"Panel G uses {yyyymm_range_label(panel_i_months.min(), panel_i_months.max())} "
            f"excluding {yyyymm_range_label(exclude_corona_start_yyyymm, exclude_corona_end_yyyymm)} "
            f"({panel_i_months.nunique()} months)."
        )
    return notes


def write_outputs(
    output_dir: Path,
    panels: list[tuple[str, pd.DataFrame]],
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
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Mean excess returns of ten portfolios sorted on duration (subsamples)}",
            r"\begin{tabular}{lrrrrrrrrrrr}",
            r"\toprule",
            " & " + " & ".join(COLUMN_LABELS) + r" \\",
            r"\midrule",
            panels_to_latex(panels),
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
            r"\setcounter{table}{4}",
            table_body,
            r"\end{document}",
        ]
    )

    tex_path = output_dir / "table5_duration_subsamples.tex"
    tex_path.write_text(latex_document + "\n", encoding="utf-8")

    plain_parts = ["Table 5: Mean excess returns of ten portfolios sorted on duration (subsamples)"]
    for title, panel in panels:
        plain_parts.extend([title, format_plain(panel)])
    plain_parts.extend(["Notes:", *notes])
    (output_dir / "table5_duration_subsamples.txt").write_text(
        "\n\n".join(plain_parts) + "\n",
        encoding="utf-8",
    )

    if build_pdf:
        compile_pdf(tex_path, output_dir)


def write_yearly_outputs(
    output_dir: Path,
    yearly_table: pd.DataFrame,
    notes: list[str],
    *,
    build_pdf: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table_body = "\n\n".join(
        [
            r"\begin{center}",
            r"\tiny",
            r"\setlength{\tabcolsep}{2pt}",
            r"\begin{longtable}{lrrrrrrrrrrrrr}",
            r"\caption{Yearly mean excess returns of ten portfolios sorted on duration}\\",
            r"\toprule",
            "Year & Months & " + " & ".join(COLUMN_LABELS) + r" & SE D1-D10 \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            "Year & Months & " + " & ".join(COLUMN_LABELS) + r" & SE D1-D10 \\",
            r"\midrule",
            r"\endhead",
            yearly_table_to_latex(yearly_table),
            r"\bottomrule",
            r"\end{longtable}",
            r"\begin{minipage}{0.95\linewidth}",
            r"\footnotesize Notes: " + " ".join(latex_escape(note) for note in notes),
            r"\end{minipage}",
            r"\end{center}",
        ]
    )
    latex_document = "\n\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[landscape,margin=0.45in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{caption}",
            r"\usepackage{longtable}",
            r"\captionsetup{font=small}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\setcounter{table}{4}",
            table_body,
            r"\end{document}",
        ]
    )

    tex_path = output_dir / "table5_yearly_subsamples.tex"
    tex_path.write_text(latex_document + "\n", encoding="utf-8")

    plain = "\n\n".join(
        [
            "Table 5 yearly subsamples: Mean excess returns of ten portfolios sorted on duration",
            format_yearly_plain(yearly_table),
            "Notes:",
            *notes,
        ]
    )
    (output_dir / "table5_yearly_subsamples.txt").write_text(plain + "\n", encoding="utf-8")

    if build_pdf:
        compile_pdf(tex_path, output_dir)


def main() -> None:
    args = parse_args()
    duration, monthly = load_inputs(args.duration_parquet, args.monthly_parquet)
    end_year = args.end_year
    if end_year is None:
        end_year = latest_complete_end_year(duration, monthly)
    post_end_yyyymm = args.post_end_yyyymm
    if post_end_yyyymm is None:
        post_end_yyyymm = (end_year + 1) * 100 + 6
    yearly_start_yyyymm = args.yearly_start_yyyymm or args.post_start_yyyymm
    yearly_end_yyyymm = args.yearly_end_yyyymm or post_end_yyyymm

    assignments = build_assignments(
        duration,
        monthly,
        start_year=args.start_year,
        end_year=end_year,
    )
    if assignments.empty:
        raise ValueError("No duration-decile assignments could be built from the local files.")

    monthly_returns = build_monthly_returns(assignments, monthly)
    returns = portfolio_returns(monthly_returns)
    factors = read_factor_csv(args.fama_french, ["RF"])
    data = align_with_rf(returns, factors)
    panels = build_panels(
        data,
        args.post_start_yyyymm,
        post_end_yyyymm,
        args.exclude_corona_start_yyyymm,
        args.exclude_corona_end_yyyymm,
    )

    notes = [
        coverage_note(data),
        *panel_coverage_notes(
            data,
            args.post_start_yyyymm,
            post_end_yyyymm,
            args.exclude_corona_start_yyyymm,
            args.exclude_corona_end_yyyymm,
        ),
        "Returns and OLS standard errors are monthly percentages.",
        (
            "Duration portfolios are rebuilt in memory from cash_flow_duration.parquet "
            "and crsp_monthly_clean.parquet, then equally weighted by month."
        ),
    ]
    write_outputs(
        args.output_dir,
        panels,
        notes,
        build_pdf=not args.no_pdf,
    )
    yearly_table = build_yearly_table(data, yearly_start_yyyymm, yearly_end_yyyymm)
    yearly_notes = [
        (
            "Yearly rows are July-June windows from "
            f"{yyyymm_to_label(yearly_start_yyyymm)} to {yyyymm_to_label(yearly_end_yyyymm)}."
        ),
        "Portfolio columns report monthly mean excess returns in percentages.",
        "SE D1-D10 reports the OLS standard error of the monthly long-short spread in percentages.",
    ]
    write_yearly_outputs(
        args.output_dir,
        yearly_table,
        yearly_notes,
        build_pdf=not args.no_pdf,
    )

    print("Table 5: Mean excess returns of ten portfolios sorted on duration (subsamples)\n")
    for title, panel in panels:
        print(title)
        print(format_plain(panel))
        print()
    print("Notes:")
    for note in notes:
        print(f"- {note}")
    print("\nYearly subsamples")
    print(format_yearly_plain(yearly_table))
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
