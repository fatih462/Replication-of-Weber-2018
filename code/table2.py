#!/usr/bin/env python3
"""Recreate Weber (2018) Table 2, Panels A-C, from local cleaned files.

The script uses duration decile assignments and monthly returns produced by
build_duration_portfolios.py. Panel A and Panel B use equal-weighted returns.
Panel C rebuilds value-weighted monthly returns from stock-level CRSP returns
and lagged market equity.
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
6

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save Weber Table 2 Panels A-C."
    )
    parser.add_argument(
        "--monthly-returns",
        type=Path,
        default=Path("data/duration_decile_monthly_returns.parquet"),
        help="Equal-weight duration decile monthly returns parquet.",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/duration_decile_assignments.parquet"),
        help="Duration decile stock assignments parquet.",
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
        help=(
            "Fama-French five-factor CSV containing Mkt-RF and RF. If this path "
            "does not exist, the script also checks data/raw/ with the same filename."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Directory for formatted table outputs.",
    )
    parser.add_argument(
        "--start-yyyymm",
        type=int,
        default=None,
        help="First return month to include. Defaults to the earliest overlapping input month.",
    )
    parser.add_argument(
        "--end-yyyymm",
        type=int,
        default=None,
        help="Last return month to include. Defaults to the latest overlapping input month.",
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
            if parts[0] == "" and "Mkt-RF" in parts:
                columns = ["YYYYMM", *parts[1:]]
                continue
            if columns is None:
                continue
            date = parts[0]
            if not (date.isdigit() and len(date) == 6):
                continue
            values = [float(value) / 100.0 for value in parts[1 : len(columns)]]
            rows.append(dict(zip(columns, [int(date), *values])))

    if not rows:
        raise ValueError(f"No monthly Fama-French rows found in {path}.")
    factors = pd.DataFrame(rows)
    return factors.loc[:, ["YYYYMM", "Mkt-RF", "RF"]].sort_values("YYYYMM")


def add_spread(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[11] = out[1] - out[10]
    return out


def equal_weight_returns(path: Path) -> pd.DataFrame:
    monthly = pd.read_parquet(path)
    required = {"YYYYMM", "duration_decile", "ew_ret"}
    require_columns(monthly, required, path)
    portfolio = monthly.pivot_table(
        index="YYYYMM",
        columns="duration_decile",
        values="ew_ret",
        aggfunc="mean",
    )
    portfolio = portfolio.reindex(columns=range(1, 11)).sort_index()
    return add_spread(portfolio).reset_index()


def holding_period_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    out["sort_year"] = np.where(out["month"].ge(7), out["year"], out["year"] - 1)
    out["holding_month"] = np.where(out["month"].ge(7), out["month"] - 6, out["month"] + 6)
    return out


def add_lagged_market_equity(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.sort_values(["PERMNO", "YYYYMM"]).copy()
    out["lag_yyyymm"] = out.groupby("PERMNO")["YYYYMM"].shift(1)
    out["lag_me_millions"] = out.groupby("PERMNO")["me_firm_millions"].shift(1)
    previous_month = pd.to_datetime(out["YYYYMM"].astype(str), format="%Y%m") - pd.offsets.MonthBegin(
        1
    )
    out["expected_lag_yyyymm"] = previous_month.dt.year * 100 + previous_month.dt.month
    out.loc[out["lag_yyyymm"].ne(out["expected_lag_yyyymm"]), "lag_me_millions"] = np.nan
    return out


def value_weight_returns(assignments_path: Path, crsp_path: Path) -> pd.DataFrame:
    assignments = pd.read_parquet(
        assignments_path,
        columns=["sort_year", "PERMNO", "duration_decile", "me_firm_millions"],
    )
    monthly = pd.read_parquet(
        crsp_path,
        columns=["PERMNO", "YYYYMM", "year", "month", "ret", "me_firm_millions"],
    )
    require_columns(
        assignments,
        {"sort_year", "PERMNO", "duration_decile", "me_firm_millions"},
        assignments_path,
    )
    require_columns(
        monthly,
        {"PERMNO", "YYYYMM", "year", "month", "ret", "me_firm_millions"},
        crsp_path,
    )

    for column in ["sort_year", "PERMNO", "duration_decile", "me_firm_millions"]:
        assignments[column] = pd.to_numeric(assignments[column], errors="coerce")
    for column in ["PERMNO", "YYYYMM", "year", "month", "ret", "me_firm_millions"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    monthly = add_lagged_market_equity(monthly)
    monthly = holding_period_returns(monthly)
    stock_months = assignments.merge(
        monthly.loc[
            monthly["holding_month"].between(1, 12) & monthly["ret"].notna(),
            ["PERMNO", "YYYYMM", "sort_year", "holding_month", "ret", "lag_me_millions"],
        ],
        on=["PERMNO", "sort_year"],
        how="inner",
    )
    stock_months["weight_me"] = stock_months["lag_me_millions"].combine_first(
        stock_months["me_firm_millions"]
    )
    stock_months = stock_months.loc[stock_months["weight_me"].gt(0)].copy()
    stock_months["weighted_ret"] = stock_months["ret"] * stock_months["weight_me"]

    portfolio = (
        stock_months.groupby(["YYYYMM", "duration_decile"], as_index=False)
        .agg(weighted_ret=("weighted_ret", "sum"), weight=("weight_me", "sum"))
        .assign(vw_ret=lambda frame: frame["weighted_ret"] / frame["weight"])
        .pivot(index="YYYYMM", columns="duration_decile", values="vw_ret")
        .reindex(columns=range(1, 11))
        .sort_index()
    )
    return add_spread(portfolio).reset_index()


def align_with_factors(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    start_yyyymm: int | None,
    end_yyyymm: int | None,
) -> pd.DataFrame:
    data = returns.merge(factors, on="YYYYMM", how="inner")
    if start_yyyymm is not None:
        data = data.loc[data["YYYYMM"].ge(start_yyyymm)].copy()
    if end_yyyymm is not None:
        data = data.loc[data["YYYYMM"].le(end_yyyymm)].copy()
    if data.empty:
        raise ValueError("No overlapping return and factor months in the requested sample.")
    return data.sort_values("YYYYMM").reset_index(drop=True)


def mean_and_se(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan, np.nan
    return clean.mean(), clean.std(ddof=1) / math.sqrt(len(clean))


def capm(y: pd.Series, market: pd.Series) -> dict[str, float]:
    data = pd.concat([y, market], axis=1).dropna()
    data.columns = ["y", "market"]
    if len(data) < 3:
        return {"alpha": np.nan, "alpha_se": np.nan, "beta": np.nan, "beta_se": np.nan}

    x = np.column_stack([np.ones(len(data)), data["market"].to_numpy()])
    y_array = data["y"].to_numpy()
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ x.T @ y_array
    resid = y_array - x @ coef
    sigma2 = float(resid.T @ resid) / (len(data) - x.shape[1])
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return {
        "alpha": float(coef[0]),
        "alpha_se": float(se[0]),
        "beta": float(coef[1]),
        "beta_se": float(se[1]),
    }


def panel_a(data: pd.DataFrame) -> pd.DataFrame:
    excess = data.loc[:, range(1, 11)].sub(data["RF"], axis=0)
    excess[11] = data[11]
    rows: dict[str, list[float]] = {
        "Mean": [],
        "SE": [],
        "beta_CAPM": [],
        "SE_beta": [],
        "alpha_CAPM": [],
        "SE_alpha": [],
    }

    for column in range(1, 12):
        mean, se = mean_and_se(excess[column])
        model = capm(excess[column], data["Mkt-RF"])
        rows["Mean"].append(mean)
        rows["SE"].append(se)
        rows["beta_CAPM"].append(model["beta"])
        rows["SE_beta"].append(model["beta_se"])
        rows["alpha_CAPM"].append(model["alpha"])
        rows["SE_alpha"].append(model["alpha_se"])

    return pd.DataFrame(rows, index=COLUMN_LABELS).T


def panel_b(data: pd.DataFrame) -> pd.DataFrame:
    excess = data.loc[:, range(1, 11)].sub(data["RF"], axis=0)
    excess[11] = data[11]
    ratios = []
    for column in range(1, 12):
        clean = excess[column].dropna()
        ratios.append(clean.mean() / clean.std(ddof=1) if len(clean) > 1 else np.nan)
    return pd.DataFrame([ratios], index=["Sharpe ratio"], columns=COLUMN_LABELS)


def panel_c(data: pd.DataFrame) -> pd.DataFrame:
    excess = data.loc[:, range(1, 10 + 1)].sub(data["RF"], axis=0)
    excess[11] = data[11]
    means = []
    ses = []
    for column in range(1, 12):
        mean, se = mean_and_se(excess[column])
        means.append(mean)
        ses.append(se)
    return pd.DataFrame([means, ses], index=["Mean", "SE"], columns=COLUMN_LABELS)


def percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * value:.2f}"


def decimal_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def format_plain(panel: pd.DataFrame, percent_rows: set[str]) -> str:
    formatted = pd.DataFrame(index=panel.index, columns=panel.columns, dtype="object")
    for row in panel.index:
        formatter = percent_or_blank if row in percent_rows else decimal_or_blank
        formatted.loc[row] = [formatter(value) for value in panel.loc[row]]
    return formatted.to_string()


def latex_value(row: str, value: float, percent_rows: set[str]) -> str:
    if pd.isna(value):
        return ""
    if row in percent_rows:
        return f"{100.0 * value:.2f}"
    return f"{value:.2f}"


def panel_to_latex(title: str, panel: pd.DataFrame, percent_rows: set[str]) -> str:
    lines = [rf"\addlinespace", rf"\multicolumn{{12}}{{l}}{{\textit{{{title}}}}}\\"]
    for row in panel.index:
        values = [latex_value(row, panel.loc[row, column], percent_rows) for column in panel.columns]
        label_map = {
            "beta_CAPM": r"$\beta_{\mathrm{CAPM}}$",
            "alpha_CAPM": r"$\alpha_{\mathrm{CAPM}}$",
            "SE_beta": "SE",
            "SE_alpha": "SE",
        }
        label = label_map.get(row, row)
        lines.append(label + " & " + " & ".join(values) + r" \\")
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    panel_a_df: pd.DataFrame,
    panel_b_df: pd.DataFrame,
    panel_c_df: pd.DataFrame,
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
            r"\caption{Mean excess returns of ten portfolios sorted on duration}",
            r"\begin{tabular}{lrrrrrrrrrrr}",
            r"\toprule",
            " & " + " & ".join(COLUMN_LABELS) + r" \\",
            r"\midrule",
            panel_to_latex(
                "Panel A: mean-excess returns and CAPM",
                panel_a_df,
                {"Mean", "SE", "alpha_CAPM", "SE_alpha"},
            ),
            panel_to_latex("Panel B: monthly Sharpe ratios", panel_b_df, set()),
            panel_to_latex(
                "Panel C: value-weighted returns",
                panel_c_df,
                {"Mean", "SE"},
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.95\linewidth}",
            r"\footnotesize Notes: "
            + " ".join(note.replace("%", r"\%") for note in notes),
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
            r"\setcounter{table}{1}",
            table_body,
            r"\end{document}",
        ]
    )
    tex_path = output_dir / "table2_panels_ac.tex"
    tex_path.write_text(latex_document + "\n", encoding="utf-8")

    plain = "\n\n".join(
        [
            "Table 2: Mean excess returns of ten portfolios sorted on duration",
            "Panel A: mean-excess returns and CAPM",
            format_plain(panel_a_df, {"Mean", "SE", "alpha_CAPM", "SE_alpha"}),
            "Panel B: monthly Sharpe ratios",
            format_plain(panel_b_df, set()),
            "Panel C: value-weighted returns",
            format_plain(panel_c_df, {"Mean", "SE"}),
            "Notes:",
            *notes,
        ]
    )
    (output_dir / "table2_panels_ac.txt").write_text(plain + "\n", encoding="utf-8")
    if build_pdf:
        compile_pdf(tex_path, output_dir)


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


def coverage_note(name: str, data: pd.DataFrame) -> str:
    months = data["YYYYMM"]
    return (
        f"{name} sample covers {yyyymm_range_label(months.min(), months.max())} "
        f"({months.nunique()} months)."
    )


def main() -> None:
    args = parse_args()
    factors = read_fama_french(args.fama_french)

    ew = align_with_factors(
        equal_weight_returns(args.monthly_returns),
        factors,
        start_yyyymm=args.start_yyyymm,
        end_yyyymm=args.end_yyyymm,
    )
    vw = align_with_factors(
        value_weight_returns(args.assignments, args.crsp_monthly),
        factors,
        start_yyyymm=args.start_yyyymm,
        end_yyyymm=args.end_yyyymm,
    )

    panel_a_df = panel_a(ew)
    panel_b_df = panel_b(ew)
    panel_c_df = panel_c(vw)

    notes = [
        coverage_note("Equal-weighted", ew),
        coverage_note("Value-weighted", vw),
        "Returns, alphas, and standard errors are monthly percentages; betas and Sharpe ratios are decimals.",
    ]
    write_outputs(
        args.output_dir,
        panel_a_df,
        panel_b_df,
        panel_c_df,
        notes,
        build_pdf=not args.no_pdf,
    )

    print("Table 2: Mean excess returns of ten portfolios sorted on duration\n")
    print("Panel A: mean-excess returns and CAPM")
    print(format_plain(panel_a_df, {"Mean", "SE", "alpha_CAPM", "SE_alpha"}))
    print("\nPanel B: monthly Sharpe ratios")
    print(format_plain(panel_b_df, set()))
    print("\nPanel C: value-weighted returns")
    print(format_plain(panel_c_df, {"Mean", "SE"}))
    print("\nNotes:")
    for note in notes:
        print(f"- {note}")
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
