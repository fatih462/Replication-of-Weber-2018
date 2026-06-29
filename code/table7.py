#!/usr/bin/env python3
"""Recreate Weber (2018) Table 7 from local cleaned files.

The table reports Fama-French three-factor adjusted returns for duration
portfolios after high and low investor sentiment months, plus sentiment betas
for benchmark-adjusted returns.
"""

from __future__ import annotations

import argparse
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

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save Weber Table 7 sentiment alphas and betas."
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
        help="Fama-French five-factor monthly CSV containing Mkt-RF, SMB, HML, and RF.",
    )
    parser.add_argument(
        "--sentiment",
        type=Path,
        default=Path("data/raw/SENTIMENT.xlsx"),
        help="Baker-Wurgler sentiment workbook.",
    )
    parser.add_argument(
        "--sentiment-column",
        default="SENT",
        help="Sentiment column to use from the DATA sheet.",
    )
    parser.add_argument(
        "--sentiment-mean-scope",
        choices=["sample", "available"],
        default="sample",
        help="Use the analysis sample or all available rows to define high sentiment.",
    )
    parser.add_argument(
        "--raw-sentiment-change",
        action="store_true",
        help="Use raw monthly sentiment changes in Panel B instead of sample-standardized changes.",
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
            if parts[0] == "" and all(column in parts for column in FACTOR_COLUMNS):
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
    require_columns(factors, {"YYYYMM", "RF", *FACTOR_COLUMNS}, path)
    return factors.loc[:, ["YYYYMM", "RF", *FACTOR_COLUMNS]].sort_values("YYYYMM")


def read_sentiment(path: Path, sentiment_column: str) -> pd.DataFrame:
    path = resolve_input_path(path)
    sentiment = pd.read_excel(path, sheet_name="DATA")
    require_columns(sentiment, {"yearmo", sentiment_column}, path)
    sentiment = sentiment.rename(
        columns={"yearmo": "YYYYMM", sentiment_column: "sentiment"}
    )
    sentiment = sentiment.loc[:, ["YYYYMM", "sentiment"]].copy()
    sentiment["YYYYMM"] = pd.to_numeric(sentiment["YYYYMM"], errors="coerce")
    sentiment["sentiment"] = pd.to_numeric(sentiment["sentiment"], errors="coerce")
    sentiment = sentiment.dropna(subset=["YYYYMM", "sentiment"])
    sentiment["YYYYMM"] = sentiment["YYYYMM"].astype(int)
    sentiment = sentiment.sort_values("YYYYMM")
    sentiment["lag_sentiment"] = sentiment["sentiment"].shift(1)
    sentiment["sentiment_change"] = sentiment["sentiment"] - sentiment["lag_sentiment"]
    return sentiment


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


def align_inputs(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    sentiment: pd.DataFrame,
    *,
    start_yyyymm: int | None,
    end_yyyymm: int | None,
) -> pd.DataFrame:
    data = returns.merge(factors, on="YYYYMM", how="inner")
    data = data.merge(sentiment, on="YYYYMM", how="inner")
    if start_yyyymm is not None:
        data = data.loc[data["YYYYMM"].ge(start_yyyymm)].copy()
    if end_yyyymm is not None:
        data = data.loc[data["YYYYMM"].le(end_yyyymm)].copy()
    data = data.dropna(subset=["lag_sentiment", "sentiment_change"])
    if data.empty:
        raise ValueError("No overlapping return, factor, and sentiment months.")
    return data.sort_values("YYYYMM").reset_index(drop=True)


def ols(y: pd.Series, x: pd.DataFrame, *, add_constant: bool) -> tuple[np.ndarray, np.ndarray]:
    data = pd.concat([y, x], axis=1).dropna()
    if len(data) <= x.shape[1] + int(add_constant):
        return np.full(x.shape[1] + int(add_constant), np.nan), np.full(
            x.shape[1] + int(add_constant), np.nan
        )

    y_array = data.iloc[:, 0].to_numpy(dtype=float)
    x_array = data.iloc[:, 1:].to_numpy(dtype=float)
    if add_constant:
        x_array = np.column_stack([np.ones(len(data)), x_array])
    xtx_inv = np.linalg.inv(x_array.T @ x_array)
    coef = xtx_inv @ x_array.T @ y_array
    resid = y_array - x_array @ coef
    sigma2 = float(resid.T @ resid) / (len(data) - x_array.shape[1])
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return coef, se


def sentiment_alpha(
    excess_return: pd.Series,
    data: pd.DataFrame,
) -> tuple[float, float, float, float]:
    regressors = pd.DataFrame(
        {
            "HighSent": data["high_sentiment"].astype(float),
            "LowSent": data["low_sentiment"].astype(float),
            **{column: data[column] for column in FACTOR_COLUMNS},
        }
    )
    coef, se = ols(excess_return, regressors, add_constant=False)
    return float(coef[0]), float(se[0]), float(coef[1]), float(se[1])


def benchmark_adjusted_return(excess_return: pd.Series, data: pd.DataFrame) -> pd.Series:
    regressors = data.loc[:, FACTOR_COLUMNS]
    coef, _ = ols(excess_return, regressors, add_constant=True)
    if pd.isna(coef).all():
        return pd.Series(np.nan, index=data.index)
    fitted_factors = regressors.to_numpy(dtype=float) @ coef[1:]
    return excess_return - fitted_factors


def sentiment_beta(
    excess_return: pd.Series,
    data: pd.DataFrame,
) -> tuple[float, float]:
    adjusted = benchmark_adjusted_return(excess_return, data)
    regressors = data.loc[:, ["sentiment_change_for_beta"]]
    coef, se = ols(adjusted, regressors, add_constant=True)
    return float(coef[1]), float(se[1])


def build_table(data: pd.DataFrame, sentiment_mean: float) -> pd.DataFrame:
    data = data.copy()
    data["high_sentiment"] = data["lag_sentiment"].gt(sentiment_mean)
    data["low_sentiment"] = ~data["high_sentiment"]

    excess = data.loc[:, range(1, 11)].sub(data["RF"], axis=0)
    excess[11] = data[11]

    rows = {
        "alpha_high_sent": [],
        "SE_alpha_high_sent": [],
        "alpha_low_sent": [],
        "SE_alpha_low_sent": [],
        "beta_sent": [],
        "SE_beta_sent": [],
    }

    for column in range(1, 12):
        alpha_high, se_high, alpha_low, se_low = sentiment_alpha(excess[column], data)
        beta, beta_se = sentiment_beta(excess[column], data)
        rows["alpha_high_sent"].append(alpha_high)
        rows["SE_alpha_high_sent"].append(se_high)
        rows["alpha_low_sent"].append(alpha_low)
        rows["SE_alpha_low_sent"].append(se_low)
        rows["beta_sent"].append(beta)
        rows["SE_beta_sent"].append(beta_se)

    return pd.DataFrame(rows, index=COLUMN_LABELS).T


def sentiment_mean_for_scope(
    sentiment: pd.DataFrame,
    data: pd.DataFrame,
    scope: str,
) -> float:
    if scope == "available":
        values = sentiment["sentiment"].dropna()
    else:
        values = data["lag_sentiment"].dropna()
    if values.empty:
        raise ValueError("No sentiment observations available for the requested mean scope.")
    return float(values.mean())


def percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * value:.2f}"


def parenthesized_percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"({100.0 * value:.2f})"


def format_plain(table: pd.DataFrame) -> str:
    display = pd.DataFrame(
        [
            [percent_or_blank(value) for value in table.loc["alpha_high_sent"]],
            [parenthesized_percent_or_blank(value) for value in table.loc["SE_alpha_high_sent"]],
            [percent_or_blank(value) for value in table.loc["alpha_low_sent"]],
            [parenthesized_percent_or_blank(value) for value in table.loc["SE_alpha_low_sent"]],
            [percent_or_blank(value) for value in table.loc["beta_sent"]],
            [parenthesized_percent_or_blank(value) for value in table.loc["SE_beta_sent"]],
        ],
        index=[
            "alpha HighSent",
            "SE",
            "alpha LowSent",
            "SE",
            "beta Sent",
            "SE",
        ],
        columns=table.columns,
    )
    panel_a = display.iloc[:4].to_string()
    panel_b = display.iloc[4:].to_string()
    return "\n".join(["Panel A: sentiment alphas", panel_a, "", "Panel B: sentiment betas", panel_b])


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
    formatter = parenthesized_percent_or_blank if row.startswith("SE_") else percent_or_blank
    return formatter(value)


def table_to_latex(table: pd.DataFrame) -> str:
    row_labels = {
        "alpha_high_sent": r"$\alpha_{\mathrm{HighSent}}$",
        "SE_alpha_high_sent": "SE",
        "alpha_low_sent": r"$\alpha_{\mathrm{LowSent}}$",
        "SE_alpha_low_sent": "SE",
        "beta_sent": r"$\beta_{\mathrm{Sent}}$",
        "SE_beta_sent": "SE",
    }
    lines = [
        r"\addlinespace",
        r"\multicolumn{12}{l}{\textit{Panel A: sentiment alphas}}\\",
    ]
    for row in [
        "alpha_high_sent",
        "SE_alpha_high_sent",
        "alpha_low_sent",
        "SE_alpha_low_sent",
    ]:
        values = [latex_value(row, table.loc[row, column]) for column in table.columns]
        lines.append(row_labels[row] + " & " + " & ".join(values) + r" \\")

    lines.extend(
        [
            r"\addlinespace",
            r"\multicolumn{12}{l}{\textit{Panel B: sentiment betas}}\\",
        ]
    )
    for row in ["beta_sent", "SE_beta_sent"]:
        values = [latex_value(row, table.loc[row, column]) for column in table.columns]
        lines.append(row_labels[row] + " & " + " & ".join(values) + r" \\")
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
        f"Sample covers {yyyymm_range_label(months.min(), months.max())} "
        f"({months.nunique()} months)."
    )


def sentiment_counts_note(data: pd.DataFrame, sentiment_mean: float) -> str:
    high_count = int(data["lag_sentiment"].gt(sentiment_mean).sum())
    low_count = int(data["lag_sentiment"].le(sentiment_mean).sum())
    return (
        f"High sentiment is defined by lagged sentiment above {sentiment_mean:.4f}; "
        f"the sample has {high_count} high-sentiment and {low_count} low-sentiment months."
    )


def add_beta_sentiment_change(data: pd.DataFrame, use_raw_change: bool) -> pd.DataFrame:
    out = data.copy()
    if use_raw_change:
        out["sentiment_change_for_beta"] = out["sentiment_change"]
        return out

    change = out["sentiment_change"]
    change_std = change.std(ddof=0)
    if pd.isna(change_std) or change_std == 0:
        raise ValueError("Cannot standardize sentiment changes because their standard deviation is zero.")
    out["sentiment_change_for_beta"] = (change - change.mean()) / change_std
    return out


def write_outputs(
    output_dir: Path,
    table: pd.DataFrame,
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
            r"\caption{Fama-French alphas of ten portfolios sorted on duration conditional on investor sentiment}",
            r"\begin{tabular}{lrrrrrrrrrrr}",
            r"\toprule",
            " & " + " & ".join(COLUMN_LABELS) + r" \\",
            r"\midrule",
            table_to_latex(table),
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
            r"\setcounter{table}{6}",
            table_body,
            r"\end{document}",
        ]
    )

    tex_path = output_dir / "table7_sentiment_alphas.tex"
    tex_path.write_text(latex_document + "\n", encoding="utf-8")

    plain = "\n\n".join(
        [
            "Table 7: Fama-French alphas of ten portfolios sorted on duration conditional on investor sentiment",
            format_plain(table),
            "Notes:",
            *notes,
        ]
    )
    (output_dir / "table7_sentiment_alphas.txt").write_text(
        plain + "\n",
        encoding="utf-8",
    )

    if build_pdf:
        compile_pdf(tex_path, output_dir)


def main() -> None:
    args = parse_args()
    sentiment = read_sentiment(args.sentiment, args.sentiment_column)
    data = align_inputs(
        equal_weight_returns(args.monthly_returns),
        read_fama_french(args.fama_french),
        sentiment,
        start_yyyymm=args.start_yyyymm,
        end_yyyymm=args.end_yyyymm,
    )
    sentiment_mean = sentiment_mean_for_scope(
        sentiment,
        data,
        args.sentiment_mean_scope,
    )
    data = add_beta_sentiment_change(data, args.raw_sentiment_change)
    table = build_table(data, sentiment_mean)

    beta_change_note = (
        "Panel B uses raw monthly changes in the Baker-Wurgler sentiment index."
        if args.raw_sentiment_change
        else (
            "Panel B uses sample-standardized monthly changes in the Baker-Wurgler "
            "sentiment index."
        )
    )
    notes = [
        coverage_note(data),
        sentiment_counts_note(data, sentiment_mean),
        "Alphas, betas, and standard errors are monthly percentages.",
        (
            "Panel A estimates alphas from excess returns on high- and low-sentiment "
            "dummies plus Mkt-RF, SMB, and HML, without a separate intercept."
        ),
        (
            "Panel B regresses Fama-French three-factor adjusted returns on a constant "
            "and the monthly change in the Baker-Wurgler sentiment index."
        ),
        beta_change_note,
        (
            "Sentiment variables use the Baker-Wurgler index lagged one month; "
            "observations without lagged sentiment or sentiment changes are dropped."
        ),
    ]
    write_outputs(
        args.output_dir,
        table,
        notes,
        build_pdf=not args.no_pdf,
    )

    print("Table 7: Fama-French alphas conditional on investor sentiment\n")
    print(format_plain(table))
    print("\nNotes:")
    for note in notes:
        print(f"- {note}")
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
