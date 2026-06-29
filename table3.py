#!/usr/bin/env python3
"""Recreate Weber (2018) Table 3 from local cleaned files.

The table reports pricing errors for ten equal-weight duration-sorted
portfolios under the Fama-French three-factor model, the Carhart four-factor
model, and the Fama-French five-factor model.
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

MODEL_ROWS = [
    ("alpha_FF3", ["Mkt-RF", "SMB", "HML"]),
    ("alpha_FF4", ["Mkt-RF", "SMB", "HML", "Mom"]),
    ("alpha_FF5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save Weber Table 3 factor alphas."
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
        help="Fama-French five-factor monthly CSV.",
    )
    parser.add_argument(
        "--momentum",
        type=Path,
        default=Path("data/raw/F-F_Momentum_Factor.csv"),
        help="Fama-French monthly momentum CSV.",
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


def read_factors(fama_french_path: Path, momentum_path: Path) -> pd.DataFrame:
    ff = read_factor_csv(
        fama_french_path,
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    mom = read_factor_csv(momentum_path, ["Mom"])
    return ff.merge(mom, on="YYYYMM", how="inner").sort_values("YYYYMM")


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


def align_returns_and_factors(
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


def ols_alpha(y: pd.Series, factors: pd.DataFrame) -> tuple[float, float]:
    data = pd.concat([y, factors], axis=1).dropna()
    if len(data) <= len(factors.columns) + 1:
        return np.nan, np.nan

    y_array = data.iloc[:, 0].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(data)),
            data.loc[:, factors.columns].to_numpy(dtype=float),
        ]
    )
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ x.T @ y_array
    resid = y_array - x @ coef
    sigma2 = float(resid.T @ resid) / (len(data) - x.shape[1])
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return float(coef[0]), float(se[0])


def build_table(data: pd.DataFrame) -> pd.DataFrame:
    excess = data.loc[:, range(1, 11)].sub(data["RF"], axis=0)
    excess[11] = data[11]

    rows: dict[str, list[float]] = {}
    for row_name, factor_columns in MODEL_ROWS:
        alphas = []
        ses = []
        factors = data.loc[:, factor_columns]
        for column in range(1, 12):
            alpha, se = ols_alpha(excess[column], factors)
            alphas.append(alpha)
            ses.append(se)
        rows[row_name] = alphas
        rows[f"SE_{row_name}"] = ses

    return pd.DataFrame(rows, index=COLUMN_LABELS).T


def percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * value:.2f}"


def parenthesized_percent_or_blank(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"({100.0 * value:.2f})"


def format_plain(table: pd.DataFrame) -> str:
    formatted = pd.DataFrame(index=table.index, columns=table.columns, dtype="object")
    for row in table.index:
        formatter = parenthesized_percent_or_blank if row.startswith("SE_") else percent_or_blank
        formatted.loc[row] = [formatter(value) for value in table.loc[row]]
    return formatted.to_string()


def latex_label(row: str) -> str:
    label_map = {
        "alpha_FF3": r"$\alpha_{\mathrm{F\&F}\ 3}$",
        "SE_alpha_FF3": "SE",
        "alpha_FF4": r"$\alpha_{\mathrm{F\&F}\ 4}$",
        "SE_alpha_FF4": "SE",
        "alpha_FF5": r"$\alpha_{\mathrm{F\&F}\ 5}$",
        "SE_alpha_FF5": "SE",
    }
    return label_map[row]


def latex_value(row: str, value: float) -> str:
    formatter = parenthesized_percent_or_blank if row.startswith("SE_") else percent_or_blank
    return formatter(value)


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


def table_to_latex(table: pd.DataFrame) -> str:
    lines = []
    for row in table.index:
        values = [latex_value(row, table.loc[row, column]) for column in table.columns]
        lines.append(latex_label(row) + " & " + " & ".join(values) + r" \\")
    return "\n".join(lines)


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
            r"\caption{Fama and French factor alphas of ten portfolios sorted on duration}",
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
            r"\setcounter{table}{2}",
            table_body,
            r"\end{document}",
        ]
    )

    tex_path = output_dir / "table3_fama_french_alphas.tex"
    tex_path.write_text(latex_document + "\n", encoding="utf-8")

    plain = "\n\n".join(
        [
            "Table 3: Fama and French factor alphas of ten portfolios sorted on duration",
            format_plain(table),
            "Notes:",
            *notes,
        ]
    )
    (output_dir / "table3_fama_french_alphas.txt").write_text(
        plain + "\n",
        encoding="utf-8",
    )
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


def coverage_note(data: pd.DataFrame) -> str:
    months = data["YYYYMM"]
    return (
        f"Sample covers {yyyymm_range_label(months.min(), months.max())} "
        f"({months.nunique()} months)."
    )


def main() -> None:
    args = parse_args()
    factors = read_factors(args.fama_french, args.momentum)
    data = align_returns_and_factors(
        equal_weight_returns(args.monthly_returns),
        factors,
        start_yyyymm=args.start_yyyymm,
        end_yyyymm=args.end_yyyymm,
    )
    table = build_table(data)

    notes = [
        coverage_note(data),
        "Alphas and standard errors are monthly percentages.",
        (
            "The F&F 3 model uses Mkt-RF, SMB, and HML; F&F 4 adds momentum; "
            "F&F 5 uses Mkt-RF, SMB, HML, RMW, and CMA."
        ),
    ]
    write_outputs(
        args.output_dir,
        table,
        notes,
        build_pdf=not args.no_pdf,
    )

    print("Table 3: Fama and French factor alphas of ten portfolios sorted on duration\n")
    print(format_plain(table))
    print("\nNotes:")
    for note in notes:
        print(f"- {note}")
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
