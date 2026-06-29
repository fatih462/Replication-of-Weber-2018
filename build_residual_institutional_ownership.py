#!/usr/bin/env python3
"""Add SEC 13F institutional ownership and residual ownership to duration data.

The residual institutional ownership construction follows Weber (2018), Eq. (7):
within each sorting year, regress logit-transformed institutional ownership on
log market equity and squared log market equity, then keep the residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_institutional_ownership_13f import (
    compute_institutional_ownership,
    infer_13f_years,
)


DURATION_KEY_COLUMNS = [
    "gvkey",
    "formation_year",
    "dur",
    "me_dec_millions",
]

REPLACED_DURATION_COLUMNS = [
    "ior_report_year",
    "ior_report_date",
    "ior_permno",
    "ior_permco",
    "ior_cusip8",
    "institutional_shares",
    "shares_outstanding",
    "matched_13f",
    "ior",
    "ior_clipped",
    "ior_logit",
    "rior",
    "ior_size_filter_eligible",
    "ior_size_cutoff_millions",
    "rior_regression_n",
    "rior_regression_r2",
    "ior_security_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute institutional ownership and Weber residual institutional "
            "ownership, then merge them onto the cash-flow duration data."
        )
    )
    parser.add_argument(
        "--duration-parquet",
        type=Path,
        default=Path("data/cash_flow_duration.parquet"),
        help="Main firm-year duration parquet to augment.",
    )
    parser.add_argument(
        "--output-duration-parquet",
        type=Path,
        default=Path("data/cash_flow_duration_with_ior.parquet"),
        help="Where to write the augmented duration parquet.",
    )
    parser.add_argument(
        "--sec-13f-dir",
        type=Path,
        default=Path("data/sec_13f"),
        help="Directory containing 13f_YYYY_12_31_holdings.parquet files.",
    )
    parser.add_argument(
        "--raw-crsp-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CRSP/Compustat CSV extracts.",
    )
    parser.add_argument(
        "--security-io-parquet",
        type=Path,
        default=Path("data/institutional_ownership_from_13f.parquet"),
        help="Security-level institutional ownership parquet to read or write.",
    )
    parser.add_argument(
        "--residual-io-parquet",
        type=Path,
        default=Path("data/residual_institutional_ownership.parquet"),
        help="Security-level residual institutional ownership output parquet.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("data/residual_institutional_ownership_report.json"),
        help="JSON diagnostics output path.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=None,
        help="13F report years to use. Defaults to all local holdings years.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per chunk when building security-level IO from raw CRSP CSVs.",
    )
    parser.add_argument(
        "--recompute-security-io",
        action="store_true",
        help="Recompute security-level IO from holdings even if the parquet exists.",
    )
    parser.add_argument(
        "--no-size-filter",
        action="store_true",
        help="Do not apply Weber's annual 20th percentile size filter before Eq. (7).",
    )
    parser.add_argument(
        "--min-yearly-obs",
        type=int,
        default=20,
        help="Minimum valid yearly observations required to estimate Eq. (7).",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def clean_gvkey(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def one_duration_row_per_year(duration: pd.DataFrame) -> pd.DataFrame:
    out = duration.loc[:, [c for c in [*DURATION_KEY_COLUMNS, "datadate"] if c in duration.columns]].copy()
    out["gvkey"] = clean_gvkey(out["gvkey"])
    out["formation_year"] = pd.to_numeric(out["formation_year"], errors="coerce")
    for column in ["dur", "me_dec_millions"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
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
    return out.drop(columns=["_valid_dur", "datadate"])


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def load_or_build_security_io(args: argparse.Namespace, years: list[int]) -> pd.DataFrame:
    if args.security_io_parquet.exists() and not args.recompute_security_io:
        io = pd.read_parquet(args.security_io_parquet)
        available_years = set(pd.to_numeric(io["report_year"], errors="coerce").dropna().astype(int))
        if set(years).issubset(available_years):
            return io.loc[io["report_year"].isin(years)].copy()

    io = compute_institutional_ownership(
        sec_13f_dir=args.sec_13f_dir,
        raw_crsp_dir=args.raw_crsp_dir,
        years=years,
        chunksize=args.chunksize,
    )
    write_parquet_atomic(io, args.security_io_parquet)
    return io


def add_size_filter(sample: pd.DataFrame, apply_filter: bool) -> pd.DataFrame:
    out = sample.copy()
    if not apply_filter:
        out["ior_size_cutoff_millions"] = np.nan
        out["ior_size_filter_eligible"] = out["me_dec_millions"].gt(0)
        return out

    cutoffs = out.groupby("formation_year")["me_dec_millions"].transform(
        lambda values: values.quantile(0.20)
    )
    out["ior_size_cutoff_millions"] = cutoffs
    out["ior_size_filter_eligible"] = out["me_dec_millions"].ge(cutoffs)
    return out


def estimate_residuals(
    sample: pd.DataFrame,
    *,
    min_yearly_obs: int,
) -> pd.DataFrame:
    out = sample.copy()
    out["ior_clipped"] = np.clip(out["ior"], 0.0001, 0.9999)
    out["ior_logit"] = np.log(out["ior_clipped"] / (1.0 - out["ior_clipped"]))
    out["rior"] = np.nan
    out["rior_regression_n"] = pd.NA
    out["rior_regression_r2"] = np.nan

    valid = (
        out["ior_size_filter_eligible"].fillna(False)
        & out["ior_logit"].notna()
        & out["me_dec_millions"].gt(0)
    )
    for year, group in out.loc[valid].groupby("formation_year", sort=True):
        if len(group) < min_yearly_obs:
            continue

        log_me = np.log(group["me_dec_millions"].to_numpy(dtype="float64"))
        y = group["ior_logit"].to_numpy(dtype="float64")
        x = np.column_stack([np.ones(len(group)), log_me, log_me**2])
        if np.linalg.matrix_rank(x) < x.shape[1]:
            continue

        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        fitted = x @ beta
        resid = y - fitted
        centered = y - y.mean()
        total_ss = float(centered @ centered)
        r2 = np.nan if total_ss <= 0 else 1.0 - float(resid @ resid) / total_ss

        out.loc[group.index, "rior"] = resid
        out.loc[group.index, "rior_regression_n"] = int(len(group))
        out.loc[group.index, "rior_regression_r2"] = r2

    return out


def build_residual_io(
    duration: pd.DataFrame,
    security_io: pd.DataFrame,
    *,
    apply_size_filter: bool,
    min_yearly_obs: int,
) -> pd.DataFrame:
    duration_keys = one_duration_row_per_year(duration)

    io = security_io.copy()
    io["gvkey"] = clean_gvkey(io["gvkey"])
    io["formation_year"] = pd.to_numeric(io["formation_year"], errors="coerce")
    for column in ["PERMNO", "PERMCO", "report_year", "ior", "institutional_shares"]:
        io[column] = pd.to_numeric(io[column], errors="coerce")

    sample = io.merge(
        duration_keys,
        on=["gvkey", "formation_year"],
        how="inner",
        validate="many_to_one",
    )
    sample = sample.dropna(subset=["gvkey", "formation_year", "ior", "me_dec_millions"])
    sample = add_size_filter(sample, apply_filter=apply_size_filter)
    sample = estimate_residuals(sample, min_yearly_obs=min_yearly_obs)

    keep = [
        "report_year",
        "report_date",
        "formation_year",
        "PERMNO",
        "PERMCO",
        "gvkey",
        "conm",
        "CUSIP",
        "cusip8",
        "MthCap",
        "shares_outstanding",
        "institutional_shares",
        "matched_13f",
        "dur",
        "me_dec_millions",
        "ior",
        "ior_clipped",
        "ior_logit",
        "rior",
        "ior_size_filter_eligible",
        "ior_size_cutoff_millions",
        "rior_regression_n",
        "rior_regression_r2",
    ]
    return sample.loc[:, keep].sort_values(["formation_year", "gvkey"]).reset_index(drop=True)


def merge_onto_duration(duration: pd.DataFrame, residual_io: pd.DataFrame) -> pd.DataFrame:
    replacement = residual_io.copy()
    replacement = replacement.sort_values(
        ["formation_year", "gvkey", "MthCap"], ascending=[True, True, False]
    )
    replacement["ior_security_count"] = replacement.groupby(
        ["gvkey", "formation_year"]
    )["PERMNO"].transform("count")
    replacement = replacement.drop_duplicates(["gvkey", "formation_year"])
    replacement = replacement.rename(
        columns={
            "report_year": "ior_report_year",
            "report_date": "ior_report_date",
            "PERMNO": "ior_permno",
            "PERMCO": "ior_permco",
            "cusip8": "ior_cusip8",
        }
    )

    merge_columns = [
        "gvkey",
        "formation_year",
        "ior_report_year",
        "ior_report_date",
        "ior_permno",
        "ior_permco",
        "ior_cusip8",
        "shares_outstanding",
        "institutional_shares",
        "matched_13f",
        "ior",
        "ior_clipped",
        "ior_logit",
        "rior",
        "ior_size_filter_eligible",
        "ior_size_cutoff_millions",
        "rior_regression_n",
        "rior_regression_r2",
        "ior_security_count",
    ]
    out = duration.drop(columns=[c for c in REPLACED_DURATION_COLUMNS if c in duration.columns])
    out = out.merge(
        replacement.loc[:, merge_columns],
        on=["gvkey", "formation_year"],
        how="left",
        validate="many_to_one",
    )
    return out


def diagnostics(
    duration: pd.DataFrame,
    residual_io: pd.DataFrame,
    years: list[int],
    *,
    apply_size_filter: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    eligible = residual_io["ior_size_filter_eligible"].fillna(False)
    with_rior = residual_io["rior"].notna()
    yearly = (
        residual_io.groupby("formation_year")
        .agg(
            rows=("ior", "size"),
            eligible=("ior_size_filter_eligible", "sum"),
            with_rior=("rior", "count"),
            mean_ior=("ior", "mean"),
            mean_rior=("rior", "mean"),
            rior_regression_r2=("rior_regression_r2", "mean"),
        )
        .reset_index()
    )
    return {
        "method": {
            "equation": (
                "Within each formation year, regress log(IOR/(1-IOR)) on a "
                "constant, log(ME), and log(ME)^2; RIOR is the residual."
            ),
            "ior_clipping": "IOR clipped to [0.0001, 0.9999] before the logit transform.",
            "size_filter": (
                "Applied annual 20th percentile me_dec_millions filter before Eq. (7)."
                if apply_size_filter
                else "No size filter applied."
            ),
        },
        "input_years": years,
        "rows": {
            "duration_rows": int(len(duration)),
            "security_io_rows_after_duration_match": int(len(residual_io)),
            "eligible_rows": int(eligible.sum()),
            "rows_with_rior": int(with_rior.sum()),
        },
        "coverage": {
            "formation_year_min": (
                int(residual_io["formation_year"].min()) if len(residual_io) else None
            ),
            "formation_year_max": (
                int(residual_io["formation_year"].max()) if len(residual_io) else None
            ),
            "formation_years_with_rior": [
                int(year)
                for year in sorted(residual_io.loc[with_rior, "formation_year"].dropna().unique())
            ],
        },
        "outputs": {
            "duration_parquet": str(args.output_duration_parquet),
            "security_io_parquet": str(args.security_io_parquet),
            "residual_io_parquet": str(args.residual_io_parquet),
            "report_json": str(args.report_json),
        },
        "yearly": yearly.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    years = sorted(set(args.years if args.years else infer_13f_years(args.sec_13f_dir)))

    duration = pd.read_parquet(args.duration_parquet)
    require_columns(duration, set(DURATION_KEY_COLUMNS), args.duration_parquet)
    duration["gvkey"] = clean_gvkey(duration["gvkey"])
    duration["formation_year"] = pd.to_numeric(duration["formation_year"], errors="coerce")

    security_io = load_or_build_security_io(args, years)
    residual_io = build_residual_io(
        duration,
        security_io,
        apply_size_filter=not args.no_size_filter,
        min_yearly_obs=args.min_yearly_obs,
    )
    augmented = merge_onto_duration(duration, residual_io)

    write_parquet_atomic(residual_io, args.residual_io_parquet)
    write_parquet_atomic(augmented, args.output_duration_parquet)

    report = diagnostics(
        augmented,
        residual_io,
        years,
        apply_size_filter=not args.no_size_filter,
        args=args,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    coverage = report["coverage"]
    rows = report["rows"]
    print("Residual institutional ownership complete")
    print(f"13F report years: {min(years)}-{max(years)}")
    print(
        "Formation years with RIOR: "
        f"{coverage['formation_year_min']}-{coverage['formation_year_max']}"
    )
    print(
        f"Rows matched to duration: {rows['security_io_rows_after_duration_match']:,}; "
        f"rows with RIOR: {rows['rows_with_rior']:,}"
    )
    print(f"Wrote augmented duration parquet to {args.output_duration_parquet}")
    print(f"Wrote security-level residual IO to {args.residual_io_parquet}")
    print(f"Wrote diagnostics to {args.report_json}")


if __name__ == "__main__":
    main()
