#!/usr/bin/env python3
"""Check SEC 13F institutional ownership against Weber Table 1 benchmarks.

This script turns raw SEC 13F holdings into an institutional ownership ratio
(IOR) at the CRSP security level:

    IOR = sum(13F shares held by institutions) / CRSP shares outstanding

It then compares Table 1-style descriptive statistics to Weber (2018), Table 1.
The local 13F files currently cover only year-end 2014 and 2015, so the report
is a late-sample sanity check rather than an attempt to reproduce Weber's full
June 1981-June 2014 time-series average.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WEBER_TABLE1 = {
    "sample": "June 1981-June 2014, above 20th size percentile",
    "ior_mean": 0.44,
    "ior_std": 0.23,
    "corr_ior_dur": -0.08,
    "corr_ior_me": 0.22,
    "corr_ior_age": 0.26,
}

CRSP_COLUMNS = [
    "PERMNO",
    "PERMCO",
    "YYYYMM",
    "MthCalDt",
    "MthPrc",
    "MthCap",
    "CUSIP",
    "ShrOut",
    "USIncFlg",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "PrimaryExch",
    "TradingStatusFlg",
    "ConditionalType",
    "SICCD",
    "HdrSICCD",
    "sic",
    "gvkey",
    "conm",
]

STRING_COLUMNS = [
    "CUSIP",
    "gvkey",
    "conm",
    "USIncFlg",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "PrimaryExch",
    "TradingStatusFlg",
    "ConditionalType",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 13F institutional ownership and compare to Weber Table 1."
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
        help="Directory containing the raw YYYY_YYYY.csv.gz CRSP/Compustat extracts.",
    )
    parser.add_argument(
        "--duration-parquet",
        type=Path,
        default=Path("data/cash_flow_duration.parquet"),
        help="Cash-flow duration parquet used for Table 1-style sample filters.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=None,
        help="Report years to check. Defaults to all local 13F holdings years.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/institutional_ownership_from_13f.parquet"),
        help="Output parquet for computed security-level institutional ownership.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("tables/institutional_ownership_check.txt"),
        help="Text report with benchmark comparisons.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per chunk when reading raw CRSP CSV extracts.",
    )
    parser.add_argument(
        "--no-size-filter",
        action="store_true",
        help="Do not apply Weber's annual 20th percentile ME filter.",
    )
    return parser.parse_args()


def infer_13f_years(sec_13f_dir: Path) -> list[int]:
    years = []
    pattern = re.compile(r"13f_(\d{4})_12_31_holdings\.parquet$")
    for path in sec_13f_dir.glob("13f_*_12_31_holdings.parquet"):
        match = pattern.match(path.name)
        if match:
            years.append(int(match.group(1)))
    if not years:
        raise FileNotFoundError(f"No year-end 13F holdings files found in {sec_13f_dir}")
    return sorted(set(years))


def raw_extract_files(raw_crsp_dir: Path, years: Iterable[int]) -> list[Path]:
    wanted = set(years)
    selected = []
    pattern = re.compile(r"(\d{4})_(\d{4})\.csv\.gz$")
    for path in sorted(raw_crsp_dir.glob("*.csv.gz")):
        match = pattern.match(path.name)
        if not match:
            continue
        start, end = int(match.group(1)), int(match.group(2))
        if any(start <= year <= end for year in wanted):
            selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No raw CRSP extracts in {raw_crsp_dir} cover {sorted(wanted)}")
    return selected


def header(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as handle:
        return next(handle).rstrip("\n").split(",")


def available_columns(paths: Iterable[Path]) -> set[str]:
    cols: set[str] = set()
    for path in paths:
        cols.update(header(path))
    return cols


def first_valid_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        replacement = values.mask(values <= 0)
        out = out.where(out.notna(), replacement)
    return out


def crsp_common_stock_mask(frame: pd.DataFrame) -> pd.Series:
    mask = (
        frame["USIncFlg"].eq("Y")
        & frame["SecurityType"].eq("EQTY")
        & frame["SecuritySubType"].eq("COM")
        & frame["ShareType"].eq("NS")
        & frame["PrimaryExch"].isin(["N", "A", "Q"])
        & frame["ConditionalType"].eq("RW")
    )
    if "TradingStatusFlg" in frame.columns:
        mask &= frame["TradingStatusFlg"].isin(["A", "S", "H"])
    return mask.fillna(False)


def nonfinancial_nonutility_mask(frame: pd.DataFrame) -> pd.Series:
    sic = first_valid_numeric(frame, ["SICCD", "HdrSICCD", "sic"])
    is_financial = sic.ge(6000) & sic.lt(7000)
    is_utility = sic.ge(4900) & sic.lt(5000)
    return (~(is_financial | is_utility)).fillna(False)


def load_crsp_december(raw_crsp_dir: Path, years: list[int], chunksize: int) -> pd.DataFrame:
    files = raw_extract_files(raw_crsp_dir, years)
    available = available_columns(files)
    missing = sorted(set(CRSP_COLUMNS) - available)
    if missing:
        raise ValueError(f"Raw CRSP extracts are missing required columns: {missing}")

    wanted_yyyymm = {year * 100 + 12 for year in years}
    parts = []
    for path in files:
        for chunk in pd.read_csv(
            path,
            usecols=CRSP_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
            dtype={column: "string" for column in STRING_COLUMNS},
        ):
            chunk = chunk.loc[chunk["YYYYMM"].isin(wanted_yyyymm)].copy()
            if chunk.empty:
                continue
            chunk = chunk.loc[crsp_common_stock_mask(chunk)].copy()
            chunk = chunk.loc[nonfinancial_nonutility_mask(chunk)].copy()
            if not chunk.empty:
                parts.append(chunk)

    if not parts:
        raise ValueError(f"No December CRSP rows found for years {years}")

    crsp = pd.concat(parts, ignore_index=True)
    for column in ["PERMNO", "PERMCO", "YYYYMM", "MthPrc", "MthCap", "ShrOut"]:
        crsp[column] = pd.to_numeric(crsp[column], errors="coerce")

    crsp = crsp.dropna(subset=["PERMNO", "YYYYMM", "CUSIP", "ShrOut"])
    crsp = crsp.sort_values(["PERMNO", "YYYYMM"]).drop_duplicates(["PERMNO", "YYYYMM"])
    crsp["report_year"] = (crsp["YYYYMM"] // 100).astype(int)
    crsp["report_date"] = pd.to_datetime(crsp["report_year"].astype(str) + "-12-31")
    crsp["formation_year"] = crsp["report_year"] + 1
    crsp["cusip8"] = crsp["CUSIP"].astype("string").str.strip().str.upper().str[:8]
    crsp["shares_outstanding"] = crsp["ShrOut"] * 1000
    return crsp


def load_13f_shares(sec_13f_dir: Path, year: int) -> pd.DataFrame:
    path = sec_13f_dir / f"13f_{year}_12_31_holdings.parquet"
    holdings = pd.read_parquet(
        path,
        columns=[
            "cusip",
            "share_or_principal_amount",
            "share_or_principal_type",
            "put_call",
        ],
    )
    holdings = holdings.loc[
        holdings["share_or_principal_type"].eq("SH") & holdings["put_call"].isna()
    ].copy()
    holdings["cusip8"] = holdings["cusip"].astype("string").str.strip().str.upper().str[:8]
    holdings["institutional_shares"] = pd.to_numeric(
        holdings["share_or_principal_amount"], errors="coerce"
    )
    holdings = holdings.dropna(subset=["cusip8", "institutional_shares"])
    return holdings.groupby("cusip8", as_index=False)["institutional_shares"].sum()


def compute_institutional_ownership(
    sec_13f_dir: Path,
    raw_crsp_dir: Path,
    years: list[int],
    chunksize: int,
) -> pd.DataFrame:
    crsp = load_crsp_december(raw_crsp_dir, years, chunksize)
    out = []
    for year in years:
        year_crsp = crsp.loc[crsp["report_year"].eq(year)].copy()
        year_holdings = load_13f_shares(sec_13f_dir, year)
        year_io = year_crsp.merge(year_holdings, on="cusip8", how="left")
        year_io["institutional_shares"] = year_io["institutional_shares"].fillna(0)
        year_io["ior"] = year_io["institutional_shares"] / year_io["shares_outstanding"]
        year_io["matched_13f"] = year_io["institutional_shares"].gt(0)
        out.append(year_io)

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
        "MthPrc",
        "MthCap",
        "ShrOut",
        "shares_outstanding",
        "institutional_shares",
        "ior",
        "matched_13f",
    ]
    return pd.concat(out, ignore_index=True).loc[:, keep]


def winsorize(series: pd.Series) -> pd.Series:
    if series.notna().sum() < 20:
        return series
    return series.clip(series.quantile(0.01), series.quantile(0.99))


def table1_sample(io: pd.DataFrame, duration_path: Path, apply_size_filter: bool) -> pd.DataFrame:
    duration = pd.read_parquet(
        duration_path,
        columns=["gvkey", "formation_year", "dur", "bm", "me_dec_millions", "age"],
    )
    duration["gvkey"] = duration["gvkey"].astype("string").str.strip()
    sample = io.copy()
    sample["gvkey"] = sample["gvkey"].astype("string").str.strip()
    sample = sample.merge(duration, on=["gvkey", "formation_year"], how="inner")

    if apply_size_filter:
        cutoff = sample.groupby("formation_year")["me_dec_millions"].transform(
            lambda values: values.quantile(0.20)
        )
        sample = sample.loc[sample["me_dec_millions"].ge(cutoff)].copy()
    return sample


def summarize(sample: pd.DataFrame) -> dict[str, float]:
    work = sample.copy()
    work["ior_winsorized"] = work.groupby("formation_year", group_keys=False)["ior"].transform(
        winsorize
    )

    annual = work.groupby("formation_year")["ior_winsorized"].agg(["count", "mean", "std"])
    return {
        "formation_years": ", ".join(str(year) for year in sorted(work["formation_year"].unique())),
        "n": int(len(work)),
        "avg_annual_n": float(annual["count"].mean()),
        "mean_ior": float(annual["mean"].mean()),
        "std_ior": float(annual["std"].mean()),
        "corr_ior_dur": float(work["ior_winsorized"].corr(work["dur"])),
        "corr_ior_me": float(work["ior_winsorized"].corr(work["me_dec_millions"])),
        "corr_ior_age": float(work["ior_winsorized"].corr(work["age"])),
        "raw_mean_ior": float(work["ior"].mean()),
        "raw_std_ior": float(work["ior"].std()),
        "share_ior_above_one": float(work["ior"].gt(1).mean()),
        "matched_13f_share": float(work["matched_13f"].mean()),
    }


def annual_diagnostics(sample: pd.DataFrame) -> pd.DataFrame:
    work = sample.copy()
    work["ior_winsorized"] = work.groupby("formation_year", group_keys=False)["ior"].transform(
        winsorize
    )
    annual = (
        work.groupby("formation_year")
        .agg(
            n=("ior", "size"),
            mean_ior=("ior_winsorized", "mean"),
            std_ior=("ior_winsorized", "std"),
            raw_mean_ior=("ior", "mean"),
            share_ior_above_one=("ior", lambda values: values.gt(1).mean()),
            matched_13f_share=("matched_13f", "mean"),
        )
        .reset_index()
    )
    return annual


def format_report(
    stats: dict[str, float],
    annual: pd.DataFrame,
    years: list[int],
    size_filter: bool,
) -> str:
    rows = [
        ("IOR mean", "ior_mean", stats["mean_ior"], WEBER_TABLE1["ior_mean"]),
        ("IOR std", "ior_std", stats["std_ior"], WEBER_TABLE1["ior_std"]),
        ("corr(IOR, Dur)", "corr_ior_dur", stats["corr_ior_dur"], WEBER_TABLE1["corr_ior_dur"]),
        ("corr(IOR, ME)", "corr_ior_me", stats["corr_ior_me"], WEBER_TABLE1["corr_ior_me"]),
        ("corr(IOR, Age)", "corr_ior_age", stats["corr_ior_age"], WEBER_TABLE1["corr_ior_age"]),
    ]
    lines = [
        "SEC 13F institutional ownership check",
        "=" * 41,
        "",
        f"13F report years checked: {', '.join(str(year) for year in years)}",
        f"Formation years in comparison sample: {stats['formation_years']}",
        f"Comparison sample size: {stats['n']:,}",
        f"Average annual sample size: {stats['avg_annual_n']:,.0f}",
        f"Applied Weber 20th percentile size filter: {size_filter}",
        "IOR statistics below are winsorized by formation year at 1%/99%.",
        "",
        f"Weber Table 1 benchmark sample: {WEBER_TABLE1['sample']}",
        "",
        f"{'Statistic':<18} {'Local 13F':>12} {'Weber T1':>12} {'Difference':>12}",
        "-" * 58,
    ]
    for label, _, local, benchmark in rows:
        lines.append(f"{label:<18} {local:>12.4f} {benchmark:>12.4f} {local - benchmark:>12.4f}")

    lines.extend(
        [
            "",
            "Annual diagnostics:",
            f"{'Formation':>9} {'N':>7} {'Mean':>8} {'Std':>8} {'RawMean':>8} {'IOR>1':>8} {'Matched':>8}",
            "-" * 67,
        ]
    )
    for row in annual.itertuples(index=False):
        lines.append(
            f"{int(row.formation_year):>9} "
            f"{int(row.n):>7,} "
            f"{row.mean_ior:>8.4f} "
            f"{row.std_ior:>8.4f} "
            f"{row.raw_mean_ior:>8.4f} "
            f"{row.share_ior_above_one:>8.4f} "
            f"{row.matched_13f_share:>8.4f}"
        )

    first_year = min(years)
    last_year = max(years)
    lines.extend(
        [
            "",
            "Diagnostics on raw, unwinsorized IOR:",
            f"  mean={stats['raw_mean_ior']:.4f}",
            f"  std={stats['raw_std_ior']:.4f}",
            f"  share with IOR > 1={stats['share_ior_above_one']:.4f}",
            f"  share matched to positive 13F holdings={stats['matched_13f_share']:.4f}",
            "",
            "Interpretation note:",
            f"  The local 13F files are year-end {first_year}-{last_year}, whereas Weber",
            "  Table 1 is a time-series average over June 1981-June 2014. A higher",
            "  late/post-sample IOR mean is therefore not automatically evidence of a",
            "  bad 13F calculation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    years = sorted(set(args.years if args.years else infer_13f_years(args.sec_13f_dir)))

    io = compute_institutional_ownership(
        sec_13f_dir=args.sec_13f_dir,
        raw_crsp_dir=args.raw_crsp_dir,
        years=years,
        chunksize=args.chunksize,
    )
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    io.to_parquet(args.output_parquet, index=False)

    sample = table1_sample(
        io=io,
        duration_path=args.duration_parquet,
        apply_size_filter=not args.no_size_filter,
    )
    stats = summarize(sample)
    annual = annual_diagnostics(sample)
    report = format_report(
        stats,
        annual=annual,
        years=years,
        size_filter=not args.no_size_filter,
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote computed IOR values to {args.output_parquet}")
    print(f"Wrote comparison report to {args.report_path}")


if __name__ == "__main__":
    main()
