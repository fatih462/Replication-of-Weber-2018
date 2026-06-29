#!/usr/bin/env python3
"""Clean the merged CRSP/Compustat CSV extracts for the Weber replication.

The script follows the data filters and accounting-variable definitions in
Weber (2018), Section 2.  It writes two base parquet files:

* crsp_monthly_clean.parquet: CRSP monthly common-stock panel after screens.
* compustat_annual_clean.parquet: annual accounting variables, lagged to the
  return year implied by the paper.

Known limitations of the current raw files:
* The Moody's/Davis et al. hand-collected book-equity supplement is not present.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CRSP_COLUMNS = [
    "PERMNO",
    "PERMCO",
    "YYYYMM",
    "MthCalDt",
    "MthPrc",
    "MthCap",
    "MthRet",
    "MthRetx",
    "MthRetFlg",
    "MthDelFlg",
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
    "datadate",
    "fyear",
]

COMPUSTAT_COLUMNS = [
    "gvkey",
    "datadate",
    "fyear",
    "conm",
    "indfmt",
    "consol",
    "popsrc",
    "datafmt",
    "curcd",
    "seq",
    "ceq",
    "pstk",
    "pstkrv",
    "pstkl",
    "txditc",
    "txdb",
    "itcb",
    "at",
    "lt",
    "dvc",
    "prstkc",
    "sstk",
    "ib",
    "sale",
]

OPTIONAL_DELISTING_COLUMNS = [
    "DLRET",
    "dlret",
    "DelRet",
    "DLSTCD",
    "dlstcd",
    "DelistCode",
]

STRING_COLUMNS = [
    "USIncFlg",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "PrimaryExch",
    "TradingStatusFlg",
    "ConditionalType",
    "MthRetFlg",
    "MthDelFlg",
    "gvkey",
    "conm",
    "indfmt",
    "consol",
    "popsrc",
    "datafmt",
    "curcd",
]


@dataclass
class CleaningStats:
    raw_rows: int = 0
    crsp_after_common_stock_screens: int = 0
    crsp_after_sic_screens: int = 0
    monthly_rows_written: int = 0
    delisting_file_rows: int = 0
    delisting_file_usable_rows: int = 0
    delisting_file_unusable_rows: int = 0
    delisting_flag_rows: int = 0
    delisting_code_matches: int = 0
    missing_return_delisting_rows: int = 0
    delisting_returns_applied: int = 0
    shumway_imputed_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean merged CRSP/Compustat csv.gz files for Weber replication."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help=(
            "Directory containing raw YYYY_YYYY.csv.gz files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for parquet outputs and the cleaning report.",
    )
    parser.add_argument(
        "--delisting-codes",
        type=Path,
        default=Path("data/raw/delisting_codes.csv.gz"),
        help=(
            "Optional CRSP delisting file with PERMNO, DLSTDT, DLSTCD, and DLRET. "
            "If present, delisting returns are compounded with monthly returns and "
            "cause delistings with missing DLRET get Shumway's -30% return."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per pandas CSV chunk.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional debugging limit for each input file.",
    )
    return parser.parse_args()


def input_files(input_dir: Path) -> list[Path]:
    pattern = "[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].csv.gz"
    files = sorted(input_dir.glob(pattern))
    if not files and input_dir != Path("data/raw"):
        fallback = Path("data/raw")
        files = sorted(fallback.glob(pattern))
    if not files:
        searched = [input_dir.resolve()]
        if input_dir != Path("data/raw"):
            searched.append((Path("data/raw")).resolve())
        searched_text = ", ".join(str(path) for path in searched)
        raise FileNotFoundError(f"No YYYY_YYYY.csv.gz files found in: {searched_text}")
    return files


def header(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as handle:
        return next(csv.reader(handle))


def available_columns(files: Iterable[Path]) -> set[str]:
    cols: set[str] = set()
    for path in files:
        cols.update(header(path))
    return cols


def dtype_map(cols: Iterable[str]) -> dict[str, str]:
    return {col: "string" for col in cols if col in STRING_COLUMNS}


def read_chunks(
    path: Path,
    usecols: list[str],
    chunksize: int,
    max_rows: int | None,
) -> Iterable[pd.DataFrame]:
    read_kwargs = {
        "usecols": usecols,
        "chunksize": chunksize,
        "low_memory": False,
        "dtype": dtype_map(usecols),
    }
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows
    yield from pd.read_csv(path, **read_kwargs)


def to_num(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")


def load_delisting_codes(path: Path, stats: CleaningStats) -> pd.DataFrame:
    columns = ["PERMNO", "YYYYMM", "delist_dlstcd", "delist_dlret"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    required = ["PERMNO", "DLSTDT", "DLSTCD", "DLRET"]
    raw = pd.read_csv(path, usecols=required, low_memory=False)
    stats.delisting_file_rows = int(len(raw))

    raw["PERMNO"] = pd.to_numeric(raw["PERMNO"], errors="coerce")
    raw["DLSTDT"] = pd.to_datetime(raw["DLSTDT"], errors="coerce")
    raw["DLSTCD"] = pd.to_numeric(raw["DLSTCD"], errors="coerce")
    raw["DLRET"] = pd.to_numeric(raw["DLRET"], errors="coerce")
    raw["YYYYMM"] = (raw["DLSTDT"].dt.year * 100 + raw["DLSTDT"].dt.month).astype("Int64")

    usable = raw.dropna(subset=["PERMNO", "YYYYMM"]).copy()
    stats.delisting_file_usable_rows = int(len(usable))
    stats.delisting_file_unusable_rows = int(len(raw) - len(usable))
    if usable.empty:
        return pd.DataFrame(columns=columns)

    usable = usable.sort_values(["PERMNO", "DLSTDT"]).drop_duplicates(
        ["PERMNO", "YYYYMM"], keep="last"
    )
    usable = usable.rename(columns={"DLSTCD": "delist_dlstcd", "DLRET": "delist_dlret"})
    usable["PERMNO"] = usable["PERMNO"].astype("int64")
    usable["YYYYMM"] = usable["YYYYMM"].astype("int64")
    return usable.loc[:, columns].reset_index(drop=True)


def clean_gvkey(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def sic_code(frame: pd.DataFrame) -> pd.Series:
    candidates = []
    for col in ["SICCD", "HdrSICCD", "sic"]:
        if col in frame.columns:
            val = pd.to_numeric(frame[col], errors="coerce")
            candidates.append(val.mask(val <= 0))
    if not candidates:
        return pd.Series(np.nan, index=frame.index)
    out = candidates[0]
    for candidate in candidates[1:]:
        out = out.combine_first(candidate)
    return out


def crsp_filter(frame: pd.DataFrame) -> pd.Series:
    """Common US stocks on NYSE, Amex, or Nasdaq, excluding funds/ADRs."""
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


def non_financial_non_utility(frame: pd.DataFrame) -> pd.Series:
    sic = sic_code(frame)
    is_financial = sic.ge(6000) & sic.lt(7000)
    is_utility = sic.ge(4900) & sic.lt(5000)
    return (~(is_financial | is_utility)).fillna(False)


def compustat_filter(frame: pd.DataFrame) -> pd.Series:
    """Standard annual industrial consolidated domestic Compustat records."""
    mask = pd.Series(True, index=frame.index)
    checks = {
        "indfmt": "INDL",
        "consol": "C",
        "popsrc": "D",
        "datafmt": "STD",
    }
    for col, expected in checks.items():
        if col in frame.columns:
            mask &= frame[col].eq(expected)
    return mask.fillna(False)


def write_parquet_chunk(
    writer: pq.ParquetWriter | None,
    frame: pd.DataFrame,
    path: Path,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    else:
        table = table.cast(writer.schema, safe=False)
    writer.write_table(table)
    return writer


def remove_stale_outputs(output_dir: Path) -> None:
    for filename in [
        "crsp_monthly_clean.parquet",
        "compustat_annual_clean.parquet",
        "crsp_compustat_panel.parquet",
        "cleaning_report.json",
    ]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def first_pass(
    files: list[Path],
    usecols: list[str],
    chunksize: int,
    max_rows: int | None,
    stats: CleaningStats,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    firm_me_parts = []
    december_link_parts = []
    annual_parts = []

    for path in files:
        print(f"First pass: {path.name}")
        for chunk in read_chunks(path, usecols, chunksize, max_rows):
            stats.raw_rows += len(chunk)
            chunk["gvkey"] = clean_gvkey(chunk["gvkey"])
            to_num(
                chunk,
                [
                    "PERMNO",
                    "PERMCO",
                    "YYYYMM",
                    "MthCap",
                    "MthRet",
                    "MthRetx",
                    "SICCD",
                    "HdrSICCD",
                    "sic",
                ],
            )

            common = crsp_filter(chunk)
            stats.crsp_after_common_stock_screens += int(common.sum())
            crsp = chunk.loc[common].copy()
            crsp["sic_clean"] = sic_code(crsp)
            sic_ok = non_financial_non_utility(crsp)
            stats.crsp_after_sic_screens += int(sic_ok.sum())
            crsp = crsp.loc[sic_ok].copy()

            delisting_mask = crsp["MthDelFlg"].fillna("N").ne("N")
            stats.delisting_flag_rows += int(delisting_mask.sum())
            stats.missing_return_delisting_rows += int(
                (delisting_mask & crsp["MthRet"].isna()).sum()
            )

            firm_me_parts.append(
                crsp.groupby(["PERMCO", "YYYYMM"], as_index=False)["MthCap"]
                .sum(min_count=1)
                .rename(columns={"MthCap": "me_firm"})
            )

            dec = crsp.loc[
                crsp["YYYYMM"].mod(100).eq(12) & crsp["gvkey"].notna(),
                ["gvkey", "PERMCO", "YYYYMM"],
            ].drop_duplicates()
            if not dec.empty:
                dec["formation_year"] = (dec["YYYYMM"] // 100 + 1).astype("Int64")
                december_link_parts.append(dec[["gvkey", "PERMCO", "YYYYMM", "formation_year"]])

            annual = chunk.loc[compustat_filter(chunk), COMPUSTAT_COLUMNS].drop_duplicates()
            if not annual.empty:
                annual_parts.append(annual)

    firm_me = (
        pd.concat(firm_me_parts, ignore_index=True)
        .groupby(["PERMCO", "YYYYMM"], as_index=False)["me_firm"]
        .sum(min_count=1)
    )
    december_links = (
        pd.concat(december_link_parts, ignore_index=True).drop_duplicates()
        if december_link_parts
        else pd.DataFrame(columns=["gvkey", "PERMCO", "YYYYMM", "formation_year"])
    )
    annual_raw = (
        pd.concat(annual_parts, ignore_index=True).drop_duplicates()
        if annual_parts
        else pd.DataFrame(columns=COMPUSTAT_COLUMNS)
    )
    return firm_me, december_links, annual_raw


def build_december_me(firm_me: pd.DataFrame, december_links: pd.DataFrame) -> pd.DataFrame:
    dec_me = december_links.merge(firm_me, on=["PERMCO", "YYYYMM"], how="left")
    dec_me = dec_me.drop_duplicates(["gvkey", "formation_year", "PERMCO"])
    dec_me = (
        dec_me.groupby(["gvkey", "formation_year"], as_index=False)["me_firm"]
        .sum(min_count=1)
        .rename(columns={"me_firm": "me_dec"})
    )
    dec_me["me_dec_millions"] = dec_me["me_dec"] / 1000.0
    return dec_me


def build_annual(annual_raw: pd.DataFrame, dec_me: pd.DataFrame) -> pd.DataFrame:
    annual = annual_raw.copy()
    annual["gvkey"] = clean_gvkey(annual["gvkey"])
    annual["datadate"] = pd.to_datetime(annual["datadate"], errors="coerce")
    to_num(
        annual,
        [
            "fyear",
            "seq",
            "ceq",
            "pstk",
            "pstkrv",
            "pstkl",
            "txditc",
            "txdb",
            "itcb",
            "at",
            "lt",
            "dvc",
            "prstkc",
            "sstk",
            "ib",
            "sale",
        ],
    )
    annual = annual.dropna(subset=["gvkey", "datadate"])
    annual = annual.sort_values(["gvkey", "datadate"]).drop_duplicates(
        ["gvkey", "datadate"], keep="last"
    )

    shareholders_equity = annual["seq"].combine_first(annual["ceq"] + annual["pstk"])
    shareholders_equity = shareholders_equity.combine_first(annual["at"] - annual["lt"])

    deferred_taxes = annual["txditc"].combine_first(
        annual[["txdb", "itcb"]].fillna(0).sum(axis=1).where(
            annual[["txdb", "itcb"]].notna().any(axis=1)
        )
    )
    preferred_stock = (
        annual["pstkrv"].combine_first(annual["pstkl"]).combine_first(annual["pstk"]).fillna(0)
    )

    annual["shareholders_equity"] = shareholders_equity
    annual["deferred_taxes_itc"] = deferred_taxes.fillna(0)
    annual["preferred_stock"] = preferred_stock
    annual["be"] = annual["shareholders_equity"] + annual["deferred_taxes_itc"] - preferred_stock

    annual = annual.sort_values(["gvkey", "datadate"])
    annual["lag_be"] = annual.groupby("gvkey")["be"].shift(1)
    annual["lag_sale"] = annual.groupby("gvkey")["sale"].shift(1)
    annual["age"] = annual.groupby("gvkey").cumcount() + 1
    annual["formation_year"] = annual["datadate"].dt.year + 1

    annual["ordinary_dividends"] = annual["dvc"].fillna(0)
    annual["net_stock_purchases"] = annual["prstkc"].fillna(0) - annual["sstk"].fillna(0)
    annual["net_payout"] = annual["ordinary_dividends"] + annual["net_stock_purchases"]
    annual["payout_ratio"] = annual["net_payout"] / annual["ib"].replace(0, np.nan)
    annual["roe"] = annual["ib"] / annual["lag_be"].replace(0, np.nan)
    annual["sales_growth"] = annual["sale"] / annual["lag_sale"].replace(0, np.nan) - 1

    annual = annual.merge(dec_me, on=["gvkey", "formation_year"], how="left")
    annual["bm"] = annual["be"] / annual["me_dec_millions"].replace(0, np.nan)
    annual.loc[annual["be"] <= 0, "bm"] = np.nan

    keep = [
        "gvkey",
        "conm",
        "datadate",
        "fyear",
        "formation_year",
        "age",
        "shareholders_equity",
        "deferred_taxes_itc",
        "preferred_stock",
        "be",
        "lag_be",
        "me_dec",
        "me_dec_millions",
        "bm",
        "ib",
        "ordinary_dividends",
        "net_stock_purchases",
        "net_payout",
        "payout_ratio",
        "roe",
        "sale",
        "sales_growth",
    ]
    return annual.loc[annual["age"].ge(2), keep].reset_index(drop=True)


def adjusted_return(frame: pd.DataFrame, stats: CleaningStats) -> pd.Series:
    """Compound CRSP monthly returns with delisting returns when available."""
    ret = pd.to_numeric(frame["MthRet"], errors="coerce")

    if (
        "delist_dlret" in frame.columns
        and "delist_dlstcd" in frame.columns
        and (frame["delist_dlret"].notna() | frame["delist_dlstcd"].notna()).any()
    ):
        dlret_col = "delist_dlret"
        dlstcd_col = "delist_dlstcd"
    else:
        dlret_col = next(
            (col for col in ["DLRET", "dlret", "DelRet"] if col in frame.columns),
            None,
        )
        dlstcd_col = next(
            (col for col in ["DLSTCD", "dlstcd", "DelistCode"] if col in frame.columns),
            None,
        )

    if dlret_col is None:
        return ret

    dlret = pd.to_numeric(frame[dlret_col], errors="coerce")
    if dlstcd_col is not None:
        dlstcd = pd.to_numeric(frame[dlstcd_col], errors="coerce")
        cause_missing = dlstcd.between(400, 591) & dlret.isna()
        dlret = dlret.mask(cause_missing, -0.30)
        stats.shumway_imputed_rows += int(cause_missing.sum())

    stats.delisting_returns_applied += int(dlret.notna().sum())
    both = ret.notna() & dlret.notna()
    out = ret.combine_first(dlret)
    out = out.mask(both, (1 + ret) * (1 + dlret) - 1)
    return out


def prepare_monthly(
    chunk: pd.DataFrame,
    firm_me: pd.DataFrame,
    delisting_codes: pd.DataFrame,
    stats: CleaningStats,
) -> pd.DataFrame:
    chunk["gvkey"] = clean_gvkey(chunk["gvkey"])
    to_num(
        chunk,
        [
            "PERMNO",
            "PERMCO",
            "YYYYMM",
            "MthPrc",
            "MthCap",
            "MthRet",
            "MthRetx",
            "SICCD",
            "HdrSICCD",
            "sic",
        ],
    )
    crsp = chunk.loc[crsp_filter(chunk)].copy()
    crsp["sic_clean"] = sic_code(crsp)
    crsp = crsp.loc[non_financial_non_utility(crsp)].copy()

    crsp = crsp.merge(firm_me, on=["PERMCO", "YYYYMM"], how="left")
    if not delisting_codes.empty:
        crsp = crsp.merge(delisting_codes, on=["PERMNO", "YYYYMM"], how="left")
        stats.delisting_code_matches += int(
            (crsp["delist_dlstcd"].notna() | crsp["delist_dlret"].notna()).sum()
        )
    else:
        crsp["delist_dlstcd"] = np.nan
        crsp["delist_dlret"] = np.nan

    crsp["date"] = pd.to_datetime(crsp["MthCalDt"], errors="coerce")
    crsp["year"] = (crsp["YYYYMM"] // 100).astype("Int64")
    crsp["month"] = crsp["YYYYMM"].mod(100).astype("Int64")
    crsp["sic_clean"] = pd.to_numeric(crsp["sic_clean"], errors="coerce").astype("float64")
    crsp["ret"] = adjusted_return(crsp, stats)
    crsp["retx"] = pd.to_numeric(crsp["MthRetx"], errors="coerce")
    crsp["me_security_millions"] = crsp["MthCap"] / 1000.0
    crsp["me_firm_millions"] = crsp["me_firm"] / 1000.0
    crsp["delisting_flag"] = crsp["MthDelFlg"].fillna("N").ne("N") | crsp[
        "delist_dlstcd"
    ].notna()

    keep = [
        "PERMNO",
        "PERMCO",
        "gvkey",
        "conm",
        "YYYYMM",
        "date",
        "year",
        "month",
        "PrimaryExch",
        "sic_clean",
        "MthRetFlg",
        "MthDelFlg",
        "delisting_flag",
        "delist_dlstcd",
        "delist_dlret",
        "ret",
        "retx",
        "MthPrc",
        "MthCap",
        "me_security_millions",
        "me_firm",
        "me_firm_millions",
    ]
    return crsp.loc[:, keep].reset_index(drop=True)


def second_pass(
    files: list[Path],
    usecols: list[str],
    chunksize: int,
    max_rows: int | None,
    output_dir: Path,
    firm_me: pd.DataFrame,
    delisting_codes: pd.DataFrame,
    stats: CleaningStats,
) -> None:
    monthly_writer = None
    monthly_path = output_dir / "crsp_monthly_clean.parquet"

    try:
        for path in files:
            print(f"Second pass: {path.name}")
            for chunk in read_chunks(path, usecols, chunksize, max_rows):
                monthly = prepare_monthly(chunk, firm_me, delisting_codes, stats)
                if monthly.empty:
                    continue
                stats.monthly_rows_written += len(monthly)
                monthly_writer = write_parquet_chunk(monthly_writer, monthly, monthly_path)
    finally:
        if monthly_writer is not None:
            monthly_writer.close()


def write_report(
    output_dir: Path,
    files: list[Path],
    cols_present: set[str],
    delisting_codes_path: Path,
    delisting_codes_used: bool,
    stats: CleaningStats,
    annual: pd.DataFrame,
) -> None:
    report = {
        "input_files": [str(path) for path in files],
        "outputs": {
            "monthly": str(output_dir / "crsp_monthly_clean.parquet"),
            "annual": str(output_dir / "compustat_annual_clean.parquet"),
        },
        "delisting_codes": {
            "path": str(delisting_codes_path),
            "used": delisting_codes_used,
            "raw_delisting_columns_found": sorted(
                set(OPTIONAL_DELISTING_COLUMNS).intersection(cols_present)
            ),
            "match_keys": ["PERMNO", "YYYYMM from DLSTDT"],
            "return_rule": (
                "Compound MthRet with DLRET when available; if 400 <= DLSTCD <= 591 "
                "and DLRET is missing or nonnumeric, set DLRET to -0.30 before compounding."
            ),
        },
        "filters": {
            "crsp": [
                "USIncFlg == 'Y'",
                "SecurityType == 'EQTY'",
                "SecuritySubType == 'COM'",
                "ShareType == 'NS'",
                "PrimaryExch in {'N', 'A', 'Q'}",
                "ConditionalType == 'RW'",
                "exclude 6000 <= SIC < 7000",
                "exclude 4900 <= SIC < 5000",
            ],
            "compustat": [
                "indfmt == 'INDL'",
                "consol == 'C'",
                "popsrc == 'D'",
                "datafmt == 'STD'",
                "age >= 2 annual Compustat records",
            ],
        },
        "variable_definitions": {
            "be": "shareholders_equity + deferred_taxes_itc - preferred_stock",
            "shareholders_equity": "seq, else ceq + pstk, else at - lt",
            "preferred_stock": "pstkrv, else pstkl, else pstk, else 0",
            "bm": "be / December t-1 firm market equity",
            "payout_ratio": "(dvc + prstkc - sstk) / ib",
            "roe": "ib / lagged be",
            "sales_growth": "sale / lagged sale - 1",
            "age": "count of annual Compustat observations by gvkey",
        },
        "limitations": [
            "No Moody's/Davis et al. book-equity supplement is included in the raw files.",
            (
                "External CRSP delisting return/code file was found and used."
                if delisting_codes_used
                else (
                    "No external CRSP delisting return/code file was found; raw-file "
                    "delisting columns are used only if present."
                )
            ),
            "Delayed delisting-return prorating is not applied without a separate delisting-return date.",
        ],
        "counts": stats.__dict__,
        "annual_rows_written": int(len(annual)),
        "annual_formation_year_min": int(annual["formation_year"].min()) if len(annual) else None,
        "annual_formation_year_max": int(annual["formation_year"].max()) if len(annual) else None,
    }
    with (output_dir / "cleaning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    args = parse_args()
    files = input_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs(args.output_dir)

    cols_present = available_columns(files)
    usecols = [
        col
        for col in dict.fromkeys(CRSP_COLUMNS + COMPUSTAT_COLUMNS + OPTIONAL_DELISTING_COLUMNS)
        if col in cols_present
    ]
    missing_required = sorted((set(CRSP_COLUMNS + COMPUSTAT_COLUMNS) - cols_present))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    stats = CleaningStats()
    delisting_codes = load_delisting_codes(args.delisting_codes, stats)
    firm_me, december_links, annual_raw = first_pass(
        files, usecols, args.chunksize, args.max_rows_per_file, stats
    )
    dec_me = build_december_me(firm_me, december_links)
    annual = build_annual(annual_raw, dec_me)

    annual_path = args.output_dir / "compustat_annual_clean.parquet"
    annual.to_parquet(annual_path, compression="zstd", index=False)

    second_pass(
        files,
        usecols,
        args.chunksize,
        args.max_rows_per_file,
        args.output_dir,
        firm_me,
        delisting_codes,
        stats,
    )
    write_report(
        args.output_dir,
        files,
        cols_present,
        args.delisting_codes,
        not delisting_codes.empty,
        stats,
        annual,
    )

    print(f"Wrote cleaned data to {args.output_dir.resolve()}")
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
