#!/usr/bin/env python3
"""Build Weber (2018) cash-flow duration from cleaned CRSP/Compustat files.

The construction follows Weber (2018), Section 2.1.. The script forecasts a finite stream of
accounting cash flows from book equity, ROE, and book-equity growth, then
assigns the remaining market value to a level-perpetuity terminal payoff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ANNUAL_COLUMNS = {
    "gvkey",
    "conm",
    "datadate",
    "fyear",
    "formation_year",
    "age",
    "be",
    "roe",
    "sales_growth",
    "me_dec_millions",
    "bm",
}

REQUIRED_MONTHLY_COLUMNS = {
    "gvkey",
    "PERMCO",
    "YYYYMM",
    "me_firm_millions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Weber cash-flow duration from cleaned parquet files."
    )
    parser.add_argument(
        "--annual-parquet",
        type=Path,
        default=Path("data/compustat_annual_clean.parquet"),
        help="Cleaned annual Compustat parquet from data_cleaning_crsp_compustat.py.",
    )
    parser.add_argument(
        "--monthly-parquet",
        type=Path,
        default=Path("data/crsp_monthly_clean.parquet"),
        help="Cleaned monthly CRSP parquet from data_cleaning_crsp_compustat.py.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/cash_flow_duration.parquet"),
        help="Output parquet containing firm-year duration.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("data/cash_flow_duration_report.json"),
        help="Output JSON report with parameter values and coverage diagnostics.",
    )
    parser.add_argument(
        "--price-source",
        choices=["fiscal", "december"],
        default="fiscal",
        help=(
            "Use fiscal-year-end market equity, matching Eq. (1), or December "
            "market equity, matching the existing book-to-market convention."
        ),
    )
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=0.12,
        help="Expected return on equity r used to discount cash flows.",
    )
    parser.add_argument(
        "--cost-of-equity",
        type=float,
        default=0.12,
        help="Long-run ROE mean used in the AR(1) forecast.",
    )
    parser.add_argument(
        "--long-run-growth",
        type=float,
        default=0.06,
        help="Long-run nominal growth mean used in the AR(1) growth forecast.",
    )
    parser.add_argument(
        "--roe-ar",
        type=float,
        default=0.41,
        help="AR(1) coefficient for ROE, as reported by Weber.",
    )
    parser.add_argument(
        "--growth-ar",
        type=float,
        default=0.24,
        help="AR(1) coefficient for book-equity growth, as reported by Weber.",
    )
    parser.add_argument(
        "--forecast-years",
        type=int,
        default=15,
        help="Finite detailed forecast horizon T.",
    )
    parser.add_argument(
        "--no-winsorize",
        action="store_true",
        help="Do not winsorize forecast inputs or duration at the 1st and 99th percentiles.",
    )
    parser.add_argument(
        "--winsorize-scope",
        choices=["global", "year"],
        default="global",
        help="Winsorize over the full sample or separately by formation year.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def fiscal_yyyymm(datadate: pd.Series) -> pd.Series:
    date = pd.to_datetime(datadate, errors="coerce")
    return (date.dt.year * 100 + date.dt.month).astype("Int64")


def build_fiscal_market_equity(monthly: pd.DataFrame) -> pd.DataFrame:
    market = monthly.loc[:, ["gvkey", "PERMCO", "YYYYMM", "me_firm_millions"]].copy()
    market["gvkey"] = market["gvkey"].astype("string").str.strip()
    market["YYYYMM"] = pd.to_numeric(market["YYYYMM"], errors="coerce").astype("Int64")
    market["PERMCO"] = pd.to_numeric(market["PERMCO"], errors="coerce")
    market["me_firm_millions"] = pd.to_numeric(market["me_firm_millions"], errors="coerce")
    market = market.dropna(subset=["gvkey", "YYYYMM", "PERMCO"])
    market = market.drop_duplicates(["gvkey", "PERMCO", "YYYYMM"])
    market = (
        market.groupby(["gvkey", "YYYYMM"], as_index=False)["me_firm_millions"]
        .sum(min_count=1)
        .rename(columns={"me_firm_millions": "price_fiscal_millions"})
    )
    return market


def winsorize_by_year(frame: pd.DataFrame, column: str) -> pd.Series:
    def clip_one_year(series: pd.Series) -> pd.Series:
        if series.notna().sum() < 20:
            return series
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        return series.clip(lower, upper)

    return frame.groupby("formation_year", group_keys=False)[column].transform(clip_one_year)


def winsorize_global(series: pd.Series) -> pd.Series:
    if series.notna().sum() < 20:
        return series
    lower = series.quantile(0.01)
    upper = series.quantile(0.99)
    return series.clip(lower, upper)


def winsorize(frame: pd.DataFrame, column: str, scope: str) -> pd.Series:
    if scope == "year":
        return winsorize_by_year(frame, column)
    if scope == "global":
        return winsorize_global(frame[column])
    raise ValueError(f"Unknown winsorization scope: {scope}")


def compute_duration(
    frame: pd.DataFrame,
    *,
    roe_column: str,
    growth_column: str,
    discount_rate: float,
    cost_of_equity: float,
    long_run_growth: float,
    roe_ar: float,
    growth_ar: float,
    forecast_years: int,
) -> pd.DataFrame:
    if discount_rate <= 0:
        raise ValueError("--discount-rate must be positive.")
    if forecast_years < 1:
        raise ValueError("--forecast-years must be at least 1.")

    out = frame.copy()
    for col in ["be", roe_column, growth_column, "price_millions"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = (
        out["be"].gt(0)
        & out["price_millions"].gt(0)
        & out[roe_column].notna()
        & out[growth_column].notna()
    )

    duration = np.full(len(out), np.nan, dtype="float64")
    pv_cash_flows = np.full(len(out), np.nan, dtype="float64")
    pv_weighted_cash_flows = np.full(len(out), np.nan, dtype="float64")
    terminal_value = np.full(len(out), np.nan, dtype="float64")

    idx = np.flatnonzero(valid.to_numpy())
    if len(idx):
        book_value = out.loc[valid, "be"].to_numpy(dtype="float64")
        roe_0 = out.loc[valid, roe_column].to_numpy(dtype="float64")
        growth_0 = out.loc[valid, growth_column].to_numpy(dtype="float64")
        price = out.loc[valid, "price_millions"].to_numpy(dtype="float64")

        pv = np.zeros(len(idx), dtype="float64")
        weighted_pv = np.zeros(len(idx), dtype="float64")

        for year in range(1, forecast_years + 1):
            roe_forecast = cost_of_equity + (roe_ar**year) * (roe_0 - cost_of_equity)
            growth_forecast = long_run_growth + (growth_ar**year) * (
                growth_0 - long_run_growth
            )
            cash_flow = book_value * (roe_forecast - growth_forecast)
            discount = (1.0 + discount_rate) ** year
            discounted_cash_flow = cash_flow / discount
            pv += discounted_cash_flow
            weighted_pv += year * discounted_cash_flow
            book_value = book_value * (1.0 + growth_forecast)

        terminal = price - pv
        terminal_weight = forecast_years + (1.0 + discount_rate) / discount_rate
        with np.errstate(divide="ignore", invalid="ignore"):
            dur = weighted_pv / price + terminal_weight * terminal / price

        duration[idx] = dur
        pv_cash_flows[idx] = pv
        pv_weighted_cash_flows[idx] = weighted_pv
        terminal_value[idx] = terminal

    out["dur_raw"] = duration
    out["pv_cash_flows_15y"] = pv_cash_flows
    out["pv_weighted_cash_flows_15y"] = pv_weighted_cash_flows
    out["terminal_value_millions"] = terminal_value
    out["duration_input_valid"] = valid
    return out


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    annual_rows: int,
    output: pd.DataFrame,
) -> None:
    report = {
        "inputs": {
            "annual_parquet": str(args.annual_parquet),
            "monthly_parquet": str(args.monthly_parquet),
        },
        "outputs": {
            "duration_parquet": str(args.output_parquet),
            "report_json": str(args.report_json),
        },
        "method": {
            "source": "Weber (2018), Section 2.1; Dechow, Sloan, and Soliman (2004)",
            "price_source": args.price_source,
            "discount_rate": args.discount_rate,
            "cost_of_equity": args.cost_of_equity,
            "long_run_growth": args.long_run_growth,
            "roe_ar": args.roe_ar,
            "growth_ar": args.growth_ar,
            "forecast_years": args.forecast_years,
            "winsorized": not args.no_winsorize,
            "winsorize_scope": None if args.no_winsorize else args.winsorize_scope,
            "winsorized_forecast_inputs": (
                [] if args.no_winsorize else ["roe_model_input", "sales_growth_model_input"]
            ),
            "winsorized_output": None if args.no_winsorize else "dur",
            "formula_notes": [
                (
                    "Forecast ROE_s = cost_of_equity + roe_ar^s "
                    "* (roe_model_input_t - cost_of_equity)."
                ),
                (
                    "Forecast book-equity growth_s = long_run_growth + growth_ar^s "
                    "* (sales_growth_model_input_t - long_run_growth)."
                ),
                "Cash flow_s = beginning book equity_s * (ROE_s - growth_s).",
                (
                    "Dur = weighted PV of detailed cash flows / price + "
                    "(T + (1+r)/r) * implied terminal value / price."
                ),
            ],
        },
        "coverage": {
            "annual_rows_read": annual_rows,
            "rows_written": int(len(output)),
            "valid_duration_inputs": int(output["duration_input_valid"].sum()),
            "nonmissing_dur_raw": int(output["dur_raw"].notna().sum()),
            "nonmissing_dur": int(output["dur"].notna().sum()),
            "missing_fiscal_price": int(output["price_fiscal_millions"].isna().sum()),
            "missing_december_price": int(output["me_dec_millions"].isna().sum()),
            "missing_roe": int(output["roe"].isna().sum()),
            "missing_sales_growth": int(output["sales_growth"].isna().sum()),
            "nonpositive_book_equity": int(output["be"].le(0).fillna(False).sum()),
            "nonpositive_price": int(output["price_millions"].le(0).fillna(False).sum()),
        },
    }
    finite = output["dur"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        report["duration_summary"] = {
            key: float(value)
            for key, value in finite.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            .to_dict()
            .items()
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    args = parse_args()

    annual = pd.read_parquet(args.annual_parquet)
    require_columns(annual, REQUIRED_ANNUAL_COLUMNS, args.annual_parquet)

    monthly = pd.read_parquet(
        args.monthly_parquet,
        columns=sorted(REQUIRED_MONTHLY_COLUMNS),
    )
    require_columns(monthly, REQUIRED_MONTHLY_COLUMNS, args.monthly_parquet)

    annual = annual.copy()
    annual["gvkey"] = annual["gvkey"].astype("string").str.strip()
    annual["fiscal_yyyymm"] = fiscal_yyyymm(annual["datadate"])

    fiscal_market_equity = build_fiscal_market_equity(monthly)
    annual = annual.merge(
        fiscal_market_equity,
        left_on=["gvkey", "fiscal_yyyymm"],
        right_on=["gvkey", "YYYYMM"],
        how="left",
    ).drop(columns=["YYYYMM"])

    if args.price_source == "fiscal":
        annual["price_millions"] = annual["price_fiscal_millions"]
    else:
        annual["price_millions"] = annual["me_dec_millions"]

    annual["roe_model_input"] = pd.to_numeric(annual["roe"], errors="coerce")
    annual["sales_growth_model_input"] = pd.to_numeric(annual["sales_growth"], errors="coerce")
    if not args.no_winsorize:
        annual["roe_model_input"] = winsorize(annual, "roe_model_input", args.winsorize_scope)
        annual["sales_growth_model_input"] = winsorize(
            annual, "sales_growth_model_input", args.winsorize_scope
        )

    duration = compute_duration(
        annual,
        roe_column="roe_model_input",
        growth_column="sales_growth_model_input",
        discount_rate=args.discount_rate,
        cost_of_equity=args.cost_of_equity,
        long_run_growth=args.long_run_growth,
        roe_ar=args.roe_ar,
        growth_ar=args.growth_ar,
        forecast_years=args.forecast_years,
    )

    duration["dur"] = duration["dur_raw"]
    if not args.no_winsorize:
        duration["dur"] = winsorize(duration, "dur_raw", args.winsorize_scope)

    keep = [
        "gvkey",
        "conm",
        "datadate",
        "fyear",
        "formation_year",
        "age",
        "fiscal_yyyymm",
        "be",
        "roe",
        "sales_growth",
        "roe_model_input",
        "sales_growth_model_input",
        "me_dec_millions",
        "price_fiscal_millions",
        "price_millions",
        "bm",
        "dur",
        "dur_raw",
        "pv_cash_flows_15y",
        "pv_weighted_cash_flows_15y",
        "terminal_value_millions",
        "duration_input_valid",
    ]
    duration = duration.loc[:, keep].sort_values(["gvkey", "datadate"]).reset_index(drop=True)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    duration.to_parquet(args.output_parquet, compression="zstd", index=False)
    write_report(args.report_json, args=args, annual_rows=len(annual), output=duration)

    valid = int(duration["dur"].notna().sum())
    print(f"Wrote {len(duration):,} firm-years to {args.output_parquet}")
    print(f"Nonmissing Dur observations: {valid:,}")
    print(f"Wrote report to {args.report_json}")


if __name__ == "__main__":
    main()
