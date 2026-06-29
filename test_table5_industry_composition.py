#!/usr/bin/env python3
"""Diagnostic for industry composition in the post-2014 Table 5 extension.

The script tests whether the less smooth middle duration deciles in the
July 2014-June 2025 Table 5 extension can plausibly be explained by industry
composition. It uses SIC codes from the cleaned CRSP file and maps them into
Fama-French 12-style industry groups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from table5 import read_factor_csv


START_YYYYMM = 201407
END_YYYYMM = 202506
OUTPUT_PATH = Path("tables/table5_industry_composition_test.txt")


def ff12_industry(sic_value: float) -> str:
    if pd.isna(sic_value):
        return "Other"
    sic = int(sic_value)

    if (
        100 <= sic <= 999
        or 2000 <= sic <= 2399
        or 2700 <= sic <= 2749
        or 2770 <= sic <= 2799
        or 3100 <= sic <= 3199
        or 3940 <= sic <= 3989
    ):
        return "Consumer NonDurables"
    if (
        2500 <= sic <= 2519
        or 2590 <= sic <= 2599
        or 3630 <= sic <= 3659
        or 3710 <= sic <= 3711
        or sic in {3714, 3716, 3792}
        or 3750 <= sic <= 3751
        or 3900 <= sic <= 3939
        or 3990 <= sic <= 3999
    ):
        return "Consumer Durables"
    if (
        2520 <= sic <= 2589
        or 2600 <= sic <= 2699
        or 2750 <= sic <= 2769
        or 3000 <= sic <= 3099
        or 3200 <= sic <= 3569
        or 3580 <= sic <= 3629
        or 3700 <= sic <= 3709
        or 3712 <= sic <= 3713
        or sic == 3715
        or 3717 <= sic <= 3749
        or 3752 <= sic <= 3791
        or 3793 <= sic <= 3799
        or 3830 <= sic <= 3839
        or 3860 <= sic <= 3899
    ):
        return "Manufacturing"
    if 1200 <= sic <= 1399 or 2900 <= sic <= 2999:
        return "Energy"
    if 2800 <= sic <= 2829 or 2840 <= sic <= 2899:
        return "Chemicals"
    if (
        3570 <= sic <= 3579
        or 3660 <= sic <= 3692
        or 3694 <= sic <= 3699
        or 3810 <= sic <= 3829
        or 7370 <= sic <= 7379
    ):
        return "Business Equipment"
    if 4800 <= sic <= 4899:
        return "Telecom"
    if 5000 <= sic <= 5999 or 7200 <= sic <= 7299 or 7600 <= sic <= 7699:
        return "Shops"
    if 2830 <= sic <= 2839 or sic == 3693 or 3840 <= sic <= 3859 or 8000 <= sic <= 8099:
        return "Healthcare"
    if 4900 <= sic <= 4949:
        return "Utilities"
    if 6000 <= sic <= 6999:
        return "Finance"
    return "Other"


def adjacent_upward_violations(values: pd.Series) -> int:
    ordered = values.sort_index().to_numpy()
    return int(np.sum(ordered[1:] > ordered[:-1]))


def format_percent_series(series: pd.Series) -> str:
    labels = ["Low Dur", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "High Dur"]
    out = pd.DataFrame(
        {
            "Portfolio": labels,
            "Mean monthly return (%)": [f"{value:.2f}" for value in series.sort_index()],
        }
    )
    return out.to_string(index=False)


def main() -> None:
    monthly = pd.read_parquet(
        "data/crsp_monthly_clean.parquet",
        columns=["PERMNO", "YYYYMM", "year", "month", "sic_clean", "ret"],
    )
    assignments = pd.read_parquet(
        "data/duration_decile_assignments.parquet",
        columns=["PERMNO", "sort_year", "duration_decile"],
    )
    factors = read_factor_csv(Path("data/raw/F-F_Research_Data_5_Factors_2x3.csv"), ["RF"])

    monthly["sort_year"] = np.where(
        monthly["month"].ge(7), monthly["year"], monthly["year"] - 1
    ).astype("int64")

    data = monthly.merge(assignments, on=["PERMNO", "sort_year"], how="inner")
    data = data.loc[data["YYYYMM"].between(START_YYYYMM, END_YYYYMM)].copy()
    data = data.merge(factors, on="YYYYMM", how="inner")
    data["industry"] = data["sic_clean"].map(ff12_industry)
    data["excess_ret"] = data["ret"] - data["RF"]

    industry_month = (
        data.groupby(["YYYYMM", "industry"])["excess_ret"].mean().rename("industry_excess_ret")
    )
    data = data.join(industry_month, on=["YYYYMM", "industry"])
    data["industry_adjusted_excess_ret"] = data["excess_ret"] - data["industry_excess_ret"]

    raw_means = data.groupby(["YYYYMM", "duration_decile"])["excess_ret"].mean().unstack().mean()
    adjusted_means = (
        data.groupby(["YYYYMM", "duration_decile"])["industry_adjusted_excess_ret"]
        .mean()
        .unstack()
        .mean()
    )
    raw_means_pct = raw_means * 100.0
    adjusted_means_pct = adjusted_means * 100.0

    shares = pd.crosstab(data["duration_decile"], data["industry"], normalize="index") * 100.0
    top_industries = []
    for decile in range(1, 11):
        top = shares.loc[decile].sort_values(ascending=False).head(4)
        top_industries.append(
            f"D{decile}: " + "; ".join(f"{industry} {share:.1f}%" for industry, share in top.items())
        )

    notes = [
        "Table 5 industry-composition diagnostic",
        "",
        f"Sample: July 2014-June 2025 ({data['YYYYMM'].nunique()} months).",
        "Industry data: sic_clean is available in data/crsp_monthly_clean.parquet.",
        "Method: map SIC codes to Fama-French 12-style industries, compute equal-weight",
        "duration-decile excess returns, then subtract each stock's industry-month",
        "equal-weight excess return before recomputing duration-decile means.",
        "",
        "Raw duration-decile mean monthly excess returns:",
        format_percent_series(raw_means_pct),
        "",
        "Industry-adjusted duration-decile mean monthly excess returns:",
        format_percent_series(adjusted_means_pct),
        "",
        "Smoothness check:",
        (
            "Adjacent upward violations in a decreasing duration profile: "
            f"raw={adjacent_upward_violations(raw_means_pct)}/9, "
            f"industry-adjusted={adjacent_upward_violations(adjusted_means_pct)}/9."
        ),
        (
            "Low-minus-high spread: "
            f"raw={raw_means_pct.loc[1] - raw_means_pct.loc[10]:.2f}%, "
            f"industry-adjusted={adjusted_means_pct.loc[1] - adjusted_means_pct.loc[10]:.2f}%."
        ),
        "",
        "Top industry shares by duration decile, measured over firm-month observations:",
        *top_industries,
        "",
        "Interpretation:",
        "Industry composition is visibly uneven across duration deciles. Healthcare and",
        "Business Equipment become much more important in the highest-duration portfolios,",
        "especially D9-D10. However, subtracting industry-month returns does not make the",
        "middle deciles smoother: the number of monotonicity violations is unchanged.",
        "This suggests that industry composition is a useful descriptive feature of the",
        "post-2014 sample, but it is not by itself a strong explanation for the less smooth",
        "middle-decile pattern. The tail result survives industry adjustment.",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
