"""Small formatting helpers shared by table scripts."""

from __future__ import annotations

from calendar import month_name


def yyyymm_to_month_label(yyyymm: int) -> str:
    value = int(yyyymm)
    year = value // 100
    month = value % 100
    if month < 1 or month > 12:
        raise ValueError(f"Invalid YYYYMM month: {yyyymm}")
    return f"{month_name[month]} {year}"


def yyyymm_range_label(start_yyyymm: int, end_yyyymm: int) -> str:
    return f"{yyyymm_to_month_label(start_yyyymm)} to {yyyymm_to_month_label(end_yyyymm)}"
