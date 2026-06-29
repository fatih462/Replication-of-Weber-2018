#!/usr/bin/env python3
"""Download SEC Form 13F holdings for a calendar year-end report date.

The scraper uses SEC quarterly master indexes to find Form 13F-HR filings,
downloads the complete submission text files, filters them by the filing
header's CONFORMED PERIOD OF REPORT, and extracts the XML information table.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib.parse import urljoin

import pandas as pd
import requests

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - this repository currently has pyarrow.
    pa = None
    pq = None

try:
    from lxml import etree
except ImportError:  # pragma: no cover - lxml is optional but helpful for SEC XML.
    etree = None


SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/"
DEFAULT_EMAIL = "yourmail@gmail.com"
DEFAULT_RATE_LIMIT = 5.0
REPORT_MONTH_DAY = "1231"
FORM_TYPES = {"13F-HR", "13F-HR/A"}

DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
TAG_LINE_RE = re.compile(r"<(?P<tag>[A-Z0-9_-]+)>\s*(?P<value>[^\r\n<]*)", re.IGNORECASE)
PERIOD_RE = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", re.IGNORECASE)
ACCEPTANCE_RE = re.compile(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", re.IGNORECASE)
ACCESSION_RE = re.compile(r"ACCESSION NUMBER:\s*([0-9-]+)", re.IGNORECASE)
SEC_FILE_RE = re.compile(r"SEC FILE NUMBER:\s*([0-9-]+)", re.IGNORECASE)
XML_WRAPPER_RE = re.compile(r"<XML>\s*(.*?)\s*</XML>", re.IGNORECASE | re.DOTALL)


HOLDING_COLUMNS = [
    "report_year",
    "report_date",
    "filer_cik",
    "filer_cik_int",
    "filer_name",
    "filing_manager_name",
    "form_type",
    "accession_number",
    "filed_date",
    "acceptance_datetime",
    "sec_file_number",
    "filing_url",
    "information_table_document",
    "row_number",
    "name_of_issuer",
    "title_of_class",
    "cusip",
    "value_13f_thousands_usd",
    "value_usd",
    "share_or_principal_amount",
    "share_or_principal_type",
    "put_call",
    "investment_discretion",
    "other_manager",
    "voting_authority_sole",
    "voting_authority_shared",
    "voting_authority_none",
]

FILING_COLUMNS = [
    "report_year",
    "report_date",
    "filer_cik",
    "filer_cik_int",
    "filer_name",
    "filing_manager_name",
    "form_type",
    "accession_number",
    "filed_date",
    "acceptance_datetime",
    "sec_file_number",
    "filing_url",
    "source_index_year",
    "source_index_quarter",
    "is_amendment",
    "amendment_number",
    "amendment_type",
    "table_entry_total",
    "table_value_total_13f_thousands_usd",
    "holdings_count",
    "selected_by_policy",
    "selection_reason",
    "information_table_documents",
    "parse_error",
]


@dataclass(frozen=True)
class IndexFiling:
    cik: str
    company_name: str
    form_type: str
    filed_date: str
    filename: str
    source_index_year: int
    source_index_quarter: int

    @property
    def cik_padded(self) -> str:
        return self.cik.zfill(10)

    @property
    def cik_int(self) -> int:
        return int(self.cik)

    @property
    def accession_number(self) -> str:
        return Path(self.filename).name.removesuffix(".txt")

    @property
    def filing_url(self) -> str:
        return urljoin(SEC_ARCHIVES_BASE, self.filename)


@dataclass
class FilingMetadata:
    report_year: int
    report_date: str
    filer_cik: str
    filer_cik_int: int
    filer_name: str
    filing_manager_name: str | None
    form_type: str
    accession_number: str
    filed_date: str
    acceptance_datetime: str | None
    sec_file_number: str | None
    filing_url: str
    source_index_year: int
    source_index_quarter: int
    is_amendment: bool | None
    amendment_number: str | None
    amendment_type: str | None
    table_entry_total: int | None
    table_value_total_13f_thousands_usd: int | None
    holdings_count: int | None = None
    selected_by_policy: bool = False
    selection_reason: str | None = None
    information_table_documents: str | None = None
    parse_error: str | None = None


class RateLimiter:
    """Simple per-process limiter for polite SEC access."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_request = time.monotonic()


class SecClient:
    def __init__(
        self,
        *,
        email: str,
        cache_dir: Path,
        rate_limit: float,
        refresh_cache: bool = False,
        timeout: int = 60,
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh_cache = refresh_cache
        self.timeout = timeout
        self.rate_limiter = RateLimiter(rate_limit)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": f"weber-13f-scraper {email}",
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            }
        )

    def get_text(self, url: str, cache_path: Path) -> str:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not self.refresh_cache:
            return read_cached_text(cache_path)

        response = None
        for attempt in range(1, 5):
            try:
                self.rate_limiter.wait()
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    break
                if attempt == 4:
                    response.raise_for_status()
            except requests.RequestException:
                if attempt == 4:
                    raise
            sleep_seconds = min(60, 2**attempt)
            logging.warning("Retrying SEC request in %s seconds: %s", sleep_seconds, url)
            time.sleep(sleep_seconds)

        if response is None:  # pragma: no cover - defensive guard.
            raise RuntimeError(f"No response received for {url}")
        text = response.content.decode("utf-8", errors="replace")
        if cache_path.suffix == ".gz":
            with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
                handle.write(text)
        else:
            cache_path.write_text(text, encoding="utf-8")
        return text


class HoldingWriter:
    def __init__(self, path: Path, output_format: str) -> None:
        self.path = path
        self.output_format = output_format
        self._rows: list[dict[str, object]] = []
        self._writer = None

        if output_format == "parquet" and pa is None:
            raise RuntimeError("pyarrow is required for parquet output")
        if output_format == "parquet":
            path.parent.mkdir(parents=True, exist_ok=True)
            self._schema = pa.schema(
                [
                    ("report_year", pa.int16()),
                    ("report_date", pa.string()),
                    ("filer_cik", pa.string()),
                    ("filer_cik_int", pa.int64()),
                    ("filer_name", pa.string()),
                    ("filing_manager_name", pa.string()),
                    ("form_type", pa.string()),
                    ("accession_number", pa.string()),
                    ("filed_date", pa.string()),
                    ("acceptance_datetime", pa.string()),
                    ("sec_file_number", pa.string()),
                    ("filing_url", pa.string()),
                    ("information_table_document", pa.string()),
                    ("row_number", pa.int32()),
                    ("name_of_issuer", pa.string()),
                    ("title_of_class", pa.string()),
                    ("cusip", pa.string()),
                    ("value_13f_thousands_usd", pa.int64()),
                    ("value_usd", pa.int64()),
                    ("share_or_principal_amount", pa.int64()),
                    ("share_or_principal_type", pa.string()),
                    ("put_call", pa.string()),
                    ("investment_discretion", pa.string()),
                    ("other_manager", pa.string()),
                    ("voting_authority_sole", pa.int64()),
                    ("voting_authority_shared", pa.int64()),
                    ("voting_authority_none", pa.int64()),
                ]
            )
            self._writer = pq.ParquetWriter(
                path,
                self._schema,
                compression="zstd",
                use_dictionary=True,
            )

    def write_rows(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        if self.output_format == "csv":
            self._rows.extend(rows)
            return
        table = pa.Table.from_pylist(rows, schema=self._schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self.output_format == "csv":
            write_csv(self.path, self._rows, HOLDING_COLUMNS)
        elif self._writer is not None:
            self._writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape SEC Form 13F year-end holdings into Parquet or CSV."
    )
    parser.add_argument("--year", type=int, default=2014, help="Calendar report year.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sec_13f"),
        help="Directory for output files and download cache.",
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
        help="Contact email included in the SEC User-Agent header.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help="SEC requests per second. Keep this below SEC's 10/sec maximum.",
    )
    parser.add_argument(
        "--filing-start-date",
        help="Earliest SEC filed date to scan, YYYY-MM-DD. Defaults to Jan 1 after --year.",
    )
    parser.add_argument(
        "--filing-end-date",
        help="Latest SEC filed date to scan, YYYY-MM-DD. Defaults to Mar 31 after --year.",
    )
    parser.add_argument(
        "--amendment-policy",
        choices=["consolidated", "latest", "all", "original-only"],
        default="consolidated",
        help=(
            "How to handle multiple 13F-HR filings for the same CIK/report date. "
            "'consolidated' keeps the latest full filing/restatement plus new-holdings amendments."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Parquet is recommended; CSV writes gzip-compressed .csv.gz files.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Redownload files even when cached copies are present.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Optional smoke-test limit on candidate 13F filings downloaded from indexes.",
    )
    parser.add_argument(
        "--max-filings",
        type=int,
        help="Optional smoke-test limit on selected report-date filings parsed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def normalize_yyyymmdd(value: str) -> str:
    if re.fullmatch(r"\d{8}", value):
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", value):
        month, day, year = value.split("-")
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    raise ValueError(f"Unsupported date format: {value}")


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(normalize_yyyymmdd(value))


def default_filing_window(year: int) -> tuple[str, str]:
    filing_year = year + 1
    return f"{filing_year}-01-01", f"{filing_year}-03-31"


def quarters_between(start_date: dt.date, end_date: dt.date) -> list[tuple[int, int]]:
    quarters: list[tuple[int, int]] = []
    year = start_date.year
    while year <= end_date.year:
        for quarter in range(1, 5):
            quarter_start_month = 3 * (quarter - 1) + 1
            quarter_start = dt.date(year, quarter_start_month, 1)
            quarter_end_month = quarter_start_month + 2
            if quarter_end_month == 12:
                quarter_end = dt.date(year, 12, 31)
            else:
                quarter_end = dt.date(year, quarter_end_month + 1, 1) - dt.timedelta(days=1)
            if quarter_end >= start_date and quarter_start <= end_date:
                quarters.append((year, quarter))
        year += 1
    return quarters


def master_index_url(year: int, quarter: int) -> str:
    return f"{SEC_ARCHIVES_BASE}edgar/full-index/{year}/QTR{quarter}/master.idx"


def master_index_cache_path(cache_dir: Path, year: int, quarter: int) -> Path:
    return cache_dir / "indexes" / f"{year}_QTR{quarter}_master.idx.gz"


def filing_cache_path(cache_dir: Path, filing: IndexFiling) -> Path:
    return cache_dir / "filings" / f"{filing.accession_number}.txt.gz"


def read_cached_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def parse_master_index(
    text: str,
    *,
    source_index_year: int,
    source_index_quarter: int,
    filing_start_date: dt.date,
    filing_end_date: dt.date,
) -> Iterator[IndexFiling]:
    in_data = False
    for line in text.splitlines():
        if not in_data:
            if line.startswith("CIK|Company Name|Form Type|Date Filed|Filename"):
                in_data = True
            continue
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company_name, form_type, filed_date, filename = [part.strip() for part in parts]
        if form_type not in FORM_TYPES:
            continue
        filed = parse_date(filed_date)
        if not (filing_start_date <= filed <= filing_end_date):
            continue
        yield IndexFiling(
            cik=cik,
            company_name=company_name,
            form_type=form_type,
            filed_date=filed_date,
            filename=filename,
            source_index_year=source_index_year,
            source_index_quarter=source_index_quarter,
        )


def parse_tag_lines(document: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in TAG_LINE_RE.finditer(document[:2000]):
        out[match.group("tag").upper()] = clean_text(match.group("value"))
    return out


def iter_documents(filing_text: str) -> Iterator[tuple[dict[str, str], str]]:
    for match in DOCUMENT_RE.finditer(filing_text):
        document = match.group(1)
        tags = parse_tag_lines(document)
        text_match = re.search(r"<TEXT>\s*(.*)", document, re.IGNORECASE | re.DOTALL)
        if not text_match:
            yield tags, ""
            continue
        text = re.sub(r"</TEXT>\s*$", "", text_match.group(1), flags=re.IGNORECASE | re.DOTALL)
        yield tags, text.strip()


def extract_header_value(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return clean_text(match.group(1)) if match else None


def clean_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def local_name(tag: object) -> str:
    text = str(tag)
    if "}" in text:
        text = text.rsplit("}", 1)[-1]
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text


def first_child(element: object, name: str) -> object | None:
    for child in element:
        if local_name(getattr(child, "tag", "")) == name:
            return child
    return None


def first_text(element: object, path: Sequence[str]) -> str | None:
    current = element
    for name in path:
        current = first_child(current, name)
        if current is None:
            return None
    return clean_text(getattr(current, "text", None))


def iter_local(element: object, name: str) -> Iterator[object]:
    for child in element.iter():
        if local_name(getattr(child, "tag", "")) == name:
            yield child


def extract_xml_payload(text: str, root_names: Sequence[str]) -> bytes | None:
    wrapper = XML_WRAPPER_RE.search(text)
    if wrapper:
        text = wrapper.group(1)

    starts: list[int] = []
    xml_decl = text.find("<?xml")
    if xml_decl >= 0:
        starts.append(xml_decl)
    for root_name in root_names:
        root_match = re.search(rf"<(?:[A-Za-z0-9_.-]+:)?{re.escape(root_name)}\b", text)
        if root_match:
            starts.append(root_match.start())
    if starts:
        text = text[min(starts) :]
    else:
        return None

    text = re.sub(r"</?TEXT>", "", text, flags=re.IGNORECASE).strip()
    return text.encode("utf-8", errors="replace")


def parse_xml(payload: bytes):
    if etree is not None:
        parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False)
        root = etree.fromstring(payload, parser=parser)
        if root is None:
            raise ValueError("XML parser returned no root")
        return root

    import xml.etree.ElementTree as ET

    return ET.fromstring(payload)


def as_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.replace(",", "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def parse_primary_metadata(filing_text: str) -> dict[str, object | None]:
    metadata: dict[str, object | None] = {
        "filing_manager_name": None,
        "is_amendment": None,
        "amendment_number": None,
        "amendment_type": None,
        "table_entry_total": None,
        "table_value_total_13f_thousands_usd": None,
    }

    for tags, document_text in iter_documents(filing_text):
        if tags.get("TYPE", "").upper().startswith("INFORMATION TABLE"):
            continue
        if "edgarSubmission" not in document_text:
            continue
        payload = extract_xml_payload(document_text, ["edgarSubmission"])
        if payload is None:
            continue
        try:
            root = parse_xml(payload)
        except Exception as exc:  # noqa: BLE001 - tolerate individual filing oddities.
            logging.debug("Could not parse primary XML metadata: %s", exc)
            continue

        metadata["filing_manager_name"] = (
            first_text(root, ["formData", "coverPage", "filingManager", "name"])
            or first_text(root, ["headerData", "filerInfo", "filingManager", "name"])
        )
        metadata["is_amendment"] = parse_bool(first_text(root, ["formData", "coverPage", "isAmendment"]))
        metadata["amendment_number"] = first_text(root, ["formData", "coverPage", "amendmentNo"])
        metadata["amendment_type"] = (
            first_text(root, ["formData", "coverPage", "amendmentInfo", "amendmentType"])
            or first_text(root, ["formData", "coverPage", "amendmentType"])
        )
        metadata["table_entry_total"] = as_int(first_text(root, ["formData", "summaryPage", "tableEntryTotal"]))
        metadata["table_value_total_13f_thousands_usd"] = as_int(
            first_text(root, ["formData", "summaryPage", "tableValueTotal"])
        )
        return metadata

    return metadata


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def build_filing_metadata(
    filing: IndexFiling,
    filing_text: str,
    *,
    report_year: int,
    target_report_date: str,
) -> FilingMetadata:
    primary = parse_primary_metadata(filing_text)
    return FilingMetadata(
        report_year=report_year,
        report_date=target_report_date,
        filer_cik=filing.cik_padded,
        filer_cik_int=filing.cik_int,
        filer_name=filing.company_name,
        filing_manager_name=clean_text(primary.get("filing_manager_name")),
        form_type=filing.form_type,
        accession_number=extract_header_value(ACCESSION_RE, filing_text) or filing.accession_number,
        filed_date=filing.filed_date,
        acceptance_datetime=extract_header_value(ACCEPTANCE_RE, filing_text),
        sec_file_number=extract_header_value(SEC_FILE_RE, filing_text),
        filing_url=filing.filing_url,
        source_index_year=filing.source_index_year,
        source_index_quarter=filing.source_index_quarter,
        is_amendment=primary.get("is_amendment"),
        amendment_number=clean_text(primary.get("amendment_number")),
        amendment_type=clean_text(primary.get("amendment_type")),
        table_entry_total=primary.get("table_entry_total"),
        table_value_total_13f_thousands_usd=primary.get("table_value_total_13f_thousands_usd"),
    )


def information_table_documents(filing_text: str) -> list[tuple[str, bytes]]:
    documents: list[tuple[str, bytes]] = []
    for tags, document_text in iter_documents(filing_text):
        document_type = tags.get("TYPE", "").upper()
        filename = tags.get("FILENAME") or "unknown"
        looks_like_info_table = (
            "INFORMATION TABLE" in document_type
            or re.search(r"<(?:[A-Za-z0-9_.-]+:)?informationTable\b", document_text) is not None
        )
        if not looks_like_info_table:
            continue
        payload = extract_xml_payload(document_text, ["informationTable"])
        if payload is not None:
            documents.append((filename, payload))

    if not documents:
        payload = extract_xml_payload(filing_text, ["informationTable"])
        if payload is not None:
            documents.append(("complete_submission_text", payload))
    return documents


def parse_information_table(payload: bytes) -> list[object]:
    root = parse_xml(payload)
    rows = list(iter_local(root, "infoTable"))
    if not rows and local_name(getattr(root, "tag", "")) == "infoTable":
        rows = [root]
    return rows


def parse_holding_row(
    element: object,
    *,
    metadata: FilingMetadata,
    information_table_document: str,
    row_number: int,
) -> dict[str, object]:
    value_13f = as_int(first_text(element, ["value"]))
    shares = as_int(first_text(element, ["shrsOrPrnAmt", "sshPrnamt"]))
    vote_sole = as_int(first_text(element, ["votingAuthority", "Sole"]))
    vote_shared = as_int(first_text(element, ["votingAuthority", "Shared"]))
    vote_none = as_int(first_text(element, ["votingAuthority", "None"]))
    return {
        "report_year": metadata.report_year,
        "report_date": metadata.report_date,
        "filer_cik": metadata.filer_cik,
        "filer_cik_int": metadata.filer_cik_int,
        "filer_name": metadata.filer_name,
        "filing_manager_name": metadata.filing_manager_name,
        "form_type": metadata.form_type,
        "accession_number": metadata.accession_number,
        "filed_date": metadata.filed_date,
        "acceptance_datetime": metadata.acceptance_datetime,
        "sec_file_number": metadata.sec_file_number,
        "filing_url": metadata.filing_url,
        "information_table_document": information_table_document,
        "row_number": row_number,
        "name_of_issuer": first_text(element, ["nameOfIssuer"]),
        "title_of_class": first_text(element, ["titleOfClass"]),
        "cusip": first_text(element, ["cusip"]),
        "value_13f_thousands_usd": value_13f,
        "value_usd": value_13f * 1000 if value_13f is not None else None,
        "share_or_principal_amount": shares,
        "share_or_principal_type": first_text(element, ["shrsOrPrnAmt", "sshPrnamtType"]),
        "put_call": first_text(element, ["putCall"]),
        "investment_discretion": first_text(element, ["investmentDiscretion"]),
        "other_manager": first_text(element, ["otherManager"]),
        "voting_authority_sole": vote_sole,
        "voting_authority_shared": vote_shared,
        "voting_authority_none": vote_none,
    }


def select_filings(
    filings: list[tuple[IndexFiling, FilingMetadata]],
    amendment_policy: str,
) -> list[tuple[IndexFiling, FilingMetadata]]:
    for _, metadata in filings:
        metadata.selected_by_policy = False
        metadata.selection_reason = None

    if amendment_policy == "all":
        selected = filings
        for _, metadata in selected:
            metadata.selection_reason = "all"
    elif amendment_policy == "original-only":
        selected = [(idx, meta) for idx, meta in filings if meta.form_type == "13F-HR"]
        for _, metadata in selected:
            metadata.selection_reason = "original"
    elif amendment_policy == "latest":
        latest_by_cik: dict[str, tuple[IndexFiling, FilingMetadata]] = {}
        for idx, meta in sorted(filings, key=lambda pair: (pair[1].filed_date, pair[1].accession_number)):
            latest_by_cik[meta.filer_cik] = (idx, meta)
        selected = sorted(latest_by_cik.values(), key=lambda pair: (pair[1].filer_cik, pair[1].accession_number))
        for _, metadata in selected:
            metadata.selection_reason = "latest"
    elif amendment_policy == "consolidated":
        selected = select_consolidated_filings(filings)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported amendment policy: {amendment_policy}")

    for _, metadata in selected:
        metadata.selected_by_policy = True
    return selected


def is_new_holdings_amendment(metadata: FilingMetadata) -> bool:
    amendment_type = (metadata.amendment_type or "").upper()
    return "NEW" in amendment_type and "HOLDING" in amendment_type


def select_consolidated_filings(
    filings: list[tuple[IndexFiling, FilingMetadata]],
) -> list[tuple[IndexFiling, FilingMetadata]]:
    by_cik: dict[str, list[tuple[IndexFiling, FilingMetadata]]] = {}
    for pair in filings:
        by_cik.setdefault(pair[1].filer_cik, []).append(pair)

    selected: list[tuple[IndexFiling, FilingMetadata]] = []
    for cik_filings in by_cik.values():
        ordered = sorted(cik_filings, key=lambda pair: (pair[1].filed_date, pair[1].accession_number))
        new_holdings = [pair for pair in ordered if is_new_holdings_amendment(pair[1])]
        full_filings = [pair for pair in ordered if not is_new_holdings_amendment(pair[1])]

        if full_filings:
            latest_full = full_filings[-1]
            latest_full[1].selection_reason = "latest_full_or_restatement"
            selected.append(latest_full)

        for pair in new_holdings:
            pair[1].selection_reason = "new_holdings_amendment"
            selected.append(pair)

        if not full_filings and not new_holdings and ordered:
            ordered[-1][1].selection_reason = "latest_unknown_amendment_type"
            selected.append(ordered[-1])

    return sorted(selected, key=lambda pair: (pair[1].filer_cik, pair[1].filed_date, pair[1].accession_number))


def output_paths(output_dir: Path, year: int, output_format: str) -> dict[str, Path]:
    suffix = "parquet" if output_format == "parquet" else "csv.gz"
    stem = f"13f_{year}_12_31"
    return {
        "holdings": output_dir / f"{stem}_holdings.{suffix}",
        "filings": output_dir / f"{stem}_filings.{suffix}",
        "manifest": output_dir / f"{stem}_manifest.json",
    }


def write_table(path: Path, rows: Iterable[dict[str, object]], columns: Sequence[str], output_format: str) -> None:
    rows = list(rows)
    if output_format == "csv":
        write_csv(path, rows, columns)
        return
    frame = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metadata_to_row(metadata: FilingMetadata) -> dict[str, object]:
    return {column: getattr(metadata, column) for column in FILING_COLUMNS}


def scan_indexes(
    client: SecClient,
    *,
    cache_dir: Path,
    filing_start_date: dt.date,
    filing_end_date: dt.date,
    max_candidates: int | None,
) -> list[IndexFiling]:
    candidates: list[IndexFiling] = []
    for year, quarter in quarters_between(filing_start_date, filing_end_date):
        url = master_index_url(year, quarter)
        logging.info("Downloading/scanning SEC master index %s QTR%s", year, quarter)
        text = client.get_text(url, master_index_cache_path(cache_dir, year, quarter))
        quarter_candidates = list(
            parse_master_index(
                text,
                source_index_year=year,
                source_index_quarter=quarter,
                filing_start_date=filing_start_date,
                filing_end_date=filing_end_date,
            )
        )
        logging.info("Found %s 13F-HR/13F-HR/A candidates in %s QTR%s", len(quarter_candidates), year, quarter)
        for filing in quarter_candidates:
            candidates.append(filing)
            if max_candidates is not None and len(candidates) >= max_candidates:
                logging.warning("Stopping at --max-candidates=%s", max_candidates)
                return candidates
    return candidates


def load_matching_filings(
    client: SecClient,
    *,
    cache_dir: Path,
    candidates: Sequence[IndexFiling],
    report_year: int,
    target_report_yyyymmdd: str,
    target_report_date: str,
) -> list[tuple[IndexFiling, FilingMetadata]]:
    matches: list[tuple[IndexFiling, FilingMetadata]] = []
    for number, filing in enumerate(candidates, start=1):
        if number == 1 or number % 100 == 0:
            logging.info(
                "Checking candidate filing %s/%s; matched %s report-date filings so far",
                number,
                len(candidates),
                len(matches),
            )
        filing_text = client.get_text(filing.filing_url, filing_cache_path(cache_dir, filing))
        period = extract_header_value(PERIOD_RE, filing_text)
        if period != target_report_yyyymmdd:
            continue
        metadata = build_filing_metadata(
            filing,
            filing_text,
            report_year=report_year,
            target_report_date=target_report_date,
        )
        matches.append((filing, metadata))
    return matches


def parse_selected_holdings(
    *,
    cache_dir: Path,
    selected: Sequence[tuple[IndexFiling, FilingMetadata]],
    writer: HoldingWriter,
) -> tuple[int, int]:
    total_rows = 0
    failed_filings = 0
    for number, (index_filing, metadata) in enumerate(selected, start=1):
        if number == 1 or number % 100 == 0:
            logging.info(
                "Parsing information tables %s/%s; wrote %s holdings so far",
                number,
                len(selected),
                total_rows,
            )
        filing_text = read_cached_text(filing_cache_path(cache_dir, index_filing))
        documents = information_table_documents(filing_text)
        metadata.information_table_documents = "|".join(name for name, _ in documents) or None
        rows: list[dict[str, object]] = []
        try:
            if not documents:
                raise ValueError("No XML information table document found")
            for document_name, payload in documents:
                info_rows = parse_information_table(payload)
                for row_index, row in enumerate(info_rows, start=1):
                    rows.append(
                        parse_holding_row(
                            row,
                            metadata=metadata,
                            information_table_document=document_name,
                            row_number=row_index,
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - keep going across thousands of filings.
            failed_filings += 1
            metadata.parse_error = str(exc)
            logging.warning(
                "Failed to parse holdings for %s (%s): %s",
                metadata.filer_cik,
                metadata.accession_number,
                exc,
            )
            continue

        metadata.holdings_count = len(rows)
        total_rows += len(rows)
        writer.write_rows(rows)
    return total_rows, failed_filings


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    default_start, default_end = default_filing_window(args.year)
    filing_start_date = parse_date(args.filing_start_date or default_start)
    filing_end_date = parse_date(args.filing_end_date or default_end)
    if filing_end_date < filing_start_date:
        raise ValueError("--filing-end-date must be on or after --filing-start-date")

    target_report_yyyymmdd = f"{args.year}{REPORT_MONTH_DAY}"
    target_report_date = normalize_yyyymmdd(target_report_yyyymmdd)
    cache_dir = args.output_dir / "cache"
    paths = output_paths(args.output_dir, args.year, args.output_format)

    logging.info("Target report date: %s", target_report_date)
    logging.info("Scanning filed dates from %s through %s", filing_start_date, filing_end_date)
    logging.info("Using SEC User-Agent contact email: %s", args.email)

    client = SecClient(
        email=args.email,
        cache_dir=cache_dir,
        rate_limit=args.rate_limit,
        refresh_cache=args.refresh_cache,
    )

    candidates = scan_indexes(
        client,
        cache_dir=cache_dir,
        filing_start_date=filing_start_date,
        filing_end_date=filing_end_date,
        max_candidates=args.max_candidates,
    )
    logging.info("Total indexed 13F-HR/13F-HR/A candidates: %s", len(candidates))

    matches = load_matching_filings(
        client,
        cache_dir=cache_dir,
        candidates=candidates,
        report_year=args.year,
        target_report_yyyymmdd=target_report_yyyymmdd,
        target_report_date=target_report_date,
    )
    logging.info("Filings matching CONFORMED PERIOD OF REPORT %s: %s", target_report_yyyymmdd, len(matches))

    selected = select_filings(matches, args.amendment_policy)
    if args.max_filings is not None:
        logging.warning("Stopping selected filings at --max-filings=%s", args.max_filings)
        for _, metadata in selected[args.max_filings :]:
            metadata.selected_by_policy = False
            metadata.selection_reason = None
        selected = selected[: args.max_filings]
    logging.info("Selected filings after amendment policy '%s': %s", args.amendment_policy, len(selected))

    writer = HoldingWriter(paths["holdings"], args.output_format)
    try:
        total_holdings, failed_filings = parse_selected_holdings(
            cache_dir=cache_dir,
            selected=selected,
            writer=writer,
        )
    finally:
        writer.close()

    declared_total_checks = [
        metadata
        for _, metadata in selected
        if metadata.table_entry_total is not None and metadata.holdings_count is not None
    ]
    declared_total_mismatches = [
        metadata
        for metadata in declared_total_checks
        if metadata.table_entry_total != metadata.holdings_count
    ]
    if declared_total_mismatches:
        logging.warning(
            "%s selected filings have XML row counts that differ from cover-page tableEntryTotal",
            len(declared_total_mismatches),
        )

    all_metadata = [metadata for _, metadata in matches]
    write_table(
        paths["filings"],
        (metadata_to_row(metadata) for metadata in all_metadata),
        FILING_COLUMNS,
        args.output_format,
    )

    manifest = {
        "report_year": args.year,
        "report_date": target_report_date,
        "target_report_period": target_report_yyyymmdd,
        "filing_start_date": str(filing_start_date),
        "filing_end_date": str(filing_end_date),
        "amendment_policy": args.amendment_policy,
        "output_format": args.output_format,
        "sec_user_agent_email": args.email,
        "rate_limit_requests_per_second": args.rate_limit,
        "candidate_filings": len(candidates),
        "matching_report_date_filings": len(matches),
        "selected_filings": len(selected),
        "failed_selected_filings": failed_filings,
        "holdings_rows": total_holdings,
        "declared_table_entry_total_checks": len(declared_total_checks),
        "declared_table_entry_total_mismatches": len(declared_total_mismatches),
        "paths": {key: str(path) for key, path in paths.items()},
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    logging.info("Wrote holdings: %s", paths["holdings"])
    logging.info("Wrote filings: %s", paths["filings"])
    logging.info("Wrote manifest: %s", paths["manifest"])
    logging.info("Done: %s holdings from %s selected filings", total_holdings, len(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
