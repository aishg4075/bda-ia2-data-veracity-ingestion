#!/usr/bin/env python3
"""Generate docs/references.md from Excel list first, then PDF references section."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _extract_from_xlsx(xlsx_path: Path) -> List[str]:
    references: List[str] = []

    try:
        import pandas as pd
    except Exception:
        return references

    if not xlsx_path.exists():
        return references

    xls = pd.ExcelFile(xlsx_path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}
        author_col = next((cols[c] for c in cols if "author" in c), None)
        title_col = next((cols[c] for c in cols if "title" in c), None)
        year_col = next((cols[c] for c in cols if "year" in c), None)
        venue_col = next((cols[c] for c in cols if any(k in c for k in ["venue", "journal", "conference", "source"])), None)
        link_col = next((cols[c] for c in cols if any(k in c for k in ["url", "doi", "link"])), None)

        for _, row in df.iterrows():
            if author_col and title_col:
                author = _clean_text(row.get(author_col))
                title = _clean_text(row.get(title_col))
                if not title:
                    continue
                year = _clean_text(row.get(year_col)) if year_col else ""
                venue = _clean_text(row.get(venue_col)) if venue_col else ""
                link = _clean_text(row.get(link_col)) if link_col else ""

                pieces = []
                if author:
                    pieces.append(author)
                if title:
                    pieces.append(f"{title}")
                if year:
                    pieces.append(f"({year})")
                if venue:
                    pieces.append(venue)
                if link:
                    pieces.append(link)
                references.append(". ".join([p for p in pieces if p]).strip(" ."))
            else:
                values = [_clean_text(v) for v in row.tolist()]
                values = [v for v in values if v and v.lower() != "nan"]
                if values:
                    references.append(". ".join(values).strip(" ."))

    # Remove duplicates while preserving order.
    seen = set()
    uniq: List[str] = []
    for ref in references:
        key = ref.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ref)
    return uniq


def _extract_from_pdf(pdf_path: Path) -> List[str]:
    refs: List[str] = []
    if not pdf_path.exists():
        return refs

    try:
        from pypdf import PdfReader
    except Exception:
        return refs

    reader = PdfReader(str(pdf_path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    if not text.strip():
        return refs

    # Prefer the final explicit "References" heading to avoid picking
    # earlier mentions in body text.
    heading_matches = list(re.finditer(r"(?im)^references\s*$", text))
    if not heading_matches:
        return refs

    ref_block = text[heading_matches[-1].start() :]
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in ref_block.splitlines()]
    lines = [ln for ln in lines if ln]

    current = ""
    for ln in lines:
        # Ignore section title and naked page numbers.
        if ln.lower() == "references" or re.fullmatch(r"\d+", ln):
            continue

        # Start of a new bracketed citation like: [12] ...
        if re.match(r"^\[\d+\]\s*", ln):
            if current:
                refs.append(current.strip())
            current = ln
            continue

        # Continuation lines belong to the previous reference only.
        if current:
            # Skip naked page-number lines within reference pages.
            if re.fullmatch(r"\d+", ln):
                continue
            current += " " + ln

    if current:
        refs.append(current.strip())

    # Normalize and de-duplicate while preserving order.
    seen = set()
    clean_refs: List[str] = []
    for ref in refs:
        ref = re.sub(r"\s+", " ", ref).strip()
        if not re.match(r"^\[\d+\]\s+", ref):
            continue
        key = ref.lower()
        if key in seen:
            continue
        seen.add(key)
        clean_refs.append(ref)

    return clean_refs


def write_references_md(
    output_path: Path,
    refs: List[str],
    source_notes: List[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# References\n\n")
        f.write("This list is generated from provided source files only. No references are fabricated.\n\n")
        f.write("## Source notes\n\n")
        for note in source_notes:
            f.write(f"- {note}\n")
        f.write("\n")

        if refs:
            f.write("## Extracted references\n\n")
            for idx, ref in enumerate(refs, start=1):
                clean_ref = re.sub(r"^\[\d+\]\s*", "", ref).strip()
                f.write(f"[{idx}] {clean_ref}\n")
        else:
            f.write("## Extracted references\n\n")
            f.write("No references could be extracted from the provided files in this environment.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=PROJECT_ROOT / "Reference List BDA IA2.xlsx",
        help="Path to reference Excel file",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "BDA_GRP12_IA2_LABCA (1).pdf",
        help="Path to project PDF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "references.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_notes: List[str] = []
    refs = _extract_from_xlsx(args.xlsx)
    if refs:
        source_notes.append(f"Excel parsed successfully: {_display_path(args.xlsx)}")
    else:
        source_notes.append(f"Excel unavailable or parsing failed: {_display_path(args.xlsx)}")

    if not refs:
        pdf_refs = _extract_from_pdf(args.pdf)
        if pdf_refs:
            refs = pdf_refs
            source_notes.append(f"PDF references section parsed successfully: {_display_path(args.pdf)}")
        else:
            source_notes.append(f"PDF unavailable or parsing failed: {_display_path(args.pdf)}")

    write_references_md(args.output, refs, source_notes)
    print(f"Wrote {len(refs)} references to {args.output}")


if __name__ == "__main__":
    main()
