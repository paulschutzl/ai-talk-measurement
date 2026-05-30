"""
Transcript Chunk Extractor for Seeking Alpha earnings call transcripts
======================================================================

This script is designed for Seeking Alpha transcripts, which typically contain
a structured header with participant sections at the top of the document.
The parser uses those section labels to:

1. identify the speaker list,
2. separate company speakers from other participants,
3. locate the beginning of the call body,
4. extract speaker blocks from the executive portion of the transcript,
5. apply keyword-based chunking to the selected speech text,
6. export the extracted chunks and a processing log to CSV.

Why the section labels matter:
- "Company Participants" usually lists the executives whose speech belongs to
  the company and should be analyzed.
- "Conference Call Participants" usually lists external participants such as
  analysts, which should be excluded.
- "Operator", "Presentation", and Q&A markers typically indicate the start of
  the body or the transition into the call content.

The script is intentionally configurable so users can adapt it to other
transcript sources by editing:
- the section labels,
- the speaker parsing rules,
- the keyword patterns,
- the output columns.

Run example:
    python extract_chunks.py \
        --input_dir ./transcripts \
        --output_csv ./outputs/chunks.csv \
        --log_csv ./outputs/log.csv
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# USER-EDITABLE CONFIGURATION
# -----------------------------------------------------------------------------
# These labels are tuned for Seeking Alpha transcripts.
# They help the parser locate the speaker list and determine where the body
# of the transcript begins.
COMPANY_SECTION_LABELS = [
    "company participants",
    "corporate participants",
    "company representatives",
]

ALTERNATE_PARTICIPANT_SECTION_LABELS = [
    "conference call participants",
]

BODY_MARKERS = [
    "operator",
    "presentation",
    "question-and-answer session",
    "q&a session",
]

ALL_SECTION_LABELS = set(
    COMPANY_SECTION_LABELS
    + ALTERNATE_PARTICIPANT_SECTION_LABELS
    + BODY_MARKERS
)

# Optional keyword logic.
# Leave this empty if you want to plug in your own keywords later.
KEYWORD_PATTERNS: List[Tuple[str, str]] = [
    # Example:
    # ("generative ai", r"\bgenerative ai\b"),
    # ("machine learning", r"\bmachine learning\b"),
]


# -----------------------------------------------------------------------------
# TEXT NORMALIZATION
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
    )


def normalize_name(name: str) -> str:
    """
    Normalize speaker names for matching:
    - strip honorifics/suffixes,
    - lowercase,
    - collapse whitespace,
    - keep apostrophes and hyphens.
    """
    name = str(name).strip()
    name = re.sub(
        r"\b(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?|Sir|Jr\.?|Sr\.?|II|III|IV|Esq\.?)\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = name.lower()
    name = re.sub(r"[^\w\s'\-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def is_section_header(line: str) -> Optional[str]:
    s = line.strip().lower().rstrip(":").rstrip(".")
    if s in ALL_SECTION_LABELS:
        return s
    return None


# -----------------------------------------------------------------------------
# NAME-PATTERN DETECTION
# -----------------------------------------------------------------------------
NAME_LINE_RE = re.compile(
    r"""
    ^
    (?P<name>
        [A-Z][A-Za-z'`\.\-]+              # first word
        (?:\s+[A-Z][A-Za-z'`\.\-]+){1,4}  # 1-4 more capitalized words
    )
    (?:
        \s*[-\u2013\u2014,]\s*.+          # " - Role" or " - Affiliation"
        |
        \s*\(.+\)\s*                      # " (Role)"
    )?
    \s*$
    """,
    re.VERBOSE,
)


def parse_name_line(line: str) -> Optional[str]:
    """
    If the line looks like a participant entry, return the name portion.
    Otherwise return None.
    """
    line = line.strip()
    if not line or len(line) > 120:
        return None
    if line[-1] in ".!?":
        return None

    m = NAME_LINE_RE.match(line)
    if not m:
        return None

    name = m.group("name").strip()
    if len(name.split()) < 2:
        return None

    bad = {
        "the", "and", "or", "of", "to", "for", "in", "on", "at",
        "we", "i", "our", "this", "that", "but", "so", "as",
    }
    if any(w.lower() in bad for w in name.split()):
        return None

    return name


# -----------------------------------------------------------------------------
# HEADER PARSING
# -----------------------------------------------------------------------------
def parse_header(lines: List[str], max_scan: int = 300) -> Tuple[Set[str], Optional[int]]:
    """
    Detect the participant header and collect speaker names.

    Returns:
        (speaker_set, body_start_line_index)
    """
    header_start = None
    for i, line in enumerate(lines[:max_scan]):
        if is_section_header(line) in COMPANY_SECTION_LABELS:
            header_start = i
            break

    if header_start is None:
        return set(), None

    speakers: Set[str] = set()
    i = header_start + 1

    while i < len(lines) and i < max_scan:
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # In Seeking Alpha transcripts, the conference-call participant section
        # usually contains analysts and other non-company participants.
        # We skip those names so only company speech is analyzed downstream.
        if is_section_header(line) in ALTERNATE_PARTICIPANT_SECTION_LABELS:
            i += 1
            while i < len(lines) and i < max_scan:
                inner = lines[i].strip()
                if not inner:
                    i += 1
                    continue
                if is_section_header(inner) in BODY_MARKERS:
                    return speakers, i
                if parse_name_line(inner) is not None:
                    i += 1
                    continue
                return speakers, i
            return speakers, i

        # These markers typically signal the start of the transcript body.
        if is_section_header(line) in BODY_MARKERS:
            return speakers, i

        name = parse_name_line(line)
        if name is not None:
            speakers.add(normalize_name(name))
            i += 1
        else:
            return speakers, i

    return speakers, i


# -----------------------------------------------------------------------------
# BODY PARSING
# -----------------------------------------------------------------------------
ROLE_KEYWORDS = [
    "ceo", "cfo", "coo", "cto", "cio", "cmo", "evp", "svp", "vp",
    "president", "chairman", "chair", "chief", "director",
    "officer", "head", "manager", "vice president", "executive",
    "analyst", "research", "capital", "markets", "securities",
    "bank", "morgan", "goldman", "wells", "ubs", "credit suisse",
    "barclays", "bofa", "merrill", "jpmorgan", "citi", "deutsche",
    "investor relations", "ir", "division", "group", "partner",
    "founder", "treasurer", "controller",
]


def looks_like_role(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 150:
        return False
    if line[-1] in ".!?":
        return False
    line_lower = line.lower()
    return any(k in line_lower for k in ROLE_KEYWORDS)


def parse_speaker_blocks(
    lines: List[str],
    body_start: int,
    speakers: Set[str],
) -> Iterable[Tuple[str, str]]:
    """
    Walk the body starting at body_start.
    Yield (speaker_name, speech_text) pairs.
    """
    i = body_start
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if is_section_header(line) is not None:
            i += 1
            continue

        candidate_name = parse_name_line(line)
        if candidate_name is not None:
            norm = normalize_name(candidate_name)
            if norm in speakers:
                speech_start = i + 1
                if speech_start < n and looks_like_role(lines[speech_start].strip()):
                    speech_start = i + 2

                speech_lines = []
                j = speech_start

                while j < n:
                    next_line = lines[j].strip()
                    next_name = parse_name_line(next_line)

                    if next_name is not None and normalize_name(next_name) in speakers:
                        break
                    if is_section_header(next_line) is not None:
                        break
                    if next_name is not None and len(next_line) < 60:
                        break

                    speech_lines.append(next_line)
                    j += 1

                speech_text = " ".join([s for s in speech_lines if s]).strip()
                if speech_text:
                    yield candidate_name, speech_text

                i = j
                continue

        i += 1


# -----------------------------------------------------------------------------
# SENTENCE SPLITTING & CHUNKING
# -----------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)


def compile_keywords(keyword_patterns: List[Tuple[str, str]]) -> List[Tuple[str, re.Pattern]]:
    return [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in keyword_patterns]


def matched_keywords_in_sentence(sentence: str, compiled_patterns: List[Tuple[str, re.Pattern]]) -> List[str]:
    hits = []
    for label, pattern in compiled_patterns:
        if pattern.search(sentence):
            hits.append(label)
    return sorted(set(hits))


def extract_chunks(text: str, compiled_patterns: List[Tuple[str, re.Pattern]]) -> List[dict]:
    """
    Extract chunks from a block of text based on keyword hits.

    A chunk begins at a sentence with at least one keyword hit and expands
    across nearby sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    hit_indices = [
        i for i, s in enumerate(sentences)
        if matched_keywords_in_sentence(s, compiled_patterns)
    ]
    if not hit_indices:
        return []

    results = []
    seen_starts = []
    i = 0

    while i < len(hit_indices):
        start_idx = hit_indices[i]
        last_hit_idx = hit_indices[i]

        while i + 1 < len(hit_indices) and hit_indices[i + 1] <= last_hit_idx + 2:
            i += 1
            last_hit_idx = hit_indices[i]

        end_idx = min(last_hit_idx + 3, len(sentences))
        chunk_sentences = sentences[start_idx:end_idx]

        all_keywords = set()
        for idx in range(start_idx, last_hit_idx + 1):
            all_keywords.update(matched_keywords_in_sentence(sentences[idx], compiled_patterns))

        chunk_text = " ".join([s.strip() for s in chunk_sentences if s.strip()]).strip()

        if any(sentences[start_idx].strip() in prev for prev in seen_starts):
            i += 1
            continue

        seen_starts.append(chunk_text)

        results.append(
            {
                "matched_keywords": ", ".join(sorted(all_keywords)),
                "chunk": chunk_text,
                "n_sentences_in_chunk": len(chunk_sentences),
                "matched_sentence": sentences[start_idx].strip(),
            }
        )
        i += 1

    return results


# -----------------------------------------------------------------------------
# OUTPUT FIELDS
# -----------------------------------------------------------------------------
def extract_document_id_from_filename(filename: str) -> str:
    """
    Default identifier derived from the filename stem.
    Users can replace this with any document metadata they prefer.
    """
    return Path(filename).stem


# -----------------------------------------------------------------------------
# MAIN EXTRACTION LOOP
# -----------------------------------------------------------------------------
def run_extraction(
    input_dir: Path,
    output_csv: Path,
    log_csv: Path,
) -> pd.DataFrame:
    compiled_patterns = compile_keywords(KEYWORD_PATTERNS)

    rows = []
    log_records = []
    processed = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = input_dir / filename
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        raw = normalize_text(raw)
        lines = raw.split("\n")

        speakers, body_start = parse_header(lines)

        document_id = extract_document_id_from_filename(filename)

        if not speakers or body_start is None:
            log_records.append(
                {
                    "filename": filename,
                    "document_id": document_id,
                    "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "n_speakers_in_header": len(speakers),
                    "body_start_line": body_start,
                    "n_speaker_blocks_matched": 0,
                    "n_chunks_extracted": 0,
                    "notes": "header parsing failed",
                }
            )
            print(f"  [SKIPPED] {filename}: header parsing failed")
            continue

        n_blocks_matched = 0
        n_chunks_file = 0

        for speaker_name, speech_text in parse_speaker_blocks(lines, body_start, speakers):
            n_blocks_matched += 1
            chunks = extract_chunks(speech_text, compiled_patterns)
            n_chunks_file += len(chunks)

            for item in chunks:
                rows.append(
                    {
                        "document_id": document_id,
                        "file": filename,
                        "speaker": speaker_name,
                        "speaker_norm": normalize_name(speaker_name),
                        "matched_keywords": item["matched_keywords"],
                        "n_sentences_in_chunk": item["n_sentences_in_chunk"],
                        "matched_sentence": item["matched_sentence"],
                        "chunk": item["chunk"],
                    }
                )

        log_records.append(
            {
                "filename": filename,
                "document_id": document_id,
                "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n_speakers_in_header": len(speakers),
                "body_start_line": body_start,
                "n_speaker_blocks_matched": n_blocks_matched,
                "n_chunks_extracted": n_chunks_file,
                "notes": "",
            }
        )

        print(
            f"  {filename}: {document_id} | "
            f"{len(speakers)} speakers, "
            f"body@line {body_start}, "
            f"{n_blocks_matched} blocks, {n_chunks_file} chunks"
        )
        processed += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    log_df = pd.DataFrame(log_records)
    log_df.to_csv(log_csv, index=False)

    print(f"\nDONE. Files processed: {processed}")
    print(f"Total chunks extracted: {len(df)}")
    print(f"Output saved to:        {output_csv}")
    print(f"Audit log saved to:     {log_csv}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract chunks from transcript-like documents.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing input .txt files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="CSV file path for extracted chunks.",
    )
    parser.add_argument(
        "--log_csv",
        type=Path,
        required=True,
        help="CSV file path for the extraction audit log.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(args.input_dir, args.output_csv, args.log_csv)
