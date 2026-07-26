"""
Markdown-aware chunking. Splits documents on ## headers first (so each chunk
stays topically coherent), then further splits any chunk that's still too
long by word count with overlap, so context isn't lost at chunk boundaries.
"""

import re


def split_by_headers(text: str):
    """Split a markdown document into (header, body) sections on ## headers."""
    sections = re.split(r"\n(?=## )", text.strip())
    results = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        header = lines[0].lstrip("#").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        results.append((header, body))
    return results


def chunk_text(text: str, max_words: int = 180, overlap_words: int = 30):
    """Word-count-based chunking with overlap for sections that are too long."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def chunk_document(filepath: str, doc_title: str):
    """
    Read a markdown file and return a list of chunk dicts:
    {text, source, section, chunk_id}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Drop the top-level "# Title" line, we already track doc_title separately.
    raw = re.sub(r"^#\s+.*\n", "", raw.strip(), count=1)

    sections = split_by_headers(raw)
    chunks = []
    chunk_idx = 0
    for header, body in sections:
        full_text = f"{header}. {body}" if header else body
        for piece in chunk_text(full_text):
            chunks.append({
                "text": piece,
                "source": doc_title,
                "section": header,
                "chunk_id": f"{doc_title}::{chunk_idx}",
            })
            chunk_idx += 1
    return chunks
