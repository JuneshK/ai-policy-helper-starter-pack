import os
import re
import hashlib
from typing import List, Dict, Tuple
from .settings import settings


# ---------------- File Reading ---------------- #
def _read_text_file(path: str) -> str:
    """Read a text or markdown file safely."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------- Markdown Sections ---------------- #
def _md_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split Markdown text into sections based on headings.
    Returns a list of (section_title, section_text).
    """
    parts = re.split(r"\n(?=#+\s)", text)
    out = []

    for p in parts:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else "Body"
        out.append((title, p))

    return out or [("Body", text)]


# ---------------- Chunking ---------------- #
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks of tokens.
    """
    tokens = text.split()
    chunks = []
    i = 0

    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap

    return chunks


# ---------------- Document Loader ---------------- #
def load_documents(data_dir: str) -> List[Dict]:
    """
    Load all .md and .txt files from a directory and split into sections.
    """
    docs = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        text = _read_text_file(path)
        for section, body in _md_sections(text):
            docs.append({
                "title": fname,
                "section": section,
                "text": body
            })

    return docs


# ---------------- Document Hash ---------------- #
def doc_hash(text: str) -> str:
    """Return a SHA256 hash of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
