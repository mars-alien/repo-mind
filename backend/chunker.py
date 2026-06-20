"""
Content-type aware chunker — PageMind v2
════════════════════════════════════════
Two separate strategies:

A) Web-page chunking  (chunk_by_type)
   docs  → heading-breadcrumb prefixed chunks
   wiki  → section-aware, boilerplate discarded
   general → sentence-aware overlapping windows

B) Code-aware chunking  (chunk_code)  ← GitHub RAG
   Python  → ast.parse() — exact function/class boundaries + line numbers
   JS/TS   → regex FSM  — function/class/arrow-function boundaries + line numbers
   Other   → line-aware sliding window with 3-line overlap

Every chunk produced by chunk_code carries:
  text         str  — "[filepath | chunk_type name]\\n<code>"
  filepath     str  — relative path in repo
  language     str  — canonical language string
  start_line   int  — 1-indexed, first line of this block in the file
  end_line     int  — 1-indexed, last  line of this block in the file
  chunk_type   str  — "function" | "class" | "module" | "block"
  heading_path str  — "filepath | chunk_type name"  (used for BM25 boost)
  source       str  — GitHub blob URL
  index        int  — position within this file's chunks
"""

from __future__ import annotations

import ast
import re

# ── Web-page chunker constants ────────────────────────────────────────────────

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

_WIKI_SKIP = {
    "references", "external links", "see also", "notes",
    "further reading", "bibliography", "footnotes", "citations",
    "sources", "works cited", "external resources",
}

# ── Code chunker constants ────────────────────────────────────────────────────

_MAX_BLOCK_CHARS = 1_400   # split blocks longer than this
_MIN_CHUNK_CHARS = 60      # discard trivially short blocks
_OVERLAP_LINES   = 3       # lines carried from previous chunk when splitting


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════════

def chunk_by_type(
    text:         str,
    source:       str       = "",
    content_type: str       = "general",
    title:        str       = "",
    sections:     list[dict] | None = None,
) -> list[dict]:
    """Web-page chunker dispatcher (docs / wiki / general)."""
    if content_type == "docs" and sections:
        return _docs_chunks(sections, source, title, content_type)
    if content_type == "wiki" and sections:
        return _wiki_chunks(sections, source, title, content_type)
    return _general_chunks(text, source, title, content_type)


def chunk_code(
    content:  str,
    filepath: str,
    language: str,
    repo:     str = "",
) -> list[dict]:
    """
    Code-aware chunker for GitHub repo files.

    Dispatches to language-specific splitters that preserve function/class
    boundaries and track exact line numbers for citation links.

    Parameters
    ----------
    content  : raw source file text
    filepath : relative path in the repo  e.g. "src/auth/middleware.py"
    language : canonical string           e.g. "python", "javascript"
    repo     : "owner/repo" slug          used to build GitHub blob URLs

    Returns
    -------
    list[dict]  each dict has:
      text, filepath, language, start_line, end_line, chunk_type,
      heading_path, source, index, content_type
    """
    source = (
        f"https://github.com/{repo}/blob/HEAD/{filepath}"
        if repo else filepath
    )

    if language == "python":
        chunks = _python_ast_chunks(content, filepath, source)
    elif language in ("javascript", "typescript"):
        chunks = _js_regex_chunks(content, filepath, language, source)
    else:
        chunks = _code_sliding_window(content, filepath, language, source)

    if not chunks:
        chunks = _code_sliding_window(content, filepath, language, source)

    for i, c in enumerate(chunks):
        c["index"] = i
        c["content_type"] = language

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Python AST-aware chunker
# ══════════════════════════════════════════════════════════════════════════════

def _python_ast_chunks(content: str, filepath: str, source: str) -> list[dict]:
    """
    Use Python's ast module to split at exact function/class boundaries.
    Produces chunks with precise start_line / end_line from the AST nodes.
    Falls back to sliding window on SyntaxError.
    """
    lines = content.splitlines()
    chunks: list[dict] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _code_sliding_window(content, filepath, "python", source)

    # Top-level definitions only (col_offset == 0)
    top_level = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    # ── Preamble: imports + module docstring ──────────────────────────────────
    preamble_end = (top_level[0].lineno - 1) if top_level else len(lines)
    preamble_text = "\n".join(lines[:preamble_end]).strip()
    if len(preamble_text) >= _MIN_CHUNK_CHARS:
        heading = f"{filepath} | module"
        for sub in _split_by_lines(preamble_text, source):
            chunks.append(_make_code_chunk(
                sub, heading, filepath, "python", "module",
                1, preamble_end, source,
            ))

    # ── Each top-level function / class ───────────────────────────────────────
    for node in top_level:
        start = node.lineno                           # 1-indexed
        end   = getattr(node, "end_lineno", start)   # 3.8+
        block = "\n".join(lines[start - 1 : end])

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        name       = node.name
        heading    = f"{filepath} | {chunk_type} {name}"

        for sub in _split_by_lines(block, source):
            chunks.append(_make_code_chunk(
                sub, heading, filepath, "python", chunk_type,
                start, end, source,
            ))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# JavaScript / TypeScript regex chunker
# ══════════════════════════════════════════════════════════════════════════════

# Matches the opening line of a top-level function / class / const-arrow.
# Group 1 = "function" | "class"  (named); Group 2 = const/arrow name
_JS_DEF_RE = re.compile(
    r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?"
    r"(?:"
    r"(function\*?|class)\s+(\w+)"                         # function foo / class Foo
    r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function\*?|\(|[\w]+\s*=>)"  # const foo =
    r")",
    re.MULTILINE,
)


def _js_regex_chunks(
    content: str,
    filepath: str,
    language: str,
    source: str,
) -> list[dict]:
    """Regex-based splitter for JS/TS files tracking line numbers."""
    lines   = content.splitlines()
    chunks: list[dict] = []
    matches = list(_JS_DEF_RE.finditer(content))

    if not matches:
        return _code_sliding_window(content, filepath, language, source)

    # Preamble
    first_match_line = content[: matches[0].start()].count("\n") + 1
    preamble = "\n".join(lines[: first_match_line - 1]).strip()
    if len(preamble) >= _MIN_CHUNK_CHARS:
        heading = f"{filepath} | module"
        for sub in _split_by_lines(preamble, source):
            chunks.append(_make_code_chunk(
                sub, heading, filepath, language, "module",
                1, first_match_line - 1, source,
            ))

    for i, match in enumerate(matches):
        # Determine name
        if match.group(2):                  # function foo / class Foo
            kw   = match.group(1)
            name = match.group(2)
        elif match.group(3):                # const foo = ...
            kw   = "const"
            name = match.group(3)
        else:
            kw, name = "function", "anonymous"

        chunk_type  = "class" if kw == "class" else "function"
        start_line  = content[: match.start()].count("\n") + 1

        if i + 1 < len(matches):
            end_pos  = matches[i + 1].start()
        else:
            end_pos  = len(content)

        end_line = content[:end_pos].count("\n") + 1
        block    = content[match.start() : end_pos].strip()

        heading = f"{filepath} | {chunk_type} {name}"
        for sub in _split_by_lines(block, source):
            chunks.append(_make_code_chunk(
                sub, heading, filepath, language, chunk_type,
                start_line, end_line, source,
            ))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Generic code sliding-window (Markdown, YAML, JSON, Go, Rust, etc.)
# ══════════════════════════════════════════════════════════════════════════════

def _code_sliding_window(
    content:  str,
    filepath: str,
    language: str,
    source:   str,
) -> list[dict]:
    """
    Line-aware sliding window for languages without AST/regex support.
    Never breaks mid-line. Carries _OVERLAP_LINES of context forward.
    """
    lines   = content.splitlines()
    heading = f"{filepath}"
    chunks: list[dict] = []

    i = 0
    while i < len(lines):
        block_lines: list[str] = []
        chars = 0

        while i < len(lines) and chars + len(lines[i]) + 1 <= _MAX_BLOCK_CHARS:
            block_lines.append(lines[i])
            chars += len(lines[i]) + 1
            i += 1

        block = "\n".join(block_lines).strip()
        if len(block) >= _MIN_CHUNK_CHARS:
            start = i - len(block_lines) + 1
            end   = i
            chunks.append(_make_code_chunk(
                block, heading, filepath, language, "block",
                start, end, source,
            ))

        # Overlap: step back a few lines
        i = max(i - _OVERLAP_LINES, i - len(block_lines) + max(len(block_lines) - _OVERLAP_LINES, 1))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_code_chunk(
    text_or_sub,      # str or dict with "text" key
    heading:  str,
    filepath: str,
    language: str,
    chunk_type: str,
    start_line: int,
    end_line:   int,
    source:     str,
) -> dict:
    """Build a chunk dict with all required metadata fields."""
    text = text_or_sub if isinstance(text_or_sub, str) else text_or_sub["text"]
    return {
        "text":         f"[{heading}]\n{text}",
        "filepath":     filepath,
        "language":     language,
        "start_line":   start_line,
        "end_line":     end_line,
        "chunk_type":   chunk_type,
        "heading_path": heading,
        "source":       source,
        "index":        0,          # set by caller
    }


def _split_by_lines(text: str, source: str) -> list[str]:
    """
    Split a code block into sub-chunks of at most _MAX_BLOCK_CHARS.
    Never splits mid-line; carries _OVERLAP_LINES forward.
    """
    if len(text) <= _MAX_BLOCK_CHARS:
        return [text.strip()] if text.strip() else []

    lines    = text.splitlines()
    subs:    list[str] = []
    current: list[str] = []
    chars    = 0

    for line in lines:
        line_len = len(line) + 1
        if chars + line_len > _MAX_BLOCK_CHARS and current:
            chunk = "\n".join(current).strip()
            if len(chunk) >= _MIN_CHUNK_CHARS:
                subs.append(chunk)
            current = current[-_OVERLAP_LINES:]  # overlap
            chars   = sum(len(l) + 1 for l in current)
        current.append(line)
        chars += line_len

    if current:
        chunk = "\n".join(current).strip()
        if len(chunk) >= _MIN_CHUNK_CHARS:
            subs.append(chunk)

    return subs if subs else [text[:_MAX_BLOCK_CHARS].strip()]


# ══════════════════════════════════════════════════════════════════════════════
# Web-page chunkers (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

def _docs_chunks(sections, source, title, content_type):
    chunks: list[dict] = []
    for section in sections:
        heading_path = section.get("path") or section.get("heading", "")
        prefix = f"[{heading_path}]\n" if heading_path else (f"[{title}]\n" if title else "")
        for raw in _split_text(section.get("text", ""), source):
            chunks.append({
                "text":         prefix + raw["text"],
                "index":        len(chunks),
                "source":       source,
                "heading_path": heading_path,
                "content_type": content_type,
            })
    if not chunks:
        return _general_chunks("\n\n".join(s.get("text", "") for s in sections),
                               source, title, content_type)
    return chunks


def _wiki_chunks(sections, source, title, content_type):
    chunks: list[dict] = []
    for section in sections:
        heading = section.get("heading", "")
        if heading.lower().strip() in _WIKI_SKIP:
            continue
        sep    = " › " if heading else ""
        prefix = f"[{title}{sep}{heading}]\n"
        for raw in _split_text(section.get("text", ""), source):
            chunks.append({
                "text":         prefix + raw["text"],
                "index":        len(chunks),
                "source":       source,
                "heading_path": heading,
                "content_type": content_type,
            })
    if not chunks:
        return _general_chunks("\n\n".join(s.get("text", "") for s in sections),
                               source, title, content_type)
    return chunks


def _general_chunks(text, source, title, content_type):
    prefix = f"[{title}]\n" if title else ""
    chunks = []
    for raw in _split_text(text, source):
        chunks.append({
            "text":         prefix + raw["text"],
            "index":        len(chunks),
            "source":       source,
            "heading_path": "",
            "content_type": content_type,
        })
    return chunks


def _split_text(text: str, source: str = "") -> list[dict]:
    """Sentence-aware overlapping splitter for prose content."""
    cleaned   = _clean(text)
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?\n])\s+", cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: list[dict] = []
    current = ""
    overlap = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > CHUNK_SIZE:
            if current.strip():
                chunks.append({"text": current.strip(), "index": len(chunks), "source": source})
                overlap = current[-CHUNK_OVERLAP:].strip()
            current = (overlap + " " + sentence).strip()
            overlap = ""
        else:
            current = (current + (" " if current else "") + sentence)
    if current.strip():
        chunks.append({"text": current.strip(), "index": len(chunks), "source": source})
    return chunks


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+",   " ",  text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()
