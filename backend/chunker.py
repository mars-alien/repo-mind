"""
Content-type aware text chunker for PageMind.

Three strategies dispatched by content_type:

  docs    → heading-prefixed chunks
            Every chunk is prefixed with its full breadcrumb path so the
            embedding carries navigational context:
              [FastAPI > Request Body > Multiple Parameters]
              You can declare multiple body parameters…

  wiki    → section-aware chunks, boilerplate sections skipped
            Prefix: [Article Title › Section Name]
            Sections like "References", "See also" are discarded.

  general → sentence-aware overlapping chunks (fallback)
            Same algorithm as before, plus title prefix.

sections parameter (from browser DOM extraction):
  [{"heading": "Installation", "path": "Guide > Installation", "text": "…"}, …]
"""

import re

CHUNK_SIZE    = 800   # target characters per chunk
CHUNK_OVERLAP = 150   # tail carried into the next chunk

# Wikipedia / wiki sections to discard entirely
_WIKI_SKIP = {
    "references", "external links", "see also", "notes",
    "further reading", "bibliography", "footnotes", "citations",
    "sources", "works cited", "external resources",
}


# ── Public entry point ─────────────────────────────────────────────────────────

def chunk_by_type(
    text:         str,
    source:       str       = "",
    content_type: str       = "general",
    title:        str       = "",
    sections:     list[dict] | None = None,
) -> list[dict]:
    """
    Main dispatcher.

    Parameters
    ----------
    text         : flat page text (used as fallback when sections is None)
    source       : page URL, stored in each chunk payload
    content_type : "docs" | "wiki" | "general"
    title        : page title — prepended to every chunk as context anchor
    sections     : structured sections extracted from the DOM by the browser
                   Each item: {"heading": str, "path": str, "text": str}

    Returns
    -------
    list of dicts:
      {text, index, source, heading_path, content_type}
    """
    if content_type == "docs" and sections:
        return _docs_chunks(sections, source, title, content_type)
    if content_type == "wiki" and sections:
        return _wiki_chunks(sections, source, title, content_type)
    # General fallback — also used when sections is empty/None
    return _general_chunks(text, source, title, content_type)


# ── Docs chunker ───────────────────────────────────────────────────────────────

def _docs_chunks(
    sections:     list[dict],
    source:       str,
    title:        str,
    content_type: str,
) -> list[dict]:
    """
    Bake the full heading breadcrumb into every chunk so the embedding
    'knows' exactly which section of the docs it came from.

    Example prefix:
      [React > Hooks > useEffect]
    """
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

    # If the structured extraction yielded nothing, fall back to general
    if not chunks:
        return _general_chunks(
            "\n\n".join(s.get("text", "") for s in sections),
            source, title, content_type,
        )
    return chunks


# ── Wiki chunker ───────────────────────────────────────────────────────────────

def _wiki_chunks(
    sections:     list[dict],
    source:       str,
    title:        str,
    content_type: str,
) -> list[dict]:
    """
    Skip boilerplate sections (References, See also, etc.) and prefix
    each chunk with [Article › Section] for clear attribution.
    """
    chunks: list[dict] = []

    for section in sections:
        heading = section.get("heading", "")

        # Drop noise sections
        if heading.lower().strip() in _WIKI_SKIP:
            continue

        separator = " › " if heading else ""
        prefix = f"[{title}{separator}{heading}]\n"

        for raw in _split_text(section.get("text", ""), source):
            chunks.append({
                "text":         prefix + raw["text"],
                "index":        len(chunks),
                "source":       source,
                "heading_path": heading,
                "content_type": content_type,
            })

    if not chunks:
        return _general_chunks(
            "\n\n".join(s.get("text", "") for s in sections),
            source, title, content_type,
        )
    return chunks


# ── General chunker ────────────────────────────────────────────────────────────

def _general_chunks(
    text:         str,
    source:       str,
    title:        str,
    content_type: str,
) -> list[dict]:
    """
    Sentence-aware overlapping chunks with a title prefix.
    The title prefix anchors the embedding to the document topic even
    when the chunk is taken out of context.
    """
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


# ── Core sentence-aware splitter ───────────────────────────────────────────────

def _split_text(text: str, source: str = "") -> list[dict]:
    """
    Split text into overlapping, sentence-boundary-respecting chunks.
    No prefix is applied here — callers add their own prefix.

    Returns [{text, index, source}].
    """
    cleaned = _clean(text)
    if not cleaned:
        return []

    # Split on sentence terminators while keeping delimiters
    sentences = re.split(r"(?<=[.!?\n])\s+", cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[dict] = []
    current = ""
    overlap = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > CHUNK_SIZE:
            if current.strip():
                chunks.append({
                    "text":   current.strip(),
                    "index":  len(chunks),
                    "source": source,
                })
                overlap = current[-CHUNK_OVERLAP:].strip()
            current = (overlap + " " + sentence).strip()
            overlap = ""
        else:
            current = (current + (" " if current else "") + sentence)

    if current.strip():
        chunks.append({
            "text":   current.strip(),
            "index":  len(chunks),
            "source": source,
        })

    return chunks


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+",  " ",  text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()

