"""
PageMind URL Fetcher
────────────────────
Given any public URL, this module:
  1. Fetches HTML using requests (real browser User-Agent, 15 s timeout)
  2. Detects content type: "wiki" | "docs" | "general"
  3. Extracts structured sections with heading-breadcrumb paths
       docs  → main/article container, full h2 > h3 > h4 breadcrumb
       wiki  → #mw-content-text, skips boilerplate (References, See also …)
       general → flat text walk of main content area
  4. Returns a dict compatible with chunker.chunk_by_type()

Output schema
─────────────
{
    text:         str,                 # flat joined text (used as general fallback)
    title:        str,
    url:          str,
    content_type: "wiki"|"docs"|"general",
    sections:     list[dict] | None,   # [{heading, path, text}]
}
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ── HTTP config ────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT = 15   # seconds

# ── Skip these wiki sections ───────────────────────────────────────────────────

_WIKI_SKIP = {
    "references", "external links", "see also", "notes",
    "further reading", "bibliography", "footnotes", "citations",
    "sources", "works cited", "external resources",
}

# ── HTML element categories ───────────────────────────────────────────────────

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT_TAGS    = {"p", "li", "dt", "dd", "blockquote", "td", "th", "pre"}
_WALK_TAGS    = {
    "div", "section", "article", "main", "ul", "ol", "dl",
    "table", "tbody", "tr", "figure", "details", "summary",
}

# Elements whose entire subtree we discard
_NOISE_TAGS = {
    "script", "style", "noscript", "nav", "footer",
    "header", "form", "button", "input", "select",
    "iframe", "svg", "canvas", "aside",
}

# Class / role patterns that mark noise containers
_NOISE_CLS = re.compile(
    r"\b(nav(bar)?|sidebar|menu|toc|table-of-contents|breadcrumb|"
    r"cookie|footer|header|banner|popup|modal|overlay|"
    r"ad(vert(isement)?)?|social(-share)?|share|"
    r"comment|relat(ed|ions)|recommend|pagination|search-form)\b",
    re.I,
)

_NOISE_ROLES = {"navigation", "banner", "complementary", "contentinfo", "search"}


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_and_extract(url: str) -> dict:
    """
    Fetch *url* and return structured content ready for chunk_by_type().

    Raises
    ------
    requests.exceptions.HTTPError      on 4xx / 5xx responses
    requests.exceptions.ConnectionError on network failure
    requests.exceptions.Timeout         on timeout (15 s)
    ValueError                          if the response is not HTML
    """
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT,
                        allow_redirects=True)
    resp.raise_for_status()

    content_type_header = resp.headers.get("Content-Type", "")
    if "html" not in content_type_header and "xml" not in content_type_header:
        raise ValueError(
            f"Unsupported content-type: {content_type_header}. "
            "Only HTML pages are supported."
        )

    soup = BeautifulSoup(resp.text, "lxml")

    # Nuke noise at the root before we do anything else
    for el in soup.find_all(_NOISE_TAGS):
        el.decompose()

    title        = _get_title(soup, url)
    content_type = _detect_type(url, soup)

    if content_type in ("docs", "wiki"):
        sections = _extract_sections(soup, content_type)
        text     = "\n\n".join(s["text"] for s in sections)
        # If structured extraction yielded nothing, fall back to flat
        if not text.strip():
            sections     = None
            content_type = "general"
            text         = _extract_flat(soup)
    else:
        sections = None
        text     = _extract_flat(soup)

    return {
        "text":         text,
        "title":        title,
        "url":          url,
        "content_type": content_type,
        "sections":     sections,
    }


# ── Title extraction ───────────────────────────────────────────────────────────

def _get_title(soup: BeautifulSoup, url: str) -> str:
    """Extract the cleanest title, stripping site-name suffixes."""
    _SUFFIX = re.compile(r"\s*[|–—\-]\s*.{1,60}$")

    # OG title (use attrs= to avoid Python built-in collision)
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content", "").strip():
        t = _SUFFIX.sub("", og["content"].strip()).strip()
        if t:
            return t

    # <title> tag
    if soup.title:
        raw = soup.title.get_text(strip=True)
        raw = _SUFFIX.sub("", raw).strip()
        if raw:
            return raw

    # First h1
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return url


# ── Content-type detection ────────────────────────────────────────────────────

def _detect_type(url: str, soup: BeautifulSoup) -> str:
    try:
        parsed   = urlparse(url)
        hostname = parsed.netloc.lower()
        path     = parsed.path.lower()
    except Exception:
        hostname = ""
        path     = ""

    # ── Wikipedia and MediaWiki sites ──
    if "wikipedia.org" in hostname or soup.find(id="mw-content-text"):
        return "wiki"

    # ── Known docs hostnames ──
    _docs_host = re.compile(
        r"^docs?\.|"
        r"developer\.|"
        r"readthedocs\.io$|"
        r"\.github\.io$|"
        r"devdocs\.|"
        r"^api\.",
        re.I,
    )
    if _docs_host.search(hostname):
        return "docs"

    # ── Docs URL paths ──
    _docs_path = re.compile(
        r"/docs?/|"
        r"/api(?:/|$)|"
        r"/guide(?:s)?(?:/|$)|"
        r"/reference(?:/|$)|"
        r"/manual(?:/|$)|"
        r"/tutorial(?:s)?(?:/|$)|"
        r"/getting.started|"
        r"/quickstart",
        re.I,
    )
    if _docs_path.search(path):
        return "docs"

    # ── Docs by DOM signals (many code blocks + headings) ──
    n_code    = len(soup.find_all(["pre", "code"]))
    n_heading = len(soup.find_all(["h2", "h3", "h4"]))
    if n_code >= 5 and n_heading >= 5:
        return "docs"

    return "general"


# ── Noise filter ──────────────────────────────────────────────────────────────

def _is_noise(el: Tag) -> bool:
    """Return True if this element is navigational / decorative noise."""
    if el.name in _NOISE_TAGS:
        return True
    cls = " ".join(el.get("class", []))
    if _NOISE_CLS.search(cls):
        return True
    role = (el.get("role") or "").lower()
    if role in _NOISE_ROLES:
        return True
    label = (el.get("aria-label") or "").lower()
    if any(w in label for w in ("navigation", "sidebar", "footer", "ad")):
        return True
    return False


# ── Content root ──────────────────────────────────────────────────────────────

def _find_root(soup: BeautifulSoup, content_type: str) -> Tag:
    """Return the primary content container for this page."""
    if content_type == "wiki":
        for sel in [
            "#mw-content-text .mw-parser-output",
            "#mw-content-text",
            "#content",
        ]:
            el = soup.select_one(sel)
            if el:
                return el

    # Docs / general — prefer semantic landmarks with enough text
    for sel in [
        "main",
        "article",
        "[role='main']",
        ".content",
        "#content",
        ".docs-content",
        "#docs-content",
        ".documentation",
        ".main-content",
        "#main-content",
        ".markdown-body",     # GitHub READMEs / pages
        ".rst-content",       # ReadTheDocs
        ".bd-content",        # Sphinx Bootstrap Theme
    ]:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 200:
            return el

    return soup.body or soup


# ── Structured extraction (docs / wiki) ──────────────────────────────────────

def _extract_sections(soup: BeautifulSoup, content_type: str) -> list[dict]:
    """
    Walk the content root and build heading-breadcrumb sections.
    Returns [{heading, path, text}] — same shape the browser extension produces.
    """
    root = _find_root(soup, content_type)

    sections: list[dict] = []
    heading_stack: list[dict] = []   # [{level: int, text: str}]
    cur_heading = ""
    cur_path    = ""
    cur_lines:  list[str] = []

    def build_path(level: int, text: str) -> str:
        while heading_stack and heading_stack[-1]["level"] >= level:
            heading_stack.pop()
        heading_stack.append({"level": level, "text": text})
        return " > ".join(h["text"] for h in heading_stack)

    def flush() -> None:
        joined = " ".join(cur_lines).strip()
        if len(joined) < 40:
            return
        if content_type == "wiki" and cur_heading.lower().strip() in _WIKI_SKIP:
            return
        sections.append({"heading": cur_heading, "path": cur_path, "text": joined})

    def walk(node: Tag) -> None:
        nonlocal cur_heading, cur_path, cur_lines
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue

            tag = (child.name or "").lower()
            if not tag:
                continue
            if _is_noise(child):
                continue

            if tag in _HEADING_TAGS:
                flush()
                cur_lines = []
                level = int(tag[1])
                h_text = re.sub(r"\s+", " ",
                    child.get_text(separator=" ", strip=True)
                         .replace("[edit]", "")
                         .replace("[source]", "")
                         .strip()
                )
                cur_heading = h_text
                cur_path    = build_path(level, h_text)

            elif tag in _TEXT_TAGS:
                # Avoid double-counting: skip container li/p that nest other text tags
                if tag in ("li", "p") and child.find(list(_TEXT_TAGS - {tag})):
                    walk(child)
                    continue
                t = re.sub(r"\s+", " ", child.get_text(separator=" ", strip=True))
                if len(t) > 15:
                    cur_lines.append(t)

            elif tag in _WALK_TAGS:
                walk(child)

    walk(root)
    flush()
    return sections


# ── Flat extraction (general) ─────────────────────────────────────────────────

def _extract_flat(soup: BeautifulSoup) -> str:
    """Plain-text walk for general web pages."""
    root = _find_root(soup, "general")
    lines: list[str] = []

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                t = child.strip()
                if len(t) > 30:
                    lines.append(t)
                continue
            if not isinstance(child, Tag):
                continue
            if _is_noise(child):
                continue
            tag = (child.name or "").lower()
            if tag in _TEXT_TAGS | _HEADING_TAGS:
                t = re.sub(r"\s+", " ", child.get_text(separator=" ", strip=True))
                if len(t) > 30:
                    lines.append(t)
            elif tag:
                walk(child)

    walk(root)
    return "\n".join(lines)
