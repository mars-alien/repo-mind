"""
GitHub Repository Fetcher for PageMind
───────────────────────────────────────
Given a GitHub repo URL (https://github.com/owner/repo) this module:
  1. Parses owner/repo from the URL
  2. Fetches the full file tree via GitHub API:
       GET /repos/{owner}/{repo}/git/trees/HEAD?recursive=1
  3. Filters to indexable file types (code + docs), skips noise dirs
  4. Fetches raw content via raw.githubusercontent.com (no auth needed
     for public repos)
  5. Returns [{filepath, language, content, repo, url}]

Rate limits
───────────
  Unauthenticated: 60 req/h  — capped at 55 files max
  With PAT:       5000 req/h  — capped at 200 files max

Output schema (per file)
─────────────────────────
{
    filepath : str   e.g. "src/auth/middleware.py"
    language : str   e.g. "python"
    content  : str   raw file text
    repo     : str   e.g. "owner/repo"
    url      : str   GitHub web URL for this file
}
"""

from __future__ import annotations

import re
import time
from urllib.parse import urlparse

import requests

_HEADERS = {
    "User-Agent": "PageMind/1.0 (github-rag-indexer)",
    "Accept":     "application/vnd.github+json",
}

_TIMEOUT          = 15     # seconds per request
_MAX_FILE_BYTES   = 400_000  # 400 KB — skip large generated files
_MAX_FILES_UNAUTH = 55     # conservative limit without a token
_MAX_FILES_AUTH   = 200    # with a personal access token

# File extension → canonical language name
_LANG_MAP: dict[str, str] = {
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "typescript",
    ".jsx":   "javascript",
    ".tsx":   "typescript",
    ".java":  "java",
    ".go":    "go",
    ".rs":    "rust",
    ".cpp":   "cpp",
    ".cc":    "cpp",
    ".c":     "c",
    ".h":     "c",
    ".hpp":   "cpp",
    ".cs":    "csharp",
    ".rb":    "ruby",
    ".php":   "php",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".md":    "markdown",
    ".mdx":   "markdown",
    ".rst":   "rst",
    ".txt":   "text",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".json":  "json",
    ".toml":  "toml",
    ".sh":    "shell",
    ".bash":  "shell",
}

# Directories / path segments to skip entirely
_SKIP_RE = re.compile(
    r"(^|/)("
    r"node_modules|\.git|__pycache__|\.venv|venv|env|"
    r"dist|build|\.next|\.nuxt|\.cache|coverage|"
    r"\.pytest_cache|\.mypy_cache|\.ruff_cache|"
    r"\.eggs|\.tox|htmlcov|site-packages"
    r")/",
    re.I,
)

# Files always included regardless of extension
_ALWAYS_INCLUDE = re.compile(
    r"^(readme|license|contributing|changelog|authors|notice|security|code_of_conduct)\b",
    re.I,
)

# File types that are never useful
_NEVER_INCLUDE = re.compile(
    r"\.(png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|otf|"
    r"zip|tar|gz|rar|7z|exe|dll|so|dylib|"
    r"lock|map|min\.js|min\.css|bundle\.js|"
    r"pdf|docx|xlsx|pptx|db|sqlite)$",
    re.I,
)

# Specific filenames that are always noise (lock files, generated manifests, CI config)
_NEVER_INCLUDE_NAMES = frozenset({
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
    "go.sum",           # Go dependency checksums — not human-readable
    "uv.lock",
    ".DS_Store",
    "render.yaml",      # deployment config, no code value
    "fly.toml",
    "vercel.json",
    "netlify.toml",
    "railway.json",
})

# GitHub pages that aren't repo roots
_NON_REPO_OWNERS = frozenset({
    "login", "settings", "explore", "marketplace", "topics",
    "notifications", "pulls", "issues", "orgs", "sponsors",
    "features", "pricing", "about",
})


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_repo_url(url: str) -> tuple[str, str]:
    """
    Extract (owner, repo) from a GitHub URL.

    Accepts:
      https://github.com/owner/repo
      https://github.com/owner/repo/tree/main/src
      https://github.com/owner/repo.git

    Raises ValueError if not a valid GitHub repo URL.
    """
    parsed = urlparse(url.strip())
    if "github.com" not in parsed.netloc.lower():
        raise ValueError(f"Not a GitHub URL: {url!r}")

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse owner/repo from: {url!r}")

    owner = parts[0].lower()
    repo  = parts[1].removesuffix(".git")

    if owner in _NON_REPO_OWNERS:
        raise ValueError(f"Not a repository URL: {url!r}")

    return owner, repo


def fetch_repo(
    repo_url:     str,
    github_token: str | None = None,
) -> list[dict]:
    """
    Fetch all indexable files from a public GitHub repository.

    Parameters
    ----------
    repo_url     : Full GitHub URL  (e.g. https://github.com/owner/repo)
    github_token : Optional GitHub PAT — raises rate limit from 60 to 5000 req/h

    Returns
    -------
    list[dict] — one entry per file:
      { filepath, language, content, repo, url }

    Raises
    ------
    ValueError                          — invalid URL or private/missing repo
    requests.exceptions.HTTPError       — API error (rate limit, etc.)
    requests.exceptions.Timeout         — request exceeded 15 s
    """
    owner, repo = parse_repo_url(repo_url)
    repo_slug   = f"{owner}/{repo}"

    headers = dict(_HEADERS)
    if github_token and github_token.strip():
        headers["Authorization"] = f"Bearer {github_token.strip()}"
        max_files = _MAX_FILES_AUTH
    else:
        max_files = _MAX_FILES_UNAUTH

    # ── Step 1: Fetch full file tree ─────────────────────────────────────────
    tree_url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/git/trees/HEAD?recursive=1"
    )
    resp = requests.get(tree_url, headers=headers, timeout=_TIMEOUT)

    if resp.status_code == 404:
        raise ValueError(
            f"Repository '{repo_slug}' not found or is private.\n"
            "For private repos, provide a GitHub personal access token."
        )
    resp.raise_for_status()

    tree_data = resp.json()
    all_blobs = [
        item for item in tree_data.get("tree", [])
        if item.get("type") == "blob"
    ]

    if tree_data.get("truncated"):
        print(f"[github] {repo_slug}: tree truncated (> 100K files) — using first batch")

    # ── Step 2: Filter to indexable files ────────────────────────────────────
    indexable: list[dict] = []
    for blob in all_blobs:
        path = blob.get("path", "")
        size = blob.get("size") or 0

        if _SKIP_RE.search(path + "/"):
            continue
        if _NEVER_INCLUDE.search(path):
            continue
        if size > _MAX_FILE_BYTES:
            continue

        fname = path.rsplit("/", 1)[-1]
        if fname in _NEVER_INCLUDE_NAMES:
            continue
        ext   = ("." + fname.rsplit(".", 1)[-1].lower()) if "." in fname else ""

        language = _LANG_MAP.get(ext)
        if language is None and not _ALWAYS_INCLUDE.match(fname):
            continue

        indexable.append({
            "path":     path,
            "language": language or "text",
            "size":     size,
        })

    # Sort: code first (more useful for Q&A), then docs, then config
    def _priority(item: dict) -> int:
        lang = item["language"]
        if lang in ("python", "javascript", "typescript", "go", "rust", "java",
                    "csharp", "cpp", "c", "kotlin", "swift", "ruby", "php", "shell"):
            return 0
        if lang in ("markdown", "rst", "text"):
            return 1
        return 2

    indexable.sort(key=_priority)
    indexable = indexable[:max_files]

    print(
        f"[github] {repo_slug}: {len(all_blobs)} blobs → "
        f"{len(indexable)} selected (limit {max_files})"
    )

    # ── Step 3: Fetch raw file content ────────────────────────────────────────
    raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD"
    results:  list[dict] = []

    for i, item in enumerate(indexable):
        path    = item["path"]
        raw_url = f"{raw_base}/{path}"
        try:
            r = requests.get(raw_url, headers=headers, timeout=_TIMEOUT)
            r.raise_for_status()
            content = r.text
            if not content.strip():
                continue
            results.append({
                "filepath": path,
                "language": item["language"],
                "content":  content,
                "repo":     repo_slug,
                "url":      f"https://github.com/{owner}/{repo}/blob/HEAD/{path}",
            })
        except Exception as exc:
            print(f"[github] skip {path}: {exc}")
            continue

        # Light throttle every 10 files to stay well under rate limits
        if (i + 1) % 10 == 0:
            time.sleep(0.05)

    print(f"[github] {repo_slug}: fetched {len(results)} files")
    return results
