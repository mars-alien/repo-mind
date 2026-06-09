/**
 * PageMind API client
 * ───────────────────
 * Centralises all communication with the FastAPI backend.
 * JWT token is persisted in chrome.storage.local and attached
 * as a Bearer header to every authenticated request.
 */

const BASE = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

// ── Token storage ─────────────────────────────────────────────────────────────

export async function getToken() {
  const { pm_token } = await chrome.storage.local.get("pm_token");
  return pm_token || null;
}

export async function getUsername() {
  const { pm_username } = await chrome.storage.local.get("pm_username");
  return pm_username || null;
}

async function saveSession(token, username) {
  await chrome.storage.local.set({ pm_token: token, pm_username: username });
}

export async function clearSession() {
  await chrome.storage.local.remove(["pm_token", "pm_username"]);
}

// ── Base fetch ────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}, requireAuth = true) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };

  if (requireAuth) {
    const token = await getToken();
    if (!token) throw new AuthError("Not authenticated");
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  if (res.status === 401) {
    await clearSession();
    throw new AuthError("Session expired. Please log in again.");
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {}
    throw new Error(detail);
  }

  return res;
}

// Custom error class so the UI can detect auth failures
export class AuthError extends Error {
  constructor(msg) { super(msg); this.name = "AuthError"; }
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function register(username, password) {
  const res  = await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  }, false);
  const data = await res.json();
  await saveSession(data.access_token, data.username);
  return data;
}

export async function login(username, password) {
  const res  = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  }, false);
  const data = await res.json();
  await saveSession(data.access_token, data.username);
  return data;
}

// ── Ingest ────────────────────────────────────────────────────────────────────

/**
 * Send page text to the backend for chunking + embedding + Weaviate storage.
 * @param {string}      text         raw page text
 * @param {string}      title        page title
 * @param {string}      url          page URL
 * @param {string}      sourceType   "webpage" | "pdf" | "note"
 * @param {string}      contentType  "docs" | "wiki" | "general"
 * @param {Array|null}  sections     structured sections from DOM (docs/wiki only)
 * @returns {{ doc_id, chunks_stored, status, content_type }}
 */
export async function ingestPage(
  text,
  title,
  url,
  sourceType   = "webpage",
  contentType  = "general",
  sections     = null,
) {
  const res = await apiFetch("/ingest", {
    method: "POST",
    body: JSON.stringify({
      text,
      title,
      url,
      source_type:  sourceType,
      content_type: contentType,
      sections:     sections,
    }),
  });
  return res.json();
}

// ── KB — server-side URL ingestion ───────────────────────────────────────────

/**
 * Add any public URL to the knowledge base.
 * The backend fetches the URL, extracts content, detects type (docs/wiki/general),
 * chunks, embeds and stores it in Weaviate — no browser tab required.
 *
 * @param {string} url  Fully-qualified URL to fetch and index
 * @returns {{ doc_id, chunks_stored, status, content_type, title }}
 */
export async function addUrlToKB(url) {
  const res = await apiFetch("/kb/add", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
  return res.json();
}

// ── Query (SSE streaming) ─────────────────────────────────────────────────────

/**
 * Send a question to the backend. Calls onChunk for each text token,
 * onSources once at the end with citation metadata, and resolves when done.
 *
 * @param {string}        question
 * @param {string[]|null} sourceIds  null = search all user sources
 * @param {number}        topK
 * @param {(chunk: string) => void} onChunk
 * @param {(sources: Array) => void} [onSources]
 * @param {Array<{role: string, content: string}>} [history]
 *   Prior conversation turns in OpenAI chat format.
 *   Injected into the LLM context so follow-up questions work correctly.
 *   Each element: { role: "user"|"assistant", content: "..." }
 *   Pass [] or omit for stateless single-turn mode.
 */
export async function queryStream(
  question,
  sourceIds = null,
  topK      = 6,
  onChunk,
  onSources,
  history   = [],
) {
  const res = await apiFetch("/query", {
    method: "POST",
    body: JSON.stringify({
      question,
      source_ids: sourceIds,
      top_k:      topK,
      history:    history.length > 0 ? history : null,
    }),
  });

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop(); // keep incomplete last chunk

    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") return;
        try {
          const parsed = JSON.parse(raw);
          if (parsed.error)   throw new Error(parsed.error);
          if (parsed.text)    onChunk(parsed.text);
          if (parsed.sources) onSources?.(parsed.sources);
        } catch (e) {
          if (e.message && !e.message.startsWith("JSON")) throw e;
        }
      }
    }
  }
}

// ── Sources ───────────────────────────────────────────────────────────────────

export async function getSources() {
  const res  = await apiFetch("/sources");
  const data = await res.json();
  return data.sources; // array of SourceItem
}

export async function deleteSource(docId) {
  const res  = await apiFetch(`/sources/${docId}`, { method: "DELETE" });
  return res.json();
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function healthCheck() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}
