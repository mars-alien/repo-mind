import { useState, useEffect, useRef, useCallback } from "react";
import { extractAndIndex }                  from "../lib/embedder.js";
import {
  AuthError, clearSession, getToken, getUsername,
  login, register, getSources, deleteSource, queryStream,
  ingestGitHub,
} from "../lib/api.js";

// ─── Status enum ───────────────────────────────────────────────────────────────

const S = {
  IDLE:       "idle",
  EXTRACTING: "extracting",
  EMBEDDING:  "embedding",
  READY:      "ready",
  ASKING:     "asking",
  ERROR:      "error",
};

// ─── Starter questions ─────────────────────────────────────────────────────────

const STARTERS = [
  "Summarise what I've indexed so far",
  "What are the key concepts across my sources?",
  "What topics are covered in my knowledge base?",
  "Give me the most important facts from my sources",
];

// ─── Icons ─────────────────────────────────────────────────────────────────────

function Ic({ d, size = 16, stroke = 2 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth={stroke}
         strokeLinecap="round" strokeLinejoin="round">
      <path d={d}/>
    </svg>
  );
}

const Icons = {
  Send:       () => <Ic d="M22 2 11 13M22 2 15 22l-4-9-9-4 20-7z"/>,
  Copy:       () => <Ic d="M8 4H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2M16 4h2a2 2 0 0 1 2 2v4M21 14H11"/>,
  Check:      () => <Ic d="M20 6 9 17l-5-5"/>,
  Trash:      () => <Ic d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>,
  Download:   () => <Ic d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>,
  Refresh:    () => <Ic d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8M21 3v5h-5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16M8 16H3v5"/>,
  ChevDown:   () => <Ic d="m6 9 6 6 6-6"/>,
  Search:     () => <Ic d="m21 21-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" stroke={1.5}/>,
  Zap:        () => <Ic d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>,
  Library:    () => <Ic d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15z"/>,
  Chat:       () => <Ic d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>,
  Logout:     () => <Ic d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>,
  Globe:      () => <Ic d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0M3.6 9h16.8M3.6 15h16.8M11.5 3a17 17 0 0 0 0 18M12.5 3a17 17 0 0 1 0 18"/>,
  Link:       () => <Ic d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>,
  Eye:        () => <Ic d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 12a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/>,
  EyeOff:     () => <Ic d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24M1 1l22 22"/>,
  Plus:       () => <Ic d="M12 5v14M5 12h14"/>,
  XCircle:    () => <Ic d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM15 9l-6 6M9 9l6 6"/>,
  Paperclip:  () => <Ic d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>,
  FileText:   () => <Ic d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M16 13H8M16 17H8M10 9H8"/>,
  Spinner:    ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-spin">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-20"/>
      <path fill="currentColor" className="opacity-80"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
    </svg>
  ),
};

// ─── Helpers ───────────────────────────────────────────────────────────────────

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDate(iso) {
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" });
}

// Returns { owner, repo, slug } when url is a GitHub repo page, otherwise null.
function parseGitHubRepo(url) {
  try {
    const u = new URL(url);
    if (!u.hostname.includes("github.com")) return null;
    const parts = u.pathname.split("/").filter(Boolean);
    if (parts.length < 2) return null;
    const owner = parts[0];
    const repo  = parts[1].replace(/\.git$/, "");
    const nonRepo = new Set(["login","settings","explore","marketplace","topics",
                             "notifications","pulls","issues","orgs","sponsors",
                             "features","pricing","about"]);
    if (nonRepo.has(owner.toLowerCase())) return null;
    return { owner, repo, slug: `${owner}/${repo}` };
  } catch { return null; }
}

// ─── Avatars ───────────────────────────────────────────────────────────────────

/** Simple green circle showing the user's initial letter. */
function GreenAvatar({ initial = "?", size = 28, pulse = false }) {
  return (
    <div
      className={`rounded-full flex items-center justify-center text-white font-bold
                  flex-shrink-0 select-none ${pulse ? "avatar-pulse" : ""}`}
      style={{
        width:      size,
        height:     size,
        background: "linear-gradient(135deg, #22c55e, #06D6A0)",
        fontSize:   Math.round(size * 0.42),
        cursor:     "default",
      }}
    >
      {String(initial).toUpperCase().charAt(0)}
    </div>
  );
}

/** Small green circle with "AI" label — used in assistant message bubbles. */
function BotAvatar() {
  return (
    <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mb-0.5"
         style={{ background: "linear-gradient(135deg, #22c55e, #06D6A0)" }}>
      <span className="text-white text-[9px] font-extrabold tracking-tight">AI</span>
    </div>
  );
}

// ─── Markdown renderer ─────────────────────────────────────────────────────────
// Renders **bold**, *italic*, `code`, - bullets, 1. numbered, ### headings.
// Pure React — no external deps, no dangerouslySetInnerHTML.

function renderInline(text) {
  const parts = [];
  const rx = /\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`/g;
  let last = 0, m;
  while ((m = rx.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[1]) parts.push(<strong key={m.index} className="font-semibold">{m[1]}</strong>);
    else if (m[2]) parts.push(<em key={m.index} className="italic">{m[2]}</em>);
    else if (m[3]) parts.push(
      <code key={m.index}
            className="bg-brand-50 text-brand-700 px-1 py-0.5 rounded text-[11px] font-mono">
        {m[3]}
      </code>
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length === 0 ? text : parts;
}

function MarkdownText({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const nodes = [];
  let listBuf = [];

  const flushList = () => {
    if (listBuf.length === 0) return;
    nodes.push(
      <ul key={`ul-${nodes.length}`} className="space-y-0.5 my-1">
        {listBuf}
      </ul>
    );
    listBuf = [];
  };

  lines.forEach((line, i) => {
    const h3 = line.match(/^###\s+(.+)/);
    const h2 = line.match(/^##\s+(.+)/);
    const h1 = line.match(/^#\s+(.+)/);
    if (h3 || h2 || h1) {
      flushList();
      const content = (h3 || h2 || h1)[1];
      nodes.push(<p key={i} className="font-semibold text-slate-800 mt-2 mb-0.5">{renderInline(content)}</p>);
      return;
    }
    const bullet = line.match(/^[\-\*]\s+(.+)/);
    if (bullet) {
      listBuf.push(
        <li key={i} className="flex items-start gap-2 list-none">
          <span className="mt-[5px] w-1.5 h-1.5 rounded-full bg-brand-400 flex-shrink-0"/>
          <span>{renderInline(bullet[1])}</span>
        </li>
      );
      return;
    }
    const num = line.match(/^(\d+)\.\s+(.+)/);
    if (num) {
      listBuf.push(
        <li key={i} className="flex items-start gap-2 list-none">
          <span className="flex-shrink-0 text-brand-500 font-semibold text-xs w-4 text-right">{num[1]}.</span>
          <span>{renderInline(num[2])}</span>
        </li>
      );
      return;
    }
    if (!line.trim()) {
      flushList();
      nodes.push(<div key={i} className="h-1.5"/>);
      return;
    }
    flushList();
    nodes.push(<p key={i} className="leading-relaxed">{renderInline(line)}</p>);
  });

  flushList();
  return (
    <div className="space-y-0.5 text-sm break-words overflow-hidden w-full">
      {nodes}
    </div>
  );
}

// ─── Status pill ───────────────────────────────────────────────────────────────

function StatusPill({ status }) {
  const map = {
    [S.IDLE]:       { label: "Not indexed",  bg: "bg-slate-100",    text: "text-slate-500" },
    [S.EXTRACTING]: { label: "Reading…",     bg: "bg-curious-50",  text: "text-curious-600", spin: true },
    [S.EMBEDDING]:  { label: "Indexing…",    bg: "bg-brand-50",    text: "text-brand-600",   spin: true },
    [S.READY]:      { label: "Ready",        bg: "bg-brand-50",    text: "text-brand-600" },
    [S.ASKING]:     { label: "Thinking…",    bg: "bg-fresh-50",    text: "text-fresh-600",   spin: true },
    [S.ERROR]:      { label: "Error",        bg: "bg-red-50",      text: "text-red-500" },
  };
  const c = map[status] || map[S.IDLE];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      {c.spin ? <Icons.Spinner size={11}/> : <span className="w-1.5 h-1.5 rounded-full bg-current opacity-70"/>}
      {c.label}
    </span>
  );
}

// ─── Source citations ──────────────────────────────────────────────────────────

function SourceCitations({ sources }) {
  if (!sources?.length) return null;

  // Deduplicate: code chunks → unique by (filepath, start_line); others → by doc_id
  const seen   = new Set();
  const unique = sources.filter(s => {
    const key = s.filepath && s.start_line
      ? `${s.filepath}:${s.start_line}`
      : s.doc_id;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const hostname = (url) => {
    try { return new URL(url).hostname.replace("www.", ""); } catch { return url; }
  };

  return (
    <div className="mt-3 pt-2.5 border-t border-slate-100">
      <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
        Sources
      </p>
      <div className="space-y-1.5">
        {unique.map((s, i) => {
          const isCode      = Boolean(s.filepath && s.start_line);
          const citeLabel   = isCode
            ? `${s.filepath}:L${s.start_line}${s.end_line && s.end_line !== s.start_line ? `-L${s.end_line}` : ""}`
            : (s.title || hostname(s.url));
          const subLabel    = isCode ? (s.chunk_type || "code") : hostname(s.url);

          return (
            <a key={i} href={s.url} target="_blank" rel="noopener noreferrer"
               title={s.url}
               className="flex items-center gap-2 no-underline group">
              <span className={`w-4 h-4 rounded flex-shrink-0 text-[9px] font-bold
                               flex items-center justify-center
                               ${isCode
                                 ? "bg-slate-800 text-slate-100"
                                 : "bg-brand-50 text-brand-500"}`}>
                {i + 1}
              </span>
              <span className="min-w-0 flex-1">
                <span className={`block text-[11px] font-medium transition-colors truncate leading-tight
                                  ${isCode
                                    ? "font-mono text-slate-700 group-hover:text-brand-600"
                                    : "text-slate-600 group-hover:text-brand-600"}`}>
                  {citeLabel}
                </span>
                <span className="block text-[10px] text-slate-400 truncate leading-tight capitalize">
                  {subLabel}
                </span>
              </span>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2" strokeLinecap="round"
                   className="flex-shrink-0 text-slate-300 group-hover:text-brand-500 transition-colors">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14 21 3"/>
              </svg>
            </a>
          );
        })}
      </div>
    </div>
  );
}

// ─── Message bubble ────────────────────────────────────────────────────────────

function Message({ role, text, ts, sources, isStreaming }) {
  const isUser  = role === "user";
  const isEmpty = !text && !isUser;
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try { await navigator.clipboard.writeText(text); } catch {}
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);

  return (
    <div className={`flex items-end gap-2 animate-fade-up group w-full min-w-0
                     ${isUser ? "flex-row-reverse" : "flex-row"}`}>

      {!isUser && <BotAvatar/>}

      <div className={`flex flex-col gap-1 min-w-0 max-w-[82%]
                       ${isUser ? "items-end" : "items-start"}`}>

        {/* Bubble */}
        <div className={`w-full rounded-2xl overflow-hidden ${
          isUser
            ? "bg-brand-500 text-white rounded-br-none shadow-sm shadow-brand-200"
            : "bg-white border border-slate-100 text-slate-800 rounded-bl-none shadow-sm"
        } ${isStreaming && text ? "streaming-cursor" : ""}`}>

          <div className="px-3 py-2.5">
            {isEmpty ? (
              <div className="flex gap-1.5 px-0.5 py-1">
                {[0,1,2].map(i => (
                  <span key={i} className="dot w-1.5 h-1.5 rounded-full bg-slate-300 block"/>
                ))}
              </div>
            ) : isUser ? (
              <p className="text-sm leading-relaxed break-words whitespace-pre-wrap">{text}</p>
            ) : (
              <div className="break-words overflow-hidden">
                <MarkdownText text={text}/>
                {sources?.length > 0 && !isStreaming && (
                  <SourceCitations sources={sources}/>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Timestamp + copy — always visible copy button */}
        {!isUser && text && !isEmpty && (
          <div className="flex items-center gap-1.5 px-1">
            {ts && <span className="text-[10px] text-slate-300">{fmtTime(ts)}</span>}
            <button
              onClick={copy}
              title="Copy response"
              className={`flex items-center gap-1 px-2 py-1 rounded-lg border text-[10px]
                          font-medium transition-all active:scale-95 ${
                copied
                  ? "bg-brand-50 border-brand-200 text-brand-600"
                  : "bg-white border-slate-100 text-slate-400 hover:bg-brand-50 hover:border-brand-200 hover:text-brand-600"
              }`}
            >
              {copied
                ? <><Icons.Check size={11}/><span>Copied!</span></>
                : <><Icons.Copy  size={11}/><span>Copy</span></>
              }
            </button>
          </div>
        )}
        {isUser && ts && (
          <span className="text-[10px] text-slate-300 px-1">{fmtTime(ts)}</span>
        )}
      </div>
    </div>
  );
}

// ─── Scroll FAB ────────────────────────────────────────────────────────────────

function ScrollFAB({ onClick }) {
  return (
    <button onClick={onClick}
            className="absolute bottom-2 right-3 w-7 h-7 rounded-full bg-white
                       border border-slate-200 shadow-md flex items-center justify-center
                       text-slate-500 hover:text-brand-600 hover:border-brand-300
                       animate-scale-in transition-all z-10">
      <Icons.ChevDown size={14}/>
    </button>
  );
}


// ─── GitHub-only gate (shown on all non-GitHub pages) ─────────────────────────

function GitHubOnlyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 text-center gap-5">
      <div className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0"
           style={{ background: "linear-gradient(135deg,#1a1a2e,#0d1117)" }}>
        <svg viewBox="0 0 24 24" width="32" height="32" fill="white">
          <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
        </svg>
      </div>

      <div>
        <p className="text-sm font-semibold text-slate-700">GitHub repos only</p>
        <p className="text-xs text-slate-400 mt-1.5 leading-relaxed max-w-[220px]">
          Navigate to any public GitHub repository and click <strong>Index this repo</strong> to get started.
        </p>
      </div>

      <div className="bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 max-w-[220px] text-left space-y-1.5">
        <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Example</p>
        <p className="text-[11px] font-mono text-brand-600 break-all">github.com/owner/repo</p>
      </div>

      <p className="text-[10px] text-slate-300 max-w-[200px] leading-relaxed">
        Answers cite exact file paths and function names
      </p>
    </div>
  );
}

// ─── GitHub Empty State ────────────────────────────────────────────────────────

function GitHubEmptyState({ repo, onIndex, isIndexing, errorMsg }) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 text-center gap-5">
      {/* GitHub-flavoured brand mark */}
      <div className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0"
           style={{ background: "linear-gradient(135deg,#1a1a2e,#0d1117)" }}>
        <svg viewBox="0 0 24 24" width="32" height="32" fill="white">
          <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
        </svg>
      </div>

      <div>
        <p className="text-sm font-semibold text-slate-700">GitHub Repository</p>
        <p className="text-xs font-mono text-brand-600 mt-0.5 bg-brand-50 px-2 py-0.5 rounded-lg inline-block">
          {repo.slug}
        </p>
        <p className="text-xs text-slate-400 mt-2.5 leading-relaxed max-w-[240px]">
          Index this repo to ask questions about any file, function, or class.
          Answers will cite exact file paths.
        </p>
      </div>

      <div className="flex flex-col gap-2 w-full max-w-[220px]">
        <button onClick={onIndex} disabled={isIndexing}
                className="flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-semibold
                           text-white rounded-xl active:scale-95 transition-all
                           disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ background: isIndexing ? "#94a3b8" : "#22c55e" }}>
          {isIndexing
            ? <><Icons.Spinner/> Indexing repo…</>
            : <><Icons.Zap size={13}/> Index this repo</>
          }
        </button>

        {errorMsg && (
          <p className="text-[11px] text-red-500 text-center leading-relaxed">{errorMsg}</p>
        )}

        <p className="text-[10px] text-slate-300 text-center">
          Public repos work without a token
        </p>
      </div>
    </div>
  );
}

// ─── Auth Screen ───────────────────────────────────────────────────────────────

function AuthScreen({ onAuth }) {
  const [mode,      setMode]      = useState("login");
  const [username,  setUsername]  = useState("");
  const [password,  setPassword]  = useState("");
  const [showPass,  setShowPass]  = useState(false);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");

  const submit = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) return;
    setLoading(true);
    setError("");
    try {
      const fn   = mode === "login" ? login : register;
      const data = await fn(username.trim(), password);
      onAuth(data.username);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen items-center justify-center px-6"
         style={{ background: "linear-gradient(160deg, #f0fdf4 0%, #dcfce7 50%, #f0fdf9 100%)" }}>

      {/* Brand mark — large green circle */}
      <div className="flex flex-col items-center gap-3 mb-8">
        <div className="w-14 h-14 rounded-full flex items-center justify-center avatar-pulse"
             style={{ background: "linear-gradient(135deg, #22c55e, #06D6A0)" }}>
          <span className="text-white text-2xl select-none">✦</span>
        </div>
        <div className="text-center">
          <p className="text-xs text-slate-400 mt-1">Your personal AI knowledge assistant</p>
        </div>
      </div>

      {/* Card */}
      <div className="w-full bg-white rounded-2xl border border-slate-100 shadow-sm p-5 space-y-4">

        {/* Tab switcher */}
        <div className="flex bg-slate-100 rounded-xl p-1">
          {["login", "register"].map(m => (
            <button key={m} onClick={() => { setMode(m); setError(""); }}
                    className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                      mode === m
                        ? "bg-white text-slate-800 shadow-sm"
                        : "text-slate-400 hover:text-slate-600"
                    }`}>
              {m === "login" ? "Sign in" : "Create account"}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="space-y-3">
          {/* Username */}
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Username</label>
            <input
              type="text" value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="e.g. royal"
              autoComplete="username"
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200
                         focus:outline-none focus:ring-2 focus:ring-brand-300 focus:border-transparent
                         placeholder:text-slate-300 transition-all"
            />
          </div>

          {/* Password */}
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Password</label>
            <div className="relative">
              <input
                type={showPass ? "text" : "password"}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                className="w-full px-3 py-2.5 pr-10 text-sm rounded-xl border border-slate-200
                           focus:outline-none focus:ring-2 focus:ring-brand-300 focus:border-transparent
                           placeholder:text-slate-300 transition-all"
              />
              <button type="button" onClick={() => setShowPass(p => !p)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-300 hover:text-slate-500">
                {showPass ? <Icons.EyeOff size={14}/> : <Icons.Eye size={14}/>}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <p className="text-xs text-red-500 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          {/* Submit */}
          <button type="submit" disabled={loading || !username.trim() || !password.trim()}
                  className="w-full py-2.5 text-sm font-semibold text-white rounded-xl
                             active:scale-[0.99] transition-all
                             disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: "#22c55e" }}>
            {loading
              ? <span className="flex items-center justify-center gap-2"><Icons.Spinner size={14}/> Please wait…</span>
              : mode === "login" ? "Sign in" : "Create account"
            }
          </button>
        </form>
      </div>

      <p className="text-[10px] text-slate-400 mt-4 text-center">
        Your data is private and stored locally on your machine.
      </p>
    </div>
  );
}

// ─── Content-type badge ─────────────────────────────────────────────────────────

function ContentTypeBadge({ type }) {
  if (type === "docs") return (
    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full
                     bg-brand-50 text-brand-600 border border-brand-100 flex-shrink-0">
      DOCS
    </span>
  );
  if (type === "wiki") return (
    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full
                     bg-fresh-50 text-fresh-600 border border-fresh-100 flex-shrink-0">
      WIKI
    </span>
  );
  if (type === "code") return (
    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full
                     bg-slate-800 text-slate-100 border border-slate-700 flex-shrink-0">
      CODE
    </span>
  );
  return (
    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full
                     bg-slate-50 text-slate-400 border border-slate-100 flex-shrink-0">
      WEB
    </span>
  );
}

// ─── Library panel ─────────────────────────────────────────────────────────────

function LibraryPanel({ onFilterChat }) {
  const [sources,    setSources]    = useState([]);
  const [loading,    setLoading]    = useState(true);
  const [deletingId, setDeletingId] = useState(null);
  const [error,      setError]      = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setSources(await getSources());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (docId) => {
    setDeletingId(docId);
    try {
      await deleteSource(docId);
      setSources(prev => prev.filter(s => s.doc_id !== docId));
    } catch (e) {
      setError(e.message);
    } finally {
      setDeletingId(null);
    }
  };

  const sourceIcon = (src) => {
    if (src.source_type === "github") return "⌥";
    if (src.source_type === "url")    return "🔗";
    if (src.source_type === "note")   return "📄";
    return "🌐";
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">

      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <Icons.Spinner size={20}/>
        </div>
      ) : error ? (
        <div className="flex-1 flex flex-col items-center justify-center px-4 text-center gap-2">
          <p className="text-xs text-red-500">{error}</p>
          <button onClick={load} className="text-xs text-brand-500 underline">Retry</button>
        </div>
      ) : sources.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center px-6 text-center gap-3">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl"
               style={{ background: "linear-gradient(135deg,#f0fdf4,#dcfce7)" }}>
            📚
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-700">No repos indexed yet</p>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed max-w-[220px]">
              Navigate to any public GitHub repository and click
              <strong> Index this repo</strong> to get started.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-2">
          <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-1 mb-1">
            {sources.length} source{sources.length !== 1 ? "s" : ""} indexed
          </p>

          {sources.map(src => (
            <div key={src.doc_id}
                 className="bg-white border border-slate-100 rounded-xl px-3 py-2.5
                            flex items-start gap-2.5 group hover:border-brand-100
                            hover:bg-brand-50/20 transition-all animate-fade-up">

              <span className="text-base mt-0.5 flex-shrink-0">{sourceIcon(src)}</span>

              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <p className="text-xs font-semibold text-slate-700 truncate leading-tight">
                    {src.title || "Untitled"}
                  </p>
                  <ContentTypeBadge type={src.content_type}/>
                </div>
                <p className="text-[10px] text-slate-400 truncate mt-0.5">{src.url}</p>
                <div className="flex items-center gap-2 mt-1.5">
                  <span className="text-[10px] text-slate-300">{fmtDate(src.created_at)}</span>
                  <span className="text-[10px] text-brand-500 font-medium">
                    {src.chunk_count} chunks
                  </span>
                </div>
              </div>

              <div className="flex flex-col gap-1.5 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                <button onClick={() => onFilterChat(src)}
                        title="Chat about this source"
                        className="w-6 h-6 rounded-lg flex items-center justify-center
                                   text-slate-400 hover:text-brand-600 hover:bg-brand-50 transition-all">
                  <Icons.Chat size={11}/>
                </button>
                <button onClick={() => handleDelete(src.doc_id)}
                        disabled={deletingId === src.doc_id}
                        title="Remove from knowledge base"
                        className="w-6 h-6 rounded-lg flex items-center justify-center
                                   text-slate-400 hover:text-red-400 hover:bg-red-50 transition-all
                                   disabled:opacity-50">
                  {deletingId === src.doc_id
                    ? <Icons.Spinner size={10}/>
                    : <Icons.Trash size={11}/>
                  }
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [authed,      setAuthed]      = useState(false);
  const [username,    setUsername]    = useState("");
  const [authChecked, setAuthChecked] = useState(false);
  const [tab,         setTab]         = useState("chat");

  const [status,      setStatus]      = useState(S.IDLE);
  const [store,       setStore]       = useState(null);
  const [pageInfo,    setPageInfo]    = useState(null);
  const [messages,    setMessages]    = useState([]);
  const [input,       setInput]       = useState("");
  const [error,       setError]       = useState(null);
  const [showScroll,  setShowScroll]  = useState(false);
  const [streaming,   setStreaming]   = useState(false);

  const [sourceFilter,  setSourceFilter]  = useState(null);
  const [currentTabUrl, setCurrentTabUrl] = useState("");
  const [ghIndexing,    setGhIndexing]    = useState(false);
  const [ghError,       setGhError]       = useState("");

  const chatRef    = useRef(null);
  const chatEndRef = useRef(null);
  const inputRef   = useRef(null);

  // ── Check auth on mount ────────────────────────────────────────────────────

  useEffect(() => {
    (async () => {
      const token = await getToken();
      if (token) {
        const uname = await getUsername();
        setAuthed(true);
        setUsername(uname || "");
      }
      setAuthChecked(true);
    })();
  }, []);

  // ── Track active tab URL for GitHub detection ─────────────────────────────

  useEffect(() => {
    chrome.tabs.query({ active: true, currentWindow: true }, ([t]) => {
      if (t?.url) setCurrentTabUrl(t.url);
    });
    const handler = (msg) => {
      if (msg.type === "PAGE_NAVIGATED" || msg.type === "TAB_SWITCHED") {
        setCurrentTabUrl(msg.url || "");
        setGhError("");
      }
    };
    chrome.runtime.onMessage.addListener(handler);
    return () => chrome.runtime.onMessage.removeListener(handler);
  }, []);

  // ── Scroll helpers ─────────────────────────────────────────────────────────

  const scrollBottom = useCallback((force = false) => {
    if (!chatRef.current) return;
    const el = chatRef.current;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (force || nearBottom) {
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 40);
    }
  }, []);

  useEffect(() => {
    const el = chatRef.current;
    if (!el) return;
    const onScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
      setShowScroll(!nearBottom && el.scrollHeight > el.clientHeight + 100);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);


  // ── Index current page ─────────────────────────────────────────────────────

  const handleIndex = useCallback(async () => {
    setStatus(S.EXTRACTING);
    setError(null);
    setStore(null);
    setMessages([]);
    setSourceFilter(null);

    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) throw new Error("No active tab found.");

      const tabUrl = tab.url || "";
      const isRestricted =
        !tabUrl ||
        tabUrl.startsWith("chrome://") ||
        tabUrl.startsWith("chrome-extension://") ||
        tabUrl.startsWith("edge://") ||
        tabUrl.startsWith("about:") ||
        tabUrl.startsWith("devtools://");

      if (isRestricted) {
        setStatus(S.IDLE);
        setMessages([{
          role: "assistant",
          text: "Navigate to a regular web page, then click **Index this page** to get started.\n\nChrome system pages and new tabs cannot be indexed.",
          ts: Date.now(),
        }]);
        return;
      }

      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          const url      = location.href;
          const hostname = location.hostname.toLowerCase();

          function detectType() {
            if (hostname.includes("wikipedia.org") || document.getElementById("mw-content-text"))
              return "wiki";
            const docsPatterns = [
              /^docs?\./i, /\/docs?\//i, /\/api\//i, /\/guide\//i,
              /\/reference\//i, /\/manual\//i, /readthedocs\.io/i,
              /developer\./i, /\.github\.io/i, /devdocs\./i,
            ];
            const testUrl = hostname + url.replace("https://"+hostname,"").replace("http://"+hostname,"");
            if (docsPatterns.some(p => p.test(testUrl))) return "docs";
            const codeCount    = document.querySelectorAll("pre, code").length;
            const headingCount = document.querySelectorAll("h2, h3, h4").length;
            if (codeCount >= 5 && headingCount >= 5) return "docs";
            return "general";
          }

          function extractSections(contentType) {
            const WIKI_SKIP = new Set([
              "references","external links","see also","notes",
              "further reading","bibliography","footnotes","citations",
            ]);
            const root = document.querySelector(
              contentType === "wiki"
                ? "#mw-content-text .mw-parser-output, #mw-content-text, #content"
                : "main, article, [role='main'], .content, .docs-content, #docs-content, .documentation, #content, .container"
            ) || document.body;

            const sections     = [];
            const headingStack = [];
            let currentHeading = "";
            let currentPath    = "";
            let currentText    = "";

            function buildPath(level, text) {
              while (headingStack.length && headingStack[headingStack.length-1].level >= level)
                headingStack.pop();
              headingStack.push({ level, text });
              return headingStack.map(h => h.text).join(" > ");
            }

            function flush() {
              const t = currentText.trim();
              if (t.length < 40) return;
              if (contentType === "wiki" && WIKI_SKIP.has(currentHeading.toLowerCase())) return;
              sections.push({ heading: currentHeading, path: currentPath, text: t });
            }

            const ACCEPTED = new Set(["h1","h2","h3","h4","p","pre","blockquote","li","td","dt","dd"]);
            const REJECTED = new Set(["nav","footer","aside","header","script","style","noscript"]);

            const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, {
              acceptNode(n) {
                const tag = n.tagName.toLowerCase();
                if (REJECTED.has(tag)) return NodeFilter.FILTER_REJECT;
                const cls = (n.className || "").toLowerCase();
                if (/\b(nav|sidebar|menu|toc|table-of-content|breadcrumb|cookie)\b/.test(cls))
                  return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
              },
            });

            let node;
            while ((node = walker.nextNode())) {
              const tag    = node.tagName.toLowerCase();
              const hMatch = tag.match(/^h([1-6])$/);

              if (hMatch) {
                flush();
                const level       = parseInt(hMatch[1]);
                const headingText = node.textContent.trim()
                  .replace(/\s+/g, " ").replace(/\[edit\]/gi, "");
                currentHeading = headingText;
                currentPath    = buildPath(level, headingText);
                currentText    = "";
              } else if (ACCEPTED.has(tag)) {
                if ((tag === "p" || tag === "li") && node.querySelector("p, li, pre, blockquote")) continue;
                const t = node.textContent.trim().replace(/\s+/g, " ");
                if (t.length > 15) currentText += t + "\n";
              }
            }
            flush();
            return sections;
          }

          function extractFlatText() {
            const clone = document.cloneNode(true);
            // Only remove clearly non-content elements — avoid aside/header which
            // many sites (e.g. Stack Overflow) place main content beside or inside
            ["script","style","noscript","nav","footer",
             ".cookie-banner",".ad",".advertisement",".sidebar",
             "#sidebar","#left-sidebar","#right-sidebar"]
              .forEach(sel => {
                try { clone.querySelectorAll(sel).forEach(el => el.remove()); } catch {}
              });
            // Prefer a recognised main-content root before falling back to body
            const root = clone.querySelector(
              'main, [role="main"], article, #mainbar, #content, .question-container, .post-layout'
            ) || clone.body || clone;
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
            const lines = [];
            let n;
            while ((n = walker.nextNode())) {
              const t = n.textContent.trim();
              if (t.length > 20) lines.push(t);
            }
            return lines.join("\n");
          }

          const contentType = detectType();
          let text, sections;

          if (contentType === "docs" || contentType === "wiki") {
            sections = extractSections(contentType);
            text     = sections.map(s => s.text).join("\n\n");
          } else {
            sections = null;
            text     = extractFlatText();
          }

          return { text, title: document.title, url: location.href, contentType, sections };
        },
      });

      const page = results?.[0]?.result;
      if (!page?.text || page.text.trim().length < 50) {
        throw new Error(
          "Not enough readable text on this page. Try a content-rich article or document."
        );
      }

      setPageInfo({ title: page.title, url: page.url, contentType: page.contentType });
      setStatus(S.EMBEDDING);

      const result = await extractAndIndex(
        page.text, page.title, page.url,
        () => {},
        page.contentType,
        page.sections,
      );

      const typeLabel = page.contentType === "docs" ? "docs page"
                      : page.contentType === "wiki" ? "wiki article"
                      : "page";

      setStore(result);
      setStatus(S.READY);
      setError(null);
      setMessages([{
        role: "assistant",
        text: `Ready. Indexed **${result.chunks_stored} sections** from this **${typeLabel}**. Ask me anything or pick a suggestion below.`,
        ts: Date.now(),
      }]);
      scrollBottom(true);
      setTimeout(() => inputRef.current?.focus(), 100);
    } catch (err) {
      if (err instanceof AuthError) { setAuthed(false); return; }
      setError(err.message);
      setStatus(S.ERROR);
    }
  }, [scrollBottom, authed]);

  // ── Index GitHub repo ──────────────────────────────────────────────────────

  const handleIndexGitHub = useCallback(async () => {
    const ghRepo = parseGitHubRepo(currentTabUrl);
    if (!ghRepo) return;

    setGhIndexing(true);
    setGhError("");
    setError(null);
    setStatus(S.EMBEDDING);
    setStore(null);
    setMessages([]);
    setSourceFilter(null);

    try {
      const result = await ingestGitHub(`https://github.com/${ghRepo.slug}`);

      setStore(result);
      setStatus(S.READY);
      setPageInfo({
        title:       `GitHub: ${ghRepo.slug}`,
        url:         `https://github.com/${ghRepo.slug}`,
        contentType: "code",
      });
      setMessages([{
        role: "assistant",
        text: (
          `Ready. Indexed **${result.files_indexed} files** ` +
          `(${result.chunks_stored} chunks) from **${ghRepo.slug}**.\n\n` +
          `Ask me anything about this codebase — I'll cite exact file paths in answers.`
        ),
        ts: Date.now(),
      }]);
      scrollBottom(true);
      setTimeout(() => inputRef.current?.focus(), 100);
    } catch (err) {
      if (err instanceof AuthError) { setAuthed(false); return; }
      setGhError(err.message);
      setError(err.message);
      setStatus(S.ERROR);
    } finally {
      setGhIndexing(false);
    }
  }, [currentTabUrl, scrollBottom]);

  // ── Ask a question ─────────────────────────────────────────────────────────

  const handleAsk = useCallback(async (question) => {
    const q = (question || input).trim();
    if (!q || status === S.ASKING) return;

    const history = messages
      .filter(m =>
        m.text &&
        m.text.trim().length > 10 &&
        (m.role === "user" || m.role === "assistant")
      )
      .map(m => ({ role: m.role, content: m.text }))
      .slice(-10);

    setInput("");
    setMessages(prev => [...prev, { role: "user", text: q, ts: Date.now() }]);
    setStatus(S.ASKING);
    setStreaming(false);
    scrollBottom(true);

    setMessages(prev => [...prev, { role: "assistant", text: "", ts: Date.now(), sources: [] }]);
    setStreaming(true);

    try {
      const sourceIds = sourceFilter ? [sourceFilter.doc_id] : null;

      await queryStream(
        q,
        sourceIds,
        6,
        (chunk) => {
          setMessages(prev => {
            const updated = [...prev];
            const last    = { ...updated[updated.length - 1] };
            last.text     = last.text + chunk;
            updated[updated.length - 1] = last;
            return updated;
          });
          scrollBottom();
        },
        (sources) => {
          setMessages(prev => {
            const updated = [...prev];
            const last    = { ...updated[updated.length - 1] };
            last.sources  = sources;
            updated[updated.length - 1] = last;
            return updated;
          });
        },
        history,
      );
    } catch (err) {
      if (err instanceof AuthError) { setAuthed(false); return; }
      setMessages(prev => {
        const updated = [...prev];
        const last    = { ...updated[updated.length - 1] };
        last.text     = `Something went wrong: ${err.message}`;
        updated[updated.length - 1] = last;
        return updated;
      });
    } finally {
      setStreaming(false);
      setStatus(store ? S.READY : S.IDLE);
      scrollBottom(true);
      inputRef.current?.focus();
    }
  }, [input, status, store, sourceFilter, scrollBottom]);

  // ── Clear conversation ─────────────────────────────────────────────────────

  const handleClear = useCallback(() => {
    setMessages([{
      role: "assistant",
      text: store
        ? `Conversation cleared. I still have ${store.chunks_stored} sections indexed.`
        : "Conversation cleared.",
      ts: Date.now(),
    }]);
  }, [store]);

  // ── Export ─────────────────────────────────────────────────────────────────

  const handleExport = useCallback(() => {
    const lines = [
      `PageMind — ${pageInfo?.title || "Conversation"}`,
      `URL: ${pageInfo?.url || ""}`,
      `Exported: ${new Date().toLocaleString()}`,
      "─".repeat(50), "",
      ...messages.map(m =>
        `[${m.role === "user" ? "You" : "AI"} ${fmtTime(m.ts)}]\n${m.text}`
      ),
    ];
    const blob = new Blob([lines.join("\n\n")], { type: "text/plain" });
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = `pagemind-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [messages, pageInfo]);

  // ── Logout ─────────────────────────────────────────────────────────────────

  const handleLogout = async () => {
    await clearSession();
    setAuthed(false);
    setMessages([]);
    setStore(null);
    setStatus(S.IDLE);
  };

  // ── Filter to a library source ─────────────────────────────────────────────

  const handleFilterChat = (src) => {
    setSourceFilter(src);
    setTab("chat");
    setMessages([{
      role: "assistant",
      text: `Now chatting about: **${src.title || src.url}**\nAsk me anything about this source.`,
      ts: Date.now(),
    }]);
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); }
  };

  const ghRepo       = parseGitHubRepo(currentTabUrl);
  const isIndexing   = status === S.EXTRACTING || status === S.EMBEDDING;
  const canAsk       = status !== S.ASKING && input.trim().length > 0;
  const showChat     = messages.length > 0;
  const showStarters = (status === S.READY || status === S.IDLE) && messages.length <= 1;
  const charLeft     = 500 - input.length;

  // ── Prevent auth flash ─────────────────────────────────────────────────────

  if (!authChecked) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: "var(--bg)" }}>
        <Icons.Spinner size={24}/>
      </div>
    );
  }

  if (!authed) {
    return <AuthScreen onAuth={(uname) => { setUsername(uname); setAuthed(true); }}/>;
  }

  // ── Main UI ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen font-sans antialiased select-none"
         style={{ background: "var(--bg)", color: "var(--text)" }}>

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-3.5 py-2.5 bg-white border-b border-slate-100 flex-shrink-0">
        {/* Left: green avatar + username only (no "PageMind" text) */}
        <div className="flex items-center gap-2">
          <GreenAvatar initial={username || "?"} size={30} pulse/>
          <span className="text-sm font-semibold text-slate-800 tracking-tight leading-none">
            {username || "—"}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {tab === "chat" && showChat && (
            <>
              <button onClick={handleClear} title="Clear conversation"
                      className="w-7 h-7 rounded-lg flex items-center justify-center
                                 text-slate-400 hover:text-red-400 hover:bg-red-50 transition-all">
                <Icons.Trash size={13}/>
              </button>
              <button onClick={handleExport} title="Export conversation"
                      className="w-7 h-7 rounded-lg flex items-center justify-center
                                 text-slate-400 hover:text-brand-500 hover:bg-brand-50 transition-all">
                <Icons.Download size={13}/>
              </button>
            </>
          )}
          <StatusPill status={tab === "chat" ? status : S.IDLE}/>
          <button onClick={handleLogout} title="Sign out"
                  className="w-7 h-7 rounded-lg flex items-center justify-center
                             text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all ml-1">
            <Icons.Logout size={13}/>
          </button>
        </div>
      </header>

      {/* ── Tab bar ── */}
      <div className="flex bg-slate-50 border-b border-slate-100 px-3 pt-2 gap-1 flex-shrink-0">
        {[
          { id: "chat",    label: "Chat",    Icon: Icons.Chat    },
          { id: "library", label: "Library", Icon: Icons.Library },
        ].map(({ id, label, Icon }) => (
          <button key={id} onClick={() => setTab(id)}
                  className={`flex items-center gap-1.5 px-3 py-2 text-xs font-semibold
                             rounded-t-lg border-b-2 transition-all ${
                    tab === id
                      ? "border-brand-500 text-brand-600 bg-white"
                      : "border-transparent text-slate-400 hover:text-slate-600"
                  }`}>
            <Icon size={13}/>
            {label}
          </button>
        ))}
      </div>

      {/* ── Source filter chip ── */}
      {tab === "chat" && sourceFilter && (
        <div className="px-3 py-2 bg-brand-50 border-b border-brand-100 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 min-w-0">
            <Icons.Link size={11} className="text-brand-400 flex-shrink-0"/>
            <span className="text-[11px] text-brand-700 font-medium truncate">
              {sourceFilter.title || sourceFilter.url}
            </span>
          </div>
          <button onClick={() => setSourceFilter(null)}
                  className="text-[10px] text-brand-400 hover:text-brand-600 flex-shrink-0 transition-colors">
            Clear filter
          </button>
        </div>
      )}

      {/* ── Page info strip (chat tab only) ── */}
      {tab === "chat" && pageInfo && !sourceFilter && (
        <div className="px-3.5 py-2 border-b flex items-center justify-between gap-2 flex-shrink-0"
             style={{ background: "linear-gradient(to right,#f0fdf4,#dcfce7)", borderColor: "#bbf7d0" }}>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <p className="text-xs font-semibold text-brand-700 truncate leading-tight">{pageInfo.title}</p>
              {pageInfo.contentType === "docs" && (
                <span className="flex-shrink-0 text-[9px] font-bold px-1.5 py-0.5 rounded-full
                                 bg-brand-50 text-brand-600 border border-brand-100">DOCS</span>
              )}
              {pageInfo.contentType === "wiki" && (
                <span className="flex-shrink-0 text-[9px] font-bold px-1.5 py-0.5 rounded-full
                                 bg-fresh-50 text-fresh-600 border border-fresh-100">WIKI</span>
              )}
            </div>
            <p className="text-[10px] text-brand-500 truncate">{pageInfo.url}</p>
          </div>
          <button
            onClick={ghRepo ? handleIndexGitHub : handleIndex}
            disabled={isIndexing || ghIndexing}
            title={ghRepo ? "Re-index this repo" : "Re-index this page"}
            className="flex-shrink-0 w-6 h-6 rounded-lg flex items-center justify-center
                       text-brand-400 hover:text-brand-600 hover:bg-brand-100
                       transition-all disabled:opacity-40">
            <Icons.Refresh size={13}/>
          </button>
        </div>
      )}


      {/* ── Error banner ── */}
      {tab === "chat" && error && (
        <div className="mx-3 mt-2 px-3 py-2.5 bg-red-50 border border-red-100 rounded-xl
                        text-xs text-red-600 leading-relaxed flex items-start gap-2 animate-scale-in flex-shrink-0">
          <span className="flex-1">{error}</span>
          <button
            onClick={ghRepo ? handleIndexGitHub : handleIndex}
            className="flex-shrink-0 font-semibold underline hover:no-underline">
            Retry
          </button>
        </div>
      )}

      {/* ── Library tab ── */}
      {tab === "library" && (
        <LibraryPanel onFilterChat={handleFilterChat}/>
      )}

      {/* ── Chat tab ── */}
      {tab === "chat" && (
        <>
          {/* Chat messages */}
          <div ref={chatRef} className="flex-1 overflow-y-auto overflow-x-hidden px-3 py-4 space-y-3 relative min-w-0">
            {!showChat
              ? ghRepo
                ? (
                  <GitHubEmptyState
                    repo={ghRepo}
                    onIndex={handleIndexGitHub}
                    isIndexing={ghIndexing}
                    errorMsg={ghError}
                  />
                )
                : <GitHubOnlyState />
              : (
                <>
                  {messages.map((msg, i) => (
                    <Message key={i} role={msg.role} text={msg.text} ts={msg.ts}
                             sources={msg.sources}
                             isStreaming={streaming && i === messages.length - 1}/>
                  ))}

                  {showStarters && (
                    <div className="mt-1 animate-fade-up">
                      <p className="text-[11px] text-slate-400 font-medium mb-2 px-1">Try asking</p>
                      <div className="flex flex-wrap gap-1.5">
                        {STARTERS.map(q => (
                          <button key={q} className="chip" onClick={() => handleAsk(q)}>{q}</button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div ref={chatEndRef}/>
                </>
              )
            }

            {showScroll && (
              <div className="sticky bottom-0 flex justify-end pointer-events-none">
                <div className="pointer-events-auto">
                  <ScrollFAB onClick={() => scrollBottom(true)}/>
                </div>
              </div>
            )}
          </div>

          {/* Input area */}
          <div className="px-3 pb-3 pt-2 bg-white border-t border-slate-100 space-y-2 flex-shrink-0">


            {/* Main input + send */}
            {showChat && (
              <div className="flex gap-2 items-end">
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    rows={1}
                    value={input}
                    onChange={e => setInput(e.target.value.slice(0, 500))}
                    onKeyDown={onKeyDown}
                    disabled={status === S.ASKING || status === S.EXTRACTING || status === S.EMBEDDING}
                    placeholder={
                      status === S.ASKING    ? "Thinking…" :
                      status === S.EMBEDDING ? "Indexing, please wait…" :
                      sourceFilter           ? `Ask about ${sourceFilter.title || "this source"}…` :
                                              "Ask anything about your knowledge base…"
                    }
                    className="w-full resize-none text-sm px-3 py-2.5 pr-10 rounded-xl border
                               border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-300
                               focus:border-transparent transition-all
                               disabled:bg-slate-50 disabled:text-slate-300 disabled:cursor-not-allowed
                               placeholder:text-slate-300 leading-relaxed"
                    style={{ minHeight: "42px", maxHeight: "100px" }}
                    onInput={e => {
                      e.target.style.height = "auto";
                      e.target.style.height = Math.min(e.target.scrollHeight, 100) + "px";
                    }}
                  />
                  {input.length > 400 && (
                    <span className={`absolute bottom-2 right-2 text-[10px] ${
                      charLeft < 50 ? "text-red-400" : "text-slate-300"
                    }`}>{charLeft}</span>
                  )}
                </div>

                <button onClick={() => handleAsk()} disabled={!canAsk}
                        title="Send (Enter)"
                        className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center
                                   text-white active:scale-95 transition-all
                                   disabled:cursor-not-allowed disabled:active:scale-100"
                        style={{
                          background: canAsk ? "#22c55e" : "#e2e8f0",
                          color:      canAsk ? "white"   : "#94a3b8",
                        }}>
                  <Icons.Send size={15}/>
                </button>
              </div>
            )}

            {/* Re-index button — only shown when on a GitHub repo page */}
            {showChat && ghRepo && (
              <button onClick={handleIndexGitHub} disabled={ghIndexing || isIndexing}
                      className="w-full py-2 rounded-xl border border-slate-200 text-slate-500 text-xs
                                 font-medium bg-white hover:bg-brand-50 hover:border-brand-200
                                 hover:text-brand-600 active:scale-[0.99] transition-all
                                 disabled:opacity-50 disabled:cursor-not-allowed
                                 flex items-center justify-center gap-1.5">
                {ghIndexing
                  ? <><Icons.Spinner size={12}/> Indexing repo…</>
                  : <><Icons.Refresh size={12}/> Re-index repo</>
                }
              </button>
            )}

            {showChat && (
              <p className="text-center text-[10px] text-slate-300">
                Enter to send · Shift+Enter for new line
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
