/**
 * PageMind floating launcher — injected into every page via Shadow DOM.
 * • Simple green circle (brand identity)
 * • Draggable to any screen corner (position persisted)
 * • Click opens the side panel
 * • Polls for SPA URL changes and notifies the panel
 */

const HOST_ID    = "__pagemind_launcher__";
const STORAGE_KEY = "__pagemind_pos__";

// ── Position helpers ──────────────────────────────────────────────────────────

function loadPos() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; }
}
function savePos(right, bottom) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ right, bottom })); } catch {}
}

// ── Mount ─────────────────────────────────────────────────────────────────────

function mount() {
  if (document.getElementById(HOST_ID)) return;

  const saved  = loadPos();
  const right  = saved.right  ?? "20px";
  const bottom = saved.bottom ?? "20px";

  const host = document.createElement("div");
  host.id = HOST_ID;
  Object.assign(host.style, {
    position:   "fixed",
    bottom:     bottom,
    right:      right,
    zIndex:     "2147483647",
    userSelect: "none",
  });

  const shadow = host.attachShadow({ mode: "open" });

  shadow.innerHTML = `
    <style>
      :host { display: block; }

      .btn {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #22c55e, #06D6A0);
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow:
          0 4px 14px rgba(34, 197, 94, 0.45),
          0 1px 3px rgba(0, 0, 0, 0.12),
          inset 0 1px 0 rgba(255,255,255,0.2);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        color: #fff;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 20px;
        outline: none;
        -webkit-font-smoothing: antialiased;
      }

      .btn:hover {
        transform: scale(1.08);
        box-shadow:
          0 6px 22px rgba(34, 197, 94, 0.55),
          0 2px 6px rgba(0, 0, 0, 0.12),
          inset 0 1px 0 rgba(255,255,255,0.25);
      }

      .btn:active { transform: scale(0.96); }

      .btn.dragging {
        cursor: grabbing;
        transform: scale(1.12);
        box-shadow: 0 8px 28px rgba(34, 197, 94, 0.65);
      }

      .btn:focus-visible {
        outline: 2px solid #22c55e;
        outline-offset: 3px;
      }

      /* Tooltip */
      .tip {
        position: absolute;
        right: calc(100% + 10px);
        bottom: 50%;
        transform: translateY(50%);
        white-space: nowrap;
        background: rgba(15, 15, 25, 0.88);
        color: #fff;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 11px;
        font-weight: 500;
        padding: 4px 9px;
        border-radius: 6px;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.15s ease;
        backdrop-filter: blur(4px);
      }
      .btn:hover + .tip { opacity: 1; }
    </style>

    <button class="btn" title="PageMind" aria-label="Open PageMind">✦</button>
    <div class="tip">PageMind — Ask this page</div>
  `;

  const btn = shadow.querySelector(".btn");

  // ── Drag logic ──────────────────────────────────────────────────────────────

  let dragging = false;
  let startX, startY, startRight, startBottom;

  btn.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging  = false;
    startX    = e.clientX;
    startY    = e.clientY;

    const computedStyle = getComputedStyle(host);
    startRight  = parseInt(computedStyle.right,  10) || 20;
    startBottom = parseInt(computedStyle.bottom, 10) || 20;

    const SIZE = 54;

    function onMove(e) {
      const dx = startX - e.clientX;
      const dy = startY - e.clientY;

      if (!dragging && (Math.abs(dx) > 4 || Math.abs(dy) > 4)) {
        dragging = true;
        btn.classList.add("dragging");
      }
      if (!dragging) return;

      const newRight  = Math.max(8, Math.min(window.innerWidth  - SIZE, startRight  + dx));
      const newBottom = Math.max(8, Math.min(window.innerHeight - SIZE, startBottom + dy));

      host.style.right  = newRight  + "px";
      host.style.bottom = newBottom + "px";
    }

    function onUp() {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup",   onUp);
      btn.classList.remove("dragging");

      if (dragging) {
        savePos(host.style.right, host.style.bottom);
      }
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup",   onUp);
  });

  // Click only fires when not dragging
  btn.addEventListener("click", () => {
    if (!dragging) chrome.runtime.sendMessage({ type: "OPEN_PANEL" });
  });

  document.body.appendChild(host);
}

// ── SPA URL-change detection ───────────────────────────────────────────────────

let lastUrl = location.href;

setInterval(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    chrome.runtime.sendMessage({ type: "SPA_URL_CHANGED", url: lastUrl }).catch(() => {});
  }
}, 800);

// ── Init ──────────────────────────────────────────────────────────────────────

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mount);
} else {
  mount();
}
