/**
 * Background service worker (MV3).
 * Handles: panel open, tab switches, URL changes.
 */

const broadcast = (msg) => chrome.runtime.sendMessage(msg).catch(() => {});

// ── Open panel ────────────────────────────────────────────────────────────────

chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.type === "OPEN_PANEL" && sender.tab?.id) {
    chrome.sidePanel.open({ tabId: sender.tab.id });
  }
});

// ── Page-change detection ─────────────────────────────────────────────────────
// Fires when the user navigates to a completely new page (not SPA pushState —
// that's handled by the content script polling in launcher.js).

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url && !tab.url.startsWith("chrome")) {
    broadcast({ type: "PAGE_NAVIGATED", url: tab.url, title: tab.title || "", tabId });
  }
});

// Fires when the user switches to a different tab
chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId, (tab) => {
    if (chrome.runtime.lastError || !tab?.url) return;
    if (tab.url.startsWith("chrome")) return;
    broadcast({ type: "TAB_SWITCHED", url: tab.url, title: tab.title || "", tabId });
  });
});
