/**
 * Content script — injected into every tab at document_idle.
 * Listens for EXTRACT messages, walks the DOM, and sends PAGE_TEXT back.
 *
 * Strips noise elements (nav, footer, script, style, etc.) before walking,
 * so only meaningful body text is captured.
 */
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "EXTRACT") return;

  try {
    // Clone so we don't mutate the live DOM
    const docClone = document.cloneNode(true);

    // Remove noise elements
    const noiseSelectors = [
      "script", "style", "noscript",
      "nav", "footer", "aside", "header",
      "[aria-hidden='true']",
      ".cookie-banner", ".ad", ".advertisement",
    ];
    noiseSelectors.forEach((sel) => {
      try {
        docClone.querySelectorAll(sel).forEach((el) => el.remove());
      } catch (_) {
        // Some selectors may not be supported; skip silently
      }
    });

    // Walk text nodes from the body
    const walker = document.createTreeWalker(
      docClone.body || docClone,
      NodeFilter.SHOW_TEXT,
      null
    );

    const lines = [];
    let node;
    while ((node = walker.nextNode())) {
      const text = node.textContent.trim();
      if (text.length > 30) lines.push(text);
    }

    const fullText = lines.join("\n");

    chrome.runtime.sendMessage({
      type: "PAGE_TEXT",
      text: fullText,
      title: document.title,
      url: window.location.href,
      charCount: fullText.length,
    });

    sendResponse({ ok: true });
  } catch (err) {
    chrome.runtime.sendMessage({
      type: "PAGE_TEXT_ERROR",
      error: err.message,
    });
    sendResponse({ ok: false, error: err.message });
  }

  // Return true to keep the message channel open for async sendResponse
  return true;
});
