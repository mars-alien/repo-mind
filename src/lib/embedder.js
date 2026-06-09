/**
 * Ingestion entry-point for the Chrome extension.
 *
 * Flow:
 *  1. Receives extracted page data from App.jsx (text + contentType + sections)
 *  2. Sends it to the backend POST /ingest
 *  3. Backend runs the appropriate chunker (docs/wiki/general)
 *     → embedding → Weaviate storage
 *  4. Returns { doc_id, chunks_stored, content_type }
 */

import { ingestPage } from "./api.js";

/**
 * @param {string}      pageText    raw extracted text (flat, used as fallback)
 * @param {string}      title       page title
 * @param {string}      url         page URL
 * @param {Function}    onProgress  (done, total) => void
 * @param {string}      contentType "docs" | "wiki" | "general"
 * @param {Array|null}  sections    structured DOM sections (docs/wiki only)
 * @returns {{ doc_id, chunks_stored, content_type }}
 */
export async function extractAndIndex(
  pageText,
  title        = "",
  url          = "",
  onProgress   = null,
  contentType  = "general",
  sections     = null,
) {
  if (!pageText || pageText.trim().length < 50) {
    throw new Error("Not enough text content to index on this page.");
  }

  if (onProgress) onProgress(0, 1);

  const result = await ingestPage(pageText, title, url, "webpage", contentType, sections);

  if (onProgress) onProgress(1, 1);
  return result;  // { doc_id, chunks_stored, status, content_type }
}
