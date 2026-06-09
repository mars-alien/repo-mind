# RAG over Active Tab — Chrome Extension

A production-grade Chrome extension that runs an **in-browser RAG pipeline** over any webpage. Click "Index Page" → embeddings generated locally via WASM → ask questions → Gemini answers using only page content.

## Features

-  **Private** — embeddings run 100% in-browser via Transformers.js (no page data sent to any server)
-  **Fast** — ~23MB model cached after first load, cosine similarity in pure JS (~2–5ms)
-  **Universal** — works on any URL: docs, Wikipedia, blogs, Stack Overflow, legal pages, etc.
-  **Accurate** — Gemini 2.0 Flash Lite answers strictly from retrieved context only

## Architecture

```
[Active Tab DOM]
      │
      ▼ content script (extract.js)
[PAGE_TEXT]
      │
      ▼ sidepanel (App.jsx)
[chunkText] → [embedText via Transformers.js WASM]
      │
      ▼ user asks question
[embedQuery] → [cosine similarity] → top-5 chunks
      │
      ▼ POST /ask
[Render FastAPI] → [Gemini 2.0 Flash Lite] → answer
```

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd rag-tab-extension
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set VITE_BACKEND_URL after deploying backend
```

### 3. Deploy backend to Render

```bash
# Push repo to GitHub, connect to Render
# Set GEMINI_API_KEY in Render dashboard
# Update VITE_BACKEND_URL in .env with your Render URL
```

### 4. Build extension

```bash
npm run build
```

### 5. Load in Chrome

1. Go to `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `dist/` folder

### 6. Use it

1. Navigate to any webpage
2. Click the extension icon → side panel opens
3. Click **Index Page** — wait for "Ready" status
4. Type a question and press Enter

## Project Structure

```
rag-tab-extension/
├── manifest.json              # Chrome MV3 manifest
├── vite.config.js
├── package.json
├── tailwind.config.js
├── render.yaml                # Render deploy config
├── src/
│   ├── sidepanel/             # React UI
│   │   ├── index.html
│   │   ├── index.jsx
│   │   ├── index.css
│   │   └── App.jsx            # Main orchestration
│   ├── content/
│   │   └── extract.js         # DOM text extractor
│   ├── background/
│   │   └── service-worker.js  # Message router
│   └── lib/
│       ├── chunker.js         # Sliding window text splitter
│       ├── embedder.js        # Transformers.js pipeline
│       ├── retriever.js       # Cosine similarity search
│       └── gemini.js          # Backend API client
└── backend/
    ├── main.py                # FastAPI app
    ├── requirements.txt
    └── .env.example
```

## Known Limitations

- **PDFs** in Chrome PDF viewer — content scripts can't access PDF text
- **Paywalled content** — only visible DOM text is extracted
- **SPAs** with lazy-loaded content may miss late-rendered sections
- **YouTube** — transcript not in DOM

## Chrome Web Store Checklist

Before publishing:
- [ ] Replace `allow_origins=["*"]` in `backend/main.py` with your extension ID
- [ ] Add final icons to `icons/` folder (16, 48, 128 px PNGs)
- [ ] Update name in `manifest.json`
- [ ] Write store listing description

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | React 18 + Tailwind CSS 3 |
| Build | Vite + @crxjs/vite-plugin |
| Embeddings | @xenova/transformers (all-MiniLM-L6-v2 quantized, ~23MB) |
| Similarity | Pure JS cosine similarity |
| Backend | FastAPI + Uvicorn (Render free tier) |
| LLM | Gemini 2.0 Flash Lite via google-genai SDK |


