"""
PageMind RAG Evaluation — LLM-as-Judge (RAGAS-style)
======================================================
Measures the quality of the full RAG pipeline using three metrics:

  faithfulness       — Is every claim in the answer supported by the context?
                       1.0 = fully grounded, 0.0 = hallucinated.

  answer_relevancy   — Does the answer actually address the question?
                       1.0 = perfectly on-topic, 0.0 = ignores the question.

  context_precision  — Are the retrieved chunks relevant to the question?
                       1.0 = all chunks are useful, 0.0 = all off-topic.

How it works
────────────
  For each question the script runs the FULL production pipeline:
    embed_query() → hybrid_retrieve() → rerank_hits() → call_llm_with_context()
  Then three Groq judge calls score each result. No ground-truth needed.

Rate limit awareness
────────────────────
  Groq free tier: ~30 req/min for llama-3.3-70b-versatile.
  Each question = 1 generation + 3 judge calls = 4 calls.
  6 questions × 4 calls = 24 calls — fits comfortably.
  A --delay flag (default 8 s) is inserted between questions as a buffer.

Prerequisites
─────────────
  1. Docker + Weaviate running:  docker compose up -d
  2. Backend venv active:        .venv\\Scripts\\activate
  3. pip install openai          (for the Groq OpenAI-compat client)
  4. At least one source indexed via the extension.

Usage
─────
  python evaluate_rag.py --username YOUR_USERNAME
  python evaluate_rag.py --username YOUR_USERNAME --questions questions.txt
  python evaluate_rag.py --username YOUR_USERNAME --delay 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time

from dotenv import load_dotenv
load_dotenv()

# ── Backend imports (must run from backend/ folder) ───────────────────────────

try:
    import weaviate
    from database  import init_db, get_user_by_username
    from embedder  import embed_query
    from retriever import hybrid_retrieve, rerank_hits, CANDIDATE_K, RERANK_TOP_N
    from llm       import (
        build_generation_kwargs, call_llm_with_context,
        GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_OUTPUT_TOKENS, SYSTEM_PROMPT,
    )
except ImportError as e:
    print(f"\n❌  Import error: {e}")
    print("    Run from the backend/ directory:")
    print("    cd backend && python evaluate_rag.py --username YOUR_USERNAME\n")
    sys.exit(1)


# ── RAG pipeline (mirrors main.py /query, non-streaming) ─────────────────────

def _parse_retry_seconds(error_msg: str) -> float:
    """Extract wait seconds from Groq rate-limit message 'try again in 19m4.8s'."""
    m = re.search(r'try again in\s+(?:(\d+)m)?([\d.]+)s', str(error_msg))
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2) or 0) + 5
    return 90


def run_rag_pipeline(question: str, collection, user_id: str,
                     max_retries: int = 3) -> dict:
    """
    Full production RAG pipeline for one question.
    Returns { question, answer, contexts } where contexts is list[str].
    Auto-retries on Groq 429 rate-limit errors.
    """
    # Step 1 — embed
    q_vec = embed_query(question)

    # Step 2 — hybrid retrieve (BM25 + HNSW), same params as production
    hits = hybrid_retrieve(
        collection   = collection,
        query_text   = question,
        query_vector = q_vec,
        user_id      = user_id,
        top_k        = CANDIDATE_K,
    )

    if not hits:
        return {"question": question, "answer": "No relevant content found.", "contexts": []}

    # Step 3 — cross-encoder rerank (matches production pipeline)
    hits = rerank_hits(hits, question, top_n=RERANK_TOP_N)

    # Step 4 — build context (same as main.py)
    context_parts = []
    for hit in hits:
        label = hit.get("heading_path") or hit.get("title") or hit.get("url") or "Source"
        context_parts.append(f"[{label}]\n{hit['text']}")
    context_str = "\n\n---\n\n".join(context_parts)

    system_content = f"{SYSTEM_PROMPT}\n\nRepository code context:\n\n{context_str}"
    gen_kwargs = build_generation_kwargs(
        temperature    = GEN_TEMPERATURE,
        top_p          = GEN_TOP_P,
        max_new_tokens = GEN_MAX_OUTPUT_TOKENS,
    )

    # Step 5 — generate with retry on rate-limit
    for attempt in range(1, max_retries + 1):
        conversation = [{"role": "system", "content": system_content}]
        try:
            response = call_llm_with_context(question, conversation, **gen_kwargs)
            answer   = response["content"]
            break
        except Exception as exc:
            err_str = str(exc)
            if ("429" in err_str or "rate_limit" in err_str.lower()) and attempt < max_retries:
                wait = _parse_retry_seconds(err_str)
                print(f"    ⏳ Rate limit — waiting {wait:.0f}s (attempt {attempt}/{max_retries})…")
                time.sleep(wait)
            else:
                raise

    return {
        "question": question,
        "answer":   answer,
        "contexts": [h["text"] for h in hits],
    }


# ── LLM-as-judge scoring ──────────────────────────────────────────────────────

def _is_unanswerable(answer: str) -> bool:
    """Only skip if the answer is a pure refusal with no content at all."""
    a = answer.lower().strip()
    pure_refusals = (
        "no relevant content found",
        "i don't have enough information",
        "i cannot answer",
    )
    return any(a.startswith(p) for p in pure_refusals)


def run_evaluation(samples: list[dict], groq_api_key: str) -> tuple[dict, list[dict]]:
    """
    Three-metric LLM-as-judge evaluation via Groq.

    Per question (3 Groq calls):
      • faithfulness      — answer grounded in context?
      • answer_relevancy  — answer addresses the question?
      • context_precision — retrieved chunks relevant to question?

    Returns (scores_dict, rows_list).
    """
    from openai import OpenAI

    client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

    def _judge(prompt: str, max_retries: int = 3) -> float:
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model       = "llama-3.3-70b-versatile",
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = 5,
                    temperature = 0,
                )
                raw = resp.choices[0].message.content.strip()
                try:
                    return max(0.0, min(1.0, float(raw)))
                except ValueError:
                    m = re.search(r"[\d.]+", raw)
                    return float(m.group()) if m else 0.5
            except Exception as exc:
                err_str = str(exc)
                if ("429" in err_str or "rate_limit" in err_str.lower()) and attempt < max_retries:
                    wait = _parse_retry_seconds(err_str)
                    print(f"    ⏳ Judge rate limit — waiting {wait:.0f}s…")
                    time.sleep(wait)
                else:
                    raise
        return 0.5

    print(f"\n📊  Scoring {len(samples)} question(s) — 3 Groq calls each…\n")

    rows: list[dict] = []
    skipped = 0

    for i, s in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {s['question'][:70]}")

        if _is_unanswerable(s["answer"]):
            print("      ⏭  Skipped — LLM correctly refused (topic not in KB)")
            rows.append({
                "question":          s["question"],
                "faithfulness":      None,
                "answer_relevancy":  None,
                "context_precision": None,
                "skipped":           True,
            })
            skipped += 1
            continue

        ctx = "\n\n".join(s["contexts"][:6])[:3000]

        # ── 1. Faithfulness — is the answer grounded in the context?
        f_score = _judge(
            "Rate 0.0–1.0: how faithful is this ANSWER to the CONTEXT?\n"
            "1.0 = every claim is supported by context, 0.0 = answer contradicts/ignores context.\n"
            "Output ONLY a decimal number.\n\n"
            f"CONTEXT:\n{ctx}\n\nANSWER:\n{s['answer']}\n\nScore:"
        )

        # ── 2. Answer relevancy — does the answer address the question?
        r_score = _judge(
            "Rate 0.0–1.0: how well does this ANSWER address the QUESTION?\n"
            "1.0 = directly and completely answers it, 0.0 = answer is off-topic or evasive.\n"
            "Output ONLY a decimal number.\n\n"
            f"QUESTION: {s['question']}\n\nANSWER:\n{s['answer']}\n\nScore:"
        )

        # ── 3. Context precision — are the retrieved chunks relevant?
        p_score = _judge(
            "Rate 0.0–1.0: how relevant is this CONTEXT to answering the QUESTION?\n"
            "1.0 = context directly contains the answer, 0.0 = context is unrelated.\n"
            "Output ONLY a decimal number.\n\n"
            f"QUESTION: {s['question']}\n\nCONTEXT:\n{ctx}\n\nScore:"
        )

        print(f"      faithfulness={f_score:.2f}  "
              f"answer_relevancy={r_score:.2f}  "
              f"context_precision={p_score:.2f}")

        rows.append({
            "question":          s["question"],
            "faithfulness":      f_score,
            "answer_relevancy":  r_score,
            "context_precision": p_score,
            "skipped":           False,
        })

    scored = [r for r in rows if not r["skipped"]]
    if not scored:
        print("\n⚠  All questions skipped — no KB content found. Index your repos first.")
        sys.exit(1)

    if skipped:
        print(f"\n  ℹ  {skipped} question(s) skipped. Scores below cover {len(scored)} answerable question(s).")

    scores = {
        "faithfulness":      round(sum(r["faithfulness"]      for r in scored) / len(scored), 3),
        "answer_relevancy":  round(sum(r["answer_relevancy"]  for r in scored) / len(scored), 3),
        "context_precision": round(sum(r["context_precision"] for r in scored) / len(scored), 3),
    }
    return scores, rows


# ── Pretty report ─────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 25) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _grade(score: float) -> str:
    if score >= 0.85: return "🟢 Excellent"
    if score >= 0.70: return "🟡 Good"
    if score >= 0.50: return "🟠 Fair"
    return "🔴 Needs work"


def print_report(scores: dict, rows: list[dict], n_questions: int):
    scored = [r for r in rows if not r["skipped"]]
    avg    = sum(scores.values()) / len(scores)

    print("\n" + "═" * 62)
    print("  RAGAS-STYLE EVALUATION — PageMind RAG Pipeline")
    print("═" * 62)
    print(f"  Questions evaluated : {len(scored)} / {n_questions}")
    print(f"  Judge model         : llama-3.3-70b-versatile (Groq)")
    print(f"  Embedding model     : BAAI/bge-small-en-v1.5")
    print(f"  Retrieval           : hybrid BM25+HNSW → cross-encoder rerank")
    print("─" * 62)

    for metric, score in scores.items():
        print(f"  {metric:<22}  {_bar(score)}  {score:.3f}  {_grade(score)}")

    print("─" * 62)
    print(f"  {'Overall average':<22}  {_bar(avg)}  {avg:.3f}  {_grade(avg)}")
    print("═" * 62)

    print("\n  Metric explanations:")
    print("  • faithfulness       — answer only uses facts from retrieved context")
    print("  • answer_relevancy   — answer directly addresses the question asked")
    print("  • context_precision  — retrieved chunks are relevant to the question")

    print("\n  Improvement tips:")
    if scores.get("faithfulness", 1) < 0.70:
        print("  ⚠  Low faithfulness → LLM is hallucinating beyond the context.")
        print("     Fix: lower GEN_TEMPERATURE, strengthen the system prompt.")
    if scores.get("answer_relevancy", 1) < 0.70:
        print("  ⚠  Low answer_relevancy → answers are vague or off-topic.")
        print("     Fix: improve the system prompt to require direct answers.")
    if scores.get("context_precision", 1) < 0.70:
        print("  ⚠  Low context_precision → retriever pulling irrelevant chunks.")
        print("     Fix: lower CANDIDATE_K, raise ALPHA closer to 1.0 in retriever.py.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PageMind RAG pipeline (faithfulness, answer_relevancy, context_precision)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python evaluate_rag.py --username alice
              python evaluate_rag.py --username alice --questions questions.txt
              python evaluate_rag.py --username alice --delay 12
        """),
    )
    parser.add_argument("--username",  required=True,
                        help="PageMind username (must have indexed content)")
    parser.add_argument("--questions", default="questions.txt",
                        help="Path to .txt file, one question per line (default: questions.txt)")
    parser.add_argument("--output",    default="eval_results.json",
                        help="JSON output path (default: eval_results.json)")
    parser.add_argument("--delay",     type=float, default=8.0,
                        help="Seconds to wait between questions (default: 8, increase if hitting rate limits)")
    args = parser.parse_args()

    # ── Env check ──────────────────────────────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("❌  GROQ_API_KEY not found. Make sure backend/.env is loaded.")
        sys.exit(1)

    # ── Load questions ─────────────────────────────────────────────────────────
    q_path = args.questions
    if os.path.exists(q_path):
        with open(q_path, encoding="utf-8") as f:
            questions = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        print(f"📄  Loaded {len(questions)} question(s) from {q_path}")
    else:
        print(f"⚠   {q_path} not found — using 3 built-in fallback questions.")
        questions = [
            "What are the main API endpoints defined in the codebase?",
            "How is user authentication implemented?",
            "What database or storage layer is used and how is it accessed?",
        ]

    print(f"⏱   Inter-question delay: {args.delay}s  "
          f"(total Groq calls: {len(questions)} × 4 = {len(questions)*4})")

    # ── DB + user ──────────────────────────────────────────────────────────────
    init_db()
    user_row = get_user_by_username(args.username)
    if not user_row:
        print(f"❌  User '{args.username}' not found in pagemind.db.")
        print("    Register via the extension, then re-run.")
        sys.exit(1)
    user_id = user_row["id"]
    print(f"✅  User '{args.username}'  (id={user_id[:8]}…)")

    # ── Connect Weaviate ───────────────────────────────────────────────────────
    print("🔗  Connecting to Weaviate (localhost:8080)…")
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wv_client  = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
        collection = wv_client.collections.get("KnowledgeChunk")
        print("✅  Weaviate connected\n")
    except Exception as e:
        print(f"❌  Cannot connect to Weaviate: {e}")
        print("    Make sure Docker is running:  docker compose up -d")
        sys.exit(1)

    # ── Run RAG pipeline for each question ─────────────────────────────────────
    print("─" * 62)
    print(f"  Running RAG pipeline for {len(questions)} question(s)…")
    print("─" * 62)

    samples: list[dict] = []
    t0 = time.time()

    try:
        for i, q in enumerate(questions, 1):
            print(f"\n  [{i}/{len(questions)}] {q}")
            try:
                sample = run_rag_pipeline(q, collection, user_id)
            except Exception as exc:
                print(f"    ❌ Pipeline error: {exc} — skipping.")
                continue

            if not sample["contexts"]:
                print("    ⚠  No context retrieved — skipping.")
                continue

            print(f"    ✓ chunks={len(sample['contexts'])}  "
                  f"answer={sample['answer'][:80].rstrip()}…")
            samples.append(sample)

            # Rate-limit buffer between questions (skip after last)
            if i < len(questions):
                time.sleep(args.delay)
    finally:
        wv_client.close()

    elapsed = time.time() - t0

    if not samples:
        print("\n❌  No questions produced results. Index your repos first.")
        sys.exit(1)

    print(f"\n  RAG pipeline done in {elapsed:.1f}s  ({len(samples)} questions answered)")

    # ── Score with LLM judge ───────────────────────────────────────────────────
    scores, rows = run_evaluation(samples, groq_key)

    # ── Print report ───────────────────────────────────────────────────────────
    print_report(scores, rows, len(questions))

    # ── Save JSON results ──────────────────────────────────────────────────────
    row_map = {r["question"]: r for r in rows}
    output  = {
        "summary": scores,
        "per_question": [
            {
                "question":          s["question"],
                "answer":            s["answer"],
                "contexts_count":    len(s["contexts"]),
                "skipped":           row_map.get(s["question"], {}).get("skipped", False),
                "faithfulness":      row_map.get(s["question"], {}).get("faithfulness"),
                "answer_relevancy":  row_map.get(s["question"], {}).get("answer_relevancy"),
                "context_precision": row_map.get(s["question"], {}).get("context_precision"),
            }
            for s in samples
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"💾  Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
