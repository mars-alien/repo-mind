"""
PageMind RAG Evaluation — RAGAS
=================================
Measures the quality of the full RAG pipeline using three core metrics:

  faithfulness       — Is the answer grounded in the retrieved context?
                       Score 1.0 = every claim is supported by context.

  answer_relevancy   — Does the answer actually address the question?
                       Score 1.0 = perfectly on-topic answer.

  context_precision  — Are the retrieved chunks relevant to the question?
                       Score 1.0 = all retrieved chunks are useful.

How it works
────────────
  For each test question the script runs the FULL real pipeline:
    embed_query()  →  hybrid_retrieve()  →  call_llm_with_context()
  Then RAGAS uses Groq (same LLM you use in production) to judge the results.
  No ground-truth answers needed for these three metrics.

Prerequisites
─────────────
  1.  Docker + Weaviate running:   docker compose up -d
  2.  Backend venv active:         .venv\\Scripts\\activate
  3.  Extra package:               pip install langchain-openai
  4.  At least one source indexed via the extension.

Usage
─────
  # Use default generic questions
  python evaluate_rag.py --username YOUR_USERNAME

  # Use your own questions (one per line in a .txt file)
  python evaluate_rag.py --username YOUR_USERNAME --questions my_questions.txt

  # Increase question count for a thorough eval
  python evaluate_rag.py --username YOUR_USERNAME --top-k 10
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
    from embedder  import embed_query, embed_texts
    from retriever import hybrid_retrieve, TOP_K
    from llm       import (
        build_generation_kwargs, call_llm_with_context,
        GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_OUTPUT_TOKENS,
    )
except ImportError as e:
    print(f"\n❌  Import error: {e}")
    print("    Make sure you run this script from the backend/ directory:")
    print("    cd backend && python evaluate_rag.py --username YOUR_USERNAME\n")
    sys.exit(1)


# ── Default test questions ─────────────────────────────────────────────────────
# Edit these to match the content you have indexed, or pass --questions file.txt

DEFAULT_QUESTIONS = [
    "What are the main topics covered in the knowledge base?",
    "Summarise the key concepts from the indexed sources.",
    "What are the most important facts mentioned in the documents?",
    "What conclusions can be drawn from the indexed material?",
    "List the key points from the content that has been indexed.",
]


# ── RAG pipeline (mirrors main.py /query, non-streaming) ─────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful knowledge assistant. Answer the user's question using ONLY \
the context provided below. Be concise and accurate. If the context does not \
contain enough information to answer, say so clearly.

Context:
{context}
"""


def _parse_retry_seconds(error_msg: str) -> float:
    """Extract wait seconds from Groq rate-limit message like 'try again in 19m4.8s'."""
    m = re.search(r'try again in\s+(?:(\d+)m)?([\d.]+)s', str(error_msg))
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2) or 0)
        return minutes * 60 + seconds + 5   # +5 s safety buffer
    return 90   # default fallback


def run_rag_pipeline(question: str, collection, user_id: str,
                     max_retries: int = 3) -> dict:
    """
    Runs the full RAG pipeline for one question.
    Automatically waits and retries on Groq rate-limit (429) errors.
    Returns { question, answer, contexts } where contexts is list[str].
    """
    # Step 1 — embed
    q_vec = embed_query(question)

    # Step 2 — hybrid retrieve (BM25 + HNSW)
    hits = hybrid_retrieve(
        collection   = collection,
        query_text   = question,
        query_vector = q_vec,
        user_id      = user_id,
        top_k        = TOP_K,
    )

    if not hits:
        return {
            "question": question,
            "answer":   "No relevant content found in the knowledge base.",
            "contexts": [],
        }

    # Step 3 — build context string (same logic as main.py)
    context_parts: list[str] = []
    for hit in hits:
        label  = hit.get("title") or hit.get("url") or "Source"
        prefix = f"[{label}]"
        if hit.get("heading_path"):
            prefix += f" [{hit['heading_path']}]"
        context_parts.append(f"{prefix}\n{hit['text']}")

    context_str    = "\n\n---\n\n".join(context_parts)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
    gen_kwargs     = build_generation_kwargs(
        temperature    = GEN_TEMPERATURE,
        top_p          = GEN_TOP_P,
        max_new_tokens = GEN_MAX_OUTPUT_TOKENS,
    )

    # Step 4 — generate with retry on rate-limit
    for attempt in range(1, max_retries + 1):
        # Rebuild conversation fresh each attempt (call_llm_with_context mutates it)
        conversation = [{"role": "system", "content": system_content}]
        try:
            response = call_llm_with_context(question, conversation, **gen_kwargs)
            answer   = response["content"]
            break
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries:
                wait = _parse_retry_seconds(err_str)
                print(f"\n  ⏳ Rate limit hit. Waiting {wait:.0f}s then retrying "
                      f"(attempt {attempt}/{max_retries})…")
                time.sleep(wait)
            else:
                raise   # non-rate-limit error, or out of retries

    # RAGAS needs raw chunk texts (not the formatted label+text strings)
    raw_contexts = [hit["text"] for hit in hits]

    return {
        "question": question,
        "answer":   answer,
        "contexts": raw_contexts,
    }


# ── RAGAS evaluation — ragas 0.2.x API ────────────────────────────────────────

def _patch_langchain_community():
    """
    Stubs out two modules that ragas hardcodes but langchain-community >= 0.4
    removed.  We never call VertexAI so an empty class is enough.
    """
    import types as _t
    for _key in (
        "langchain_community.chat_models.vertexai",
        "langchain_community.llms.vertexai",
    ):
        if _key not in sys.modules:
            _m = _t.ModuleType(_key)
            _m.ChatVertexAI = type("ChatVertexAI", (), {})
            _m.VertexAI     = type("VertexAI",     (), {})
            sys.modules[_key] = _m

    # ragas also does: from langchain_community.llms import VertexAI
    try:
        import langchain_community.llms as _lc
        if not hasattr(_lc, "VertexAI"):
            _lc.VertexAI = type("VertexAI", (), {})
    except Exception:
        pass


def run_ragas_evaluation(samples: list[dict], groq_api_key: str):
    """
    Evaluates samples with RAGAS 0.2.x API.

    Metrics used (none require ground-truth / reference):
      • faithfulness                        — answer grounded in context?
      • answer_relevancy                    — answer on-topic?
      • LLMContextPrecisionWithoutReference — retrieved chunks relevant?

    Returns (scores_dict, dataframe).
    """
    _patch_langchain_community()

    # ── Imports (ragas 0.2.x paths) ───────────────────────────────────────────
    try:
        from openai          import OpenAI
        from ragas.llms      import llm_factory
        from ragas           import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics.collections import (
            faithfulness,
            ResponseRelevancy           as answer_relevancy,
            LLMContextPrecisionWithoutReference,
        )
        _NEW_API = True
    except ImportError:
        # Graceful fallback: old 0.1.x / early 0.2.x dataset API
        _NEW_API = False
        try:
            from openai          import OpenAI
            from ragas.llms      import llm_factory
            from datasets        import Dataset
            from ragas           import evaluate
            from ragas.metrics.collections import (
                faithfulness,
                ResponseRelevancy           as answer_relevancy,
                LLMContextPrecisionWithoutReference,
            )
        except ImportError as _e:
            print(f"\n❌  Import error: {_e}")
            print("    pip install ragas datasets langchain-openai openai\n")
            sys.exit(1)

    # ── Build Groq client via ragas llm_factory ────────────────────────────────
    print("\n⚙   Configuring RAGAS (LLM = Groq llama-3.3-70b | embeddings = bge-m3)…")

    groq_client = OpenAI(
        api_key  = groq_api_key,
        base_url = "https://api.groq.com/openai/v1",
    )
    ragas_llm = llm_factory("llama-3.3-70b-versatile", client=groq_client)

    # ── BGE-M3 embeddings wrapper (for answer_relevancy cosine similarity) ─────
    from ragas.embeddings import BaseRagasEmbeddings

    class BGEEmbeddings(BaseRagasEmbeddings):
        def embed_query(self, text):             return embed_query(text)
        def embed_documents(self, texts):        return embed_texts(texts)
        async def aembed_query(self, text):      return self.embed_query(text)
        async def aembed_documents(self, texts): return self.embed_documents(texts)

    ragas_emb = BGEEmbeddings()

    # ── Wire LLM + embeddings into every metric ────────────────────────────────
    ctx_precision = LLMContextPrecisionWithoutReference()
    for metric in [faithfulness, answer_relevancy, ctx_precision]:
        metric.llm        = ragas_llm
        metric.embeddings = ragas_emb

    metrics = [faithfulness, answer_relevancy, ctx_precision]

    print(f"📊  Evaluating {len(samples)} question(s) — this may take 1–3 min…\n")

    # ── Build dataset and evaluate ─────────────────────────────────────────────
    if _NEW_API:
        ragas_samples = [
            SingleTurnSample(
                user_input        = s["question"],
                response          = s["answer"],
                retrieved_contexts= s["contexts"],
            )
            for s in samples
        ]
        dataset = EvaluationDataset(samples=ragas_samples)
    else:
        from datasets import Dataset
        dataset = Dataset.from_list([
            {"question": s["question"], "answer": s["answer"], "contexts": s["contexts"]}
            for s in samples
        ])

    result = evaluate(dataset=dataset, metrics=metrics)
    df     = result.to_pandas()

    # Column names differ slightly between API versions — normalise
    col_map = {
        "faithfulness":                         "faithfulness",
        "answer_relevancy":                     "answer_relevancy",
        "response_relevancy":                   "answer_relevancy",   # new name
        "llm_context_precision_without_reference": "context_precision",
        "context_precision":                    "context_precision",
    }
    scores: dict[str, float] = {}
    for raw_col in df.columns:
        canon = col_map.get(raw_col)
        if canon:
            scores[canon] = float(df[raw_col].mean())

    return scores, df


# ── Pretty printing ───────────────────────────────────────────────────────────

def _bar(score: float, width: int = 25) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _grade(score: float) -> str:
    if score >= 0.85: return "🟢 Excellent"
    if score >= 0.70: return "🟡 Good"
    if score >= 0.50: return "🟠 Fair"
    return "🔴 Needs work"


def print_report(scores: dict, samples: list[dict]):
    print("\n" + "═" * 60)
    print("  RAGAS EVALUATION RESULTS — PageMind RAG Pipeline")
    print("═" * 60)
    print(f"  Questions evaluated : {len(samples)}")
    print(f"  Model               : llama-3.3-70b-versatile (Groq)")
    print(f"  Embeddings          : BAAI/bge-m3")
    print("─" * 60)
    for metric, score in scores.items():
        print(f"  {metric:<22}  {_bar(score)}  {score:.3f}  {_grade(score)}")
    print("═" * 60)

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Overall average: {avg:.3f}  {_grade(avg)}")

    print("\n  Metric explanations:")
    print("  • faithfulness      — answers only use facts from retrieved context")
    print("  • answer_relevancy  — answers actually address the question asked")
    print("  • context_precision — retrieved chunks are relevant to the question")

    print("\n  Improvement tips:")
    if scores.get("faithfulness", 1) < 0.7:
        print("  ⚠  Low faithfulness → LLM is hallucinating beyond the context.")
        print("     Try: stronger system prompt, reduce GEN_TEMPERATURE to 0.5.")
    if scores.get("answer_relevancy", 1) < 0.7:
        print("  ⚠  Low answer_relevancy → answers drift off-topic.")
        print("     Try: more specific questions, add 'Answer concisely' to prompt.")
    if scores.get("context_precision", 1) < 0.7:
        print("  ⚠  Low context_precision → retrieving too many irrelevant chunks.")
        print("     Try: reduce TOP_K in retriever.py, increase ALPHA closer to 1.0.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PageMind RAG pipeline with RAGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python evaluate_rag.py --username alice
              python evaluate_rag.py --username alice --questions my_qs.txt
        """),
    )
    parser.add_argument("--username",  required=True,
                        help="Your PageMind username (must have indexed content)")
    parser.add_argument("--questions", default=None,
                        help="Path to a .txt file with one question per line")
    parser.add_argument("--output",    default="eval_results.json",
                        help="Where to save the JSON results (default: eval_results.json)")
    args = parser.parse_args()

    # ── Env check ──────────────────────────────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("❌  GROQ_API_KEY not found. Make sure backend/.env is loaded.")
        sys.exit(1)

    # ── Load questions ─────────────────────────────────────────────────────────
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        print(f"📄  Loaded {len(questions)} question(s) from {args.questions}")
    else:
        questions = DEFAULT_QUESTIONS
        print(f"ℹ   No --questions file provided. Using {len(questions)} default questions.")
        print("    Tip: create a questions.txt with questions specific to your indexed content.")

    # ── DB + user ──────────────────────────────────────────────────────────────
    init_db()
    user_row = get_user_by_username(args.username)
    if not user_row:
        print(f"❌  User '{args.username}' not found in pagemind.db.")
        print("    Register first by opening the extension and creating an account.")
        sys.exit(1)
    user_id = user_row["id"]
    print(f"✅  User '{args.username}' found (id={user_id[:8]}…)")

    # ── Connect Weaviate ───────────────────────────────────────────────────────
    print("🔗  Connecting to Weaviate (localhost:8080)…")
    try:
        wv_client  = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
        collection = wv_client.collections.get("KnowledgeChunk")
        print("✅  Weaviate connected\n")
    except Exception as e:
        print(f"❌  Cannot connect to Weaviate: {e}")
        print("    Make sure Docker is running:  docker compose up -d")
        sys.exit(1)

    # ── Run RAG pipeline for each question ─────────────────────────────────────
    print("─" * 60)
    print(f"  Running RAG pipeline for {len(questions)} question(s)…")
    print("─" * 60)

    samples: list[dict] = []
    t0 = time.time()

    try:
        for i, q in enumerate(questions, 1):
            print(f"\n  [{i}/{len(questions)}] {q}")
            sample = run_rag_pipeline(q, collection, user_id)

            if not sample["contexts"]:
                print("           ⚠  No context retrieved — skipping this question.")
                continue

            print(f"           ✓ Contexts: {len(sample['contexts'])} chunks")
            print(f"           ✓ Answer:   {sample['answer'][:90].rstrip()}…")
            samples.append(sample)
    finally:
        wv_client.close()   # always close — even if an exception occurs

    elapsed = time.time() - t0

    if not samples:
        print("\n❌  No questions produced results. Make sure you have indexed content.")
        sys.exit(1)

    print(f"\n  RAG pipeline finished in {elapsed:.1f}s for {len(samples)} question(s).")

    # ── RAGAS evaluation ────────────────────────────────────────────────────────
    scores, df = run_ragas_evaluation(samples, groq_key)

    # ── Print report ────────────────────────────────────────────────────────────
    print_report(scores, samples)

    # ── Save results ────────────────────────────────────────────────────────────
    def _safe(col, idx):
        return float(df[col].iloc[idx]) if col in df.columns and idx < len(df) else None

    output = {
        "summary":      scores,
        "per_question": [
            {
                "question":          s["question"],
                "answer":            s["answer"],
                "contexts_count":    len(s["contexts"]),
                "faithfulness":      _safe("faithfulness", i),
                "answer_relevancy":  _safe("answer_relevancy",  i) or _safe("response_relevancy", i),
                "context_precision": _safe("context_precision", i) or _safe("llm_context_precision_without_reference", i),
            }
            for i, s in enumerate(samples)
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾  Full results saved → {args.output}\n")


if __name__ == "__main__":
    main()
