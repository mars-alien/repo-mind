"""
PageMind LLM — Groq-backed generation, code-optimized
═══════════════════════════════════════════════════════════════════════
Models
──────
  GROQ_MODEL           llama-3.3-70b-versatile     — general / find intent
  GROQ_MODEL_REASONING deepseek-r1-distill-llama-70b — design intent (why/tradeoffs)

Code-optimized generation parameters
──────────────────────────────────────
  temperature     = 0.1   low → precise, reproducible code explanations
  top_p           = 0.9   nucleus sampling
  frequency_penalty = 0.3  stronger anti-repetition (code tends to repeat tokens)
  max_tokens      = 1024  enough for a thorough answer + sources section

System prompt
─────────────
  Instructs the model to behave as a senior software engineer,
  cite file paths and line numbers, and format as:
  "answer first, then Sources: file.py:L10-L25"

Public functions
────────────────
  get_model_for_intent(intent) → str
  build_generation_kwargs(...)  → dict
  generate_with_multiple_input(messages, model, **kwargs) → dict
  call_llm_with_context(prompt, context, role, model, **kwargs)  → dict
  stream_with_context(prompt, context, role, model, **kwargs)    → generator[str]
"""

from __future__ import annotations

import os
from groq import Groq

# ── Models ────────────────────────────────────────────────────────────────────

GROQ_MODEL           = "llama-3.3-70b-versatile"
GROQ_MODEL_REASONING = "deepseek-r1-distill-llama-70b"

# ── Code-optimized generation parameters ─────────────────────────────────────

GEN_TEMPERATURE       = 0.1   # low → precise code explanations
GEN_TOP_P             = 0.9
GEN_FREQUENCY_PENALTY = 0.3   # stronger anti-repetition for code
GEN_MAX_OUTPUT_TOKENS = 1024

# ── Code-focused system prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior software engineer helping developers understand a GitHub repository.\n"
    "Answer ONLY from the provided code context snippets.\n"
    "Be precise and concise. When referencing code, cite file paths and line numbers.\n"
    "Do not hallucinate functions, classes, or logic not present in the context.\n"
    "Format your response as:\n"
    "  1. Direct answer to the question\n"
    "  2. Sources: filepath:L<start>-L<end> (one per line)\n"
    "If the context does not contain the answer, say so clearly.\n"
    "You may use prior conversation turns to understand follow-up questions."
)


# ── Lazy singleton client ─────────────────────────────────────────────────────

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        _client = Groq(api_key=api_key)
    return _client


# ── Model selection ───────────────────────────────────────────────────────────

def get_model_for_intent(intent: str) -> str:
    """
    Route to the appropriate Groq model based on query intent.

    "design" intent (why, tradeoffs, architecture) → reasoning model
    everything else                                 → fast versatile model
    """
    return GROQ_MODEL_REASONING if intent == "design" else GROQ_MODEL


# ── 1. kwargs builder ─────────────────────────────────────────────────────────

def build_generation_kwargs(
    prompt:         str   = "",
    temperature:    float = GEN_TEMPERATURE,
    top_p:          float = GEN_TOP_P,
    max_new_tokens: int   = GEN_MAX_OUTPUT_TOKENS,
    **extra,
) -> dict:
    """
    Build a reusable kwargs dict for LLM generation functions.
    The returned dict is meant to be **-unpacked after popping `_model` if present.
    """
    return {
        "temperature":        temperature,
        "top_p":              top_p,
        "max_tokens":         max_new_tokens,
        "frequency_penalty":  GEN_FREQUENCY_PENALTY,
        **extra,
    }


# ── 2. Non-streaming multi-message call ──────────────────────────────────────

def generate_with_multiple_input(
    messages: list[dict],
    model:    str = GROQ_MODEL,
    **kwargs,
) -> dict:
    """
    Call Groq with a full messages list and return the assistant reply.

    Parameters
    ----------
    messages : Conversation history in OpenAI chat format
    model    : Groq model ID (use get_model_for_intent for routing)
    **kwargs : Generation parameters from build_generation_kwargs()
    """
    completion = _get_client().chat.completions.create(
        model    = model,
        messages = messages,
        stream   = False,
        **kwargs,
    )
    return {
        "role":    "assistant",
        "content": completion.choices[0].message.content or "",
    }


# ── 3. Stateful context-aware call ───────────────────────────────────────────

def call_llm_with_context(
    prompt:  str,
    context: list[dict],
    role:    str = "user",
    model:   str = GROQ_MODEL,
    **kwargs,
) -> dict:
    """
    Call the LLM with conversation history; update context in-place.

    Side-effects: appends user message + assistant reply to context.
    """
    context.append({"role": role, "content": prompt})
    response = generate_with_multiple_input(context, model=model, **kwargs)
    context.append(response)
    return response


# ── 4. Streaming variant for SSE endpoints ───────────────────────────────────

def stream_with_context(
    prompt:  str,
    context: list[dict],
    role:    str = "user",
    model:   str = GROQ_MODEL,
    **kwargs,
):
    """
    Streaming variant of call_llm_with_context for SSE endpoints.

    Yields text tokens as they arrive from Groq.
    Appends the complete assembled response to context when done
    (even on early generator close / client disconnect).

    Parameters
    ----------
    prompt  : New user input
    context : Conversation history (mutated in-place)
    role    : "user" or "assistant"
    model   : Groq model ID — use get_model_for_intent() for routing
    **kwargs: Generation params from build_generation_kwargs()
    """
    context.append({"role": role, "content": prompt})

    full_content = ""
    try:
        stream = _get_client().chat.completions.create(
            model    = model,
            messages = context,
            stream   = True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_content += delta
                yield delta
    finally:
        if full_content:
            context.append({"role": "assistant", "content": full_content})
