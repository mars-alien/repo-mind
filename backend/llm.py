"""
PageMind LLM — Groq-backed generation with context-aware conversation
═══════════════════════════════════════════════════════════════════════

Model: llama-3.3-70b-versatile  (Groq free tier)
  - 70 B parameters, instruction-tuned
  - 128 K token context window
  - ~250 tokens/s on Groq's LPU inference

Three public functions
──────────────────────

1. build_generation_kwargs(prompt, temperature, top_p, max_new_tokens, **extra)
       → dict
   Returns a parameter dict you can **-unpack into the other two functions.
   Decouples "what parameters to use" from "how to call the model."

2. generate_with_multiple_input(messages, **kwargs)
       → {"role": "assistant", "content": "..."}
   Single (non-streaming) call to Groq with a full messages list.
   Used for simple one-off generation and unit-testable context building.

3. call_llm_with_context(prompt, context, role, **kwargs)
       → {"role": "assistant", "content": "..."}
   Recursive multi-turn conversation helper.
   Appends the user message to context, calls generate_with_multiple_input,
   then appends the assistant reply — so the next call remembers all prior
   turns without any extra bookkeeping by the caller.

   Example
   -------
   context = [{"role": "system", "content": "You are a helpful guide."}]
   r1 = call_llm_with_context("Name two great cities.", context)
   # context → system + user + assistant (3 messages)
   r2 = call_llm_with_context("Tell me more about the first one.", context)
   # LLM knows which city was first — full history preserved

4. stream_with_context(prompt, context, role, **kwargs)
       → generator[str]  (each yield = one text chunk)
   Streaming variant for SSE endpoints.
   Yields text tokens as they arrive from Groq.
   When the stream ends (normally or via generator.close()), the completed
   response is appended to context — same side-effect as call_llm_with_context.
"""

from __future__ import annotations

import os
from groq import Groq

# ── Model & defaults ──────────────────────────────────────────────────────────

GROQ_MODEL            = "llama-3.3-70b-versatile"

#  temperature = 0.8
#    Controls randomness of token sampling.
#    0.0 = fully deterministic; 1.0 = sample proportional to raw probs.
#    0.8 → slightly creative but still coherent answers.
#
#  top_p = 0.9  (nucleus sampling)
#    Keep only the tokens whose cumulative probability ≥ top_p, then sample.
#    0.9 → trims the low-probability / incoherent tail; preserves variety.
#
#  frequency_penalty = 0.2
#    Penalises tokens proportional to how often they've already appeared.
#    Prevents phrase-level repetition. Groq range: -2.0 … +2.0.
#    Equivalent to HuggingFace repetition_penalty ≈ 1.2.
#
#  max_tokens = 2048
#    Hard ceiling on response length (≈ 1 500 words). Enough for a detailed
#    answer without a runaway output.

GEN_TEMPERATURE       = 0.8     # creative yet coherent
GEN_TOP_P             = 0.9     # nucleus sampling — trims low-prob tail
GEN_FREQUENCY_PENALTY = 0.2     # anti-repetition (HF repetition_penalty ≈ 1.2)
GEN_MAX_OUTPUT_TOKENS = 2048    # answer length ceiling


# ── Lazy singleton client ─────────────────────────────────────────────────────

_client: Groq | None = None


def _get_client() -> Groq:
    """Return a reusable Groq client. Raises RuntimeError if API key missing."""
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

    More flexible than hardcoding parameters at each call site: callers can
    customise any parameter and **-unpack the result directly into
    generate_with_multiple_input() or stream_with_context().

    Parameters
    ----------
    prompt         : Input text (for documentation only — NOT included in dict)
    temperature    : Randomness; lower = more deterministic
    top_p          : Nucleus sampling; higher = more varied outputs
    max_new_tokens : Maximum tokens in the response (Groq: max_tokens)
    **extra        : Any additional Groq parameters (e.g. stop=["\\n"], seed=42)

    Returns
    -------
    dict  — ready to **-unpack into generate_* functions

    Examples
    --------
    # Default parameters
    kwargs = build_generation_kwargs("What is Python?")

    # Custom parameters — more deterministic, shorter output
    kwargs = build_generation_kwargs(temperature=0.2, max_new_tokens=512)

    # With extra Groq params
    kwargs = build_generation_kwargs(stop=["END"], seed=42)

    # Feed into a generation function
    response = call_llm_with_context(prompt, context, **kwargs)
    """
    return {
        "temperature":        temperature,
        "top_p":              top_p,
        "max_tokens":         max_new_tokens,   # Groq parameter name
        "frequency_penalty":  GEN_FREQUENCY_PENALTY,
        **extra,
    }


# ── 2. Non-streaming multi-message call ──────────────────────────────────────

def generate_with_multiple_input(messages: list[dict], **kwargs) -> dict:
    """
    Call Groq with a full messages list and return the assistant reply.

    Enables multi-turn conversations by accepting the entire conversation
    history (system + prior turns + current user message) in one call.

    Parameters
    ----------
    messages : Conversation history in OpenAI chat format
               [{"role": "system"|"user"|"assistant", "content": "..."},
                ...]
    **kwargs : Generation parameters from build_generation_kwargs()

    Returns
    -------
    dict  {"role": "assistant", "content": "..."}
          Ready to append directly to the context list.

    Notes
    -----
    This function does NOT modify the messages list.
    Use call_llm_with_context() for the mutating, stateful version.
    """
    completion = _get_client().chat.completions.create(
        model    = GROQ_MODEL,
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
    **kwargs,
) -> dict:
    """
    Call the LLM with conversation history and update the context in-place.

    Implements a recursive conversation pattern where each exchange is
    appended to `context` so follow-up questions have access to all
    prior turns without any extra bookkeeping by the caller.

    Parameters
    ----------
    prompt  : New user input to add to the conversation
    context : Conversation history (mutated in-place)
              e.g. [{"role": "system", "content": "..."},
                    {"role": "user",      "content": "prev question"},
                    {"role": "assistant", "content": "prev answer"}]
    role    : Role label for this message — "user" or "assistant"
    **kwargs: Generation params from build_generation_kwargs()

    Returns
    -------
    dict  {"role": "assistant", "content": "..."}

    Side-effects
    ------------
    context is updated in-place with:
      1. The new user message  (appended before the call)
      2. The assistant response (appended after the call)

    Subsequent calls receive the full history automatically.

    Example
    -------
    context = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    kwargs = build_generation_kwargs(temperature=0.7)

    r1 = call_llm_with_context("Name two great cities to visit.", context, **kwargs)
    # context now has: system + user + assistant (3 messages)

    r2 = call_llm_with_context("Tell me more about the first one.", context, **kwargs)
    # LLM knows which city was first — full history preserved
    # context now has 5 messages
    """
    # Step 1: Append the new user message so it's part of the history
    context.append({"role": role, "content": prompt})

    # Step 2: Call Groq with the full message history
    response = generate_with_multiple_input(context, **kwargs)

    # Step 3: Append the assistant reply so the next call sees it
    context.append(response)

    return response


# ── 4. Streaming variant for SSE endpoints ───────────────────────────────────

def stream_with_context(
    prompt:  str,
    context: list[dict],
    role:    str = "user",
    **kwargs,
):
    """
    Streaming variant of call_llm_with_context for SSE (Server-Sent Events).

    Yields text chunks as they arrive from Groq token-by-token.
    When the stream ends (or the generator is closed early), the complete
    assembled response is appended to context — same side-effect as
    call_llm_with_context.

    Parameters
    ----------
    prompt  : New user input
    context : Conversation history (mutated in-place)
    role    : "user" or "assistant"
    **kwargs: Generation params from build_generation_kwargs()

    Yields
    ------
    str  — individual text chunks from the LLM as they arrive

    Side-effects
    ------------
    When stream completes (or generator is closed), appends:
      1. The user message (prompt) to context
      2. The complete assembled assistant response to context

    Usage in FastAPI SSE
    --------------------
    def stream_answer():
        for chunk in stream_with_context(question, conversation, **gen_kwargs):
            yield f"data: {json.dumps({'text': chunk})}\\n\\n"
        yield f"data: {json.dumps({'sources': sources})}\\n\\n"
        yield "data: [DONE]\\n\\n"
    """
    # Append user message before streaming starts
    context.append({"role": role, "content": prompt})

    full_content = ""
    try:
        stream = _get_client().chat.completions.create(
            model    = GROQ_MODEL,
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
        # Always persist whatever was generated — even if the generator
        # was closed early (e.g. client disconnected mid-stream)
        if full_content:
            context.append({"role": "assistant", "content": full_content})
