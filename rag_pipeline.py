"""Connects everything:

question
  -> input guardrails (PII mask, injection, off-topic)
  -> memory (rewrite follow-up into a full English question)
  -> retrieve top chunks
  -> relevance check (nothing relevant -> fallback, no LLM guessing)
  -> generate answer
  -> groundedness check (unsupported -> fallback)
  -> save to history + log (latency, flags)
"""
import os
import json
import time
from datetime import datetime

from config import (
    DEFAULT_SCHEME, FALLBACK_ANSWER, GROUNDEDNESS_CHECK, LOG_FILE,
)
from guardrails import check_input, has_relevant_chunks, check_groundedness
from memory import rewrite_question
from retriever import retrieve
from generator import generate_answer


def log_query(record):
    """Append one line of JSON per question (for debugging and monitoring)."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ask(question, scheme, history, use_groundedness=GROUNDEDNESS_CHECK):
    """Answer one question. `history` is a list that this function updates.

    Returns a dict with: answer, sources, blocked_reason, grounded, rewritten, latency_s
    """
    start = time.time()
    result = {"answer": None, "sources": [], "blocked_reason": None,
              "grounded": None, "rewritten": None, "pii_masked": []}

    # 1) input guardrails
    check = check_input(question, scheme=scheme)
    result["pii_masked"] = check["pii_found"]
    if not check["ok"]:
        result["answer"] = check["message"]
        result["blocked_reason"] = check["reason"]
    else:
        safe_question = check["question"]

        # 2) memory: rewrite follow-ups into a full English question
        full_q = rewrite_question(safe_question, history, scheme)
        result["rewritten"] = full_q

        # 3) retrieve
        chunks = retrieve(full_q, scheme)

        # 4) relevance guardrail
        if not has_relevant_chunks(chunks):
            result["answer"] = FALLBACK_ANSWER
            result["blocked_reason"] = "no_relevant_chunks"
        else:
            # 5) generate
            gen = generate_answer(full_q, scheme, original_question=safe_question, chunks=chunks)
            answer = gen["answer"]
            result["sources"] = sorted({(c["file_name"], c["page"]) for c in chunks})

            # 6) groundedness guardrail
            if use_groundedness and answer != FALLBACK_ANSWER:
                g = check_groundedness(answer, gen["context"])
                result["grounded"] = g["grounded"]
                if not g["grounded"]:
                    answer = FALLBACK_ANSWER
                    result["blocked_reason"] = "not_grounded"
                    result["unsupported"] = g["unsupported"]
            result["answer"] = answer

    result["latency_s"] = round(time.time() - start, 2)

    # 7) save to memory (store the masked question, never the raw personal data)
    history.append({"role": "user", "content": check["question"] or question})
    history.append({"role": "assistant", "content": result["answer"]})

    # 8) log
    log_query({
        "time": datetime.now().isoformat(timespec="seconds"),
        "scheme": scheme,
        "question": check["question"],
        "rewritten": result["rewritten"],
        "blocked_reason": result["blocked_reason"],
        "grounded": result["grounded"],
        "pii_masked": result["pii_masked"],
        "sources": result["sources"],
        "latency_s": result["latency_s"],
    })
    return result


if __name__ == "__main__":
    history = []
    print(f"Ask about {DEFAULT_SCHEME} (English or Hindi). Type 'exit' to stop.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        r = ask(q, DEFAULT_SCHEME, history)
        print("\nBot:", r["answer"])
        print(f"  [rewritten: {r['rewritten']} | blocked: {r['blocked_reason']} | "
              f"grounded: {r['grounded']} | {r['latency_s']}s]\n")
