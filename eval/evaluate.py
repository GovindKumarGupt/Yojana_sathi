"""End-to-end evaluation of the full pipeline (guardrails + RAG + LLM).

Metrics printed:
  answer accuracy   -> LLM-as-judge: does the answer match the expected answer?
  refusal accuracy  -> did the bot refuse exactly when it should (off-topic, injection, not in docs)?
  groundedness rate -> % of answers that passed the groundedness check
  avg latency       -> seconds per question

Results for every question are saved to eval/results.csv.
Run from the project folder:  python eval/evaluate.py
"""
import os
import sys
import csv
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL, DEFAULT_SCHEME, FALLBACK_ANSWER
from rag_pipeline import ask

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
HERE = os.path.dirname(os.path.abspath(__file__))


def judge(question, expected, answer):
    """LLM-as-judge: YES if the bot answer means the same as the expected answer."""
    prompt = f"""You are grading a chatbot.
Question: {question}
Expected answer: {expected}
Bot answer: {answer}

Does the bot answer give the same key facts as the expected answer?
Extra correct detail is fine. Language (Hindi/English) does not matter.
Reply with only YES or NO."""
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2,
    )
    return r.choices[0].message.content.strip().upper().startswith("YES")


def is_refusal(result):
    return result["blocked_reason"] is not None or result["answer"].strip() == FALLBACK_ANSWER


def main():
    with open(os.path.join(HERE, "test_questions.json"), encoding="utf-8") as f:
        tests = json.load(f)

    rows = []
    for t in tests:
        result = ask(t["question"], t.get("scheme", DEFAULT_SCHEME), history=list(t.get("history", [])))  # follow-up tests carry earlier chat
        refused = is_refusal(result)
        refusal_ok = refused == t["should_refuse"]
        correct = refusal_ok if t["should_refuse"] else (
            not refused and judge(t["question"], t["expected"], result["answer"])
        )
        rows.append({
            "id": t["id"], "type": t["type"], "question": t["question"],
            "correct": correct, "refusal_ok": refusal_ok,
            "blocked_reason": result["blocked_reason"], "grounded": result["grounded"],
            "latency_s": result["latency_s"], "answer": result["answer"],
        })
        print(f"{'PASS' if correct else 'FAIL'} [{t['id']}|{t['type']}] {t['question']}")
        if not correct:
            print("      bot:", result["answer"][:200].replace("\n", " "))

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    ref = sum(r["refusal_ok"] for r in rows) / n
    checked = [r for r in rows if r["grounded"] is not None]
    grounded = sum(r["grounded"] for r in checked) / len(checked) if checked else 0
    latency = sum(r["latency_s"] for r in rows) / n

    print("\n========== RESULTS ==========")
    print(f"Answer accuracy   : {acc:.0%}  ({sum(r['correct'] for r in rows)}/{n})")
    print(f"Refusal accuracy  : {ref:.0%}")
    print(f"Groundedness rate : {grounded:.0%}  (of {len(checked)} checked answers)")
    print(f"Avg latency       : {latency:.1f}s")

    out = os.path.join(HERE, "results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved details to {out}")


if __name__ == "__main__":
    main()
