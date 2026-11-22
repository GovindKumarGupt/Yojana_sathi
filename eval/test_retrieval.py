"""Retrieval test: is the correct page in the top-k chunks?

Metrics:
  recall@k  -> % of questions where the right page was found
  MRR       -> how high the right page ranks (1.0 = always first)

Run from the project folder:  python eval/test_retrieval.py
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_SCHEME, TOP_K
from retriever import retrieve
from memory import rewrite_question

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "test_questions.json"), encoding="utf-8") as f:
    tests = [t for t in json.load(f) if t.get("expected_page") is not None]

hits, rr_total = 0, 0.0
for t in tests:
    # same as the real pipeline: rewrite (translate Hindi, resolve follow-ups) before searching
    scheme = t.get("scheme", DEFAULT_SCHEME)
    full_q = rewrite_question(t["question"], t.get("history", []), scheme)
    chunks = retrieve(full_q, scheme, top_k=TOP_K)
    rank = None
    for i, c in enumerate(chunks, start=1):
        if c["file_name"] == t["expected_file"] and c["page"] == t["expected_page"]:
            rank = i
            break
    if rank:
        hits += 1
        rr_total += 1 / rank
    found = [f"{c['file_name']} p{c['page']} ({c['score']})" for c in chunks]
    print(f"{'HIT ' if rank else 'MISS'} [{t['id']}] {t['question']}")
    if not rank:
        print("      expected:", t["expected_file"], "p", t["expected_page"], "| got:", found)

n = len(tests)
print(f"\nrecall@{TOP_K}: {hits}/{n} = {hits / n:.0%}")
print(f"MRR: {rr_total / n:.2f}")
