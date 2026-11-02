"""
show_errors.py - prints the real examples needed for Table VII of the report.

    python show_errors.py

It reads results/qa_raw.jsonl and results/stress_results.csv (nothing is run again, no tokens used).
"""

import csv
import json
import os

from evaluate import is_refusal

RAW = os.path.join("results", "qa_raw.jsonl")
STRESS = os.path.join("results", "stress_results.csv")


def short(s, n=300):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " ..."


def main():
    recs = []
    if os.path.isfile(RAW):
        with open(RAW, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
    test = [r for r in recs if r.get("split") == "test"]
    print(f"Test records: {len(test)}\n")

    print("=== 1. LLM alone answered an unanswerable question (show 2) ===")
    shown = 0
    for r in test:
        if not r["answerable"] and not is_refusal(r["llm_only"]["text"]):
            print("Q:", r["question"])
            print("LLM answer:", short(r["llm_only"]["text"]), "\n")
            shown += 1
            if shown == 2:
                break

    print("=== 2. LLM alone gave a wrong answer to an answerable question (show 2) ===")
    shown = 0
    for r in test:
        if r["answerable"] and not is_refusal(r["llm_only"]["text"]) and r["llm_only"]["verdict"] == "incorrect":
            print("Q:", r["question"])
            print("Reference:", short(r["reference"]))
            print("LLM answer:", short(r["llm_only"]["text"]), "\n")
            shown += 1
            if shown == 2:
                break

    print("=== 3. Hybrid RAG answered an unanswerable question ===")
    for r in test:
        if not r["answerable"] and not is_refusal(r["rag_hybrid"]["text"]):
            print("Q:", r["question"])
            print("Hybrid answer:", short(r["rag_hybrid"]["text"]))
            v = r["verification"]
            print(f"Retrieval strength: {v['strength']:.2f}, supported fraction: {v['supported_fraction']:.2f}\n")

    print("=== 4-6. Stress test: verifier misses and false alarm ===")
    if not os.path.isfile(STRESS):
        print("results/stress_results.csv not found.")
        return
    with open(STRESS, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    groups = [
        ("4. Missed a changed fact (answers)", "answer", "contradicted", "0"),
        ("5. Missed a fabricated explanation (MCQ)", "mcq", "fake_expl", "0"),
        ("6. Wrongly rejected a correct MCQ", "mcq", "clean", "1"),
    ]
    for title, part, kind, flag in groups:
        hits = [r for r in rows if r["part"] == part and r["type"] == kind and r["flag_any"] == flag]
        print(title)
        if not hits:
            print("  (no such row saved; see the note in the report help)\n")
        for r in hits[:1]:
            print("  Q:", short(r["question"]))
            print("  Text:", short(r["text"]), "\n")


if __name__ == "__main__":
    main()