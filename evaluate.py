"""
evaluate.py
Produces the results tables for the CampusGPT paper.

QA evaluation (answers four systems on your test questions):
    python evaluate.py qa --questions eval_questions.csv

Quiz evaluation (MCQs before vs after verification):
    python evaluate.py quiz --file YourFile.pdf --n 8 --sections 3

Metrics are judged by an INDEPENDENT judge model (JUDGE_MODEL in .env, default
openai/gpt-oss-20b), different from the model that answers and verifies.

The QA run saves every question's raw result to results/qa_raw.jsonl as it goes,
so if the free tier rate-limits you or you stop the script, running the same
command again resumes where it stopped.

CSV format for --questions (UTF-8):
    question,answer,answerable,page,file,split
      question   the question text
      answer     the correct answer, written from your document (leave empty if not answerable)
      answerable yes / no  (no = the document does NOT contain the answer)
      page       OPTIONAL page/slide number where the answer is (used for retrieval hit rate)
      file       OPTIONAL file name for that page
      split      OPTIONAL dev / test  (thresholds are chosen on dev, reported on test)
"""

import argparse
import csv
import json
import math
import os
import time
from collections import Counter

from dotenv import load_dotenv

load_dotenv()  # before any module creates an LLM client

from llm_utils import extract_json, get_llm, judge_model_name, main_model_name, safe_invoke
from verifier import (
    DEFAULT_RETRIEVAL_THRESHOLD,
    DEFAULT_SUPPORT_THRESHOLD,
    decide,
    generate_verified_mcqs,
    is_refusal,
)

RETRIEVAL_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]
SUPPORT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]


# =========================================================================
# Pure metric logic (no LLM, easy to test)
# =========================================================================
def classify(answerable, refused, verdict):
    """One question's outcome label."""
    if answerable:
        if refused:
            return "false_refusal"
        return "correct" if verdict == "correct" else "incorrect"
    return "correct_refusal" if refused else "false_answer"


def _ratio(a, b):
    return a / b if b else float("nan")


def summarize(outcomes):
    c = Counter(outcomes)
    n = len(outcomes)
    answerable = c["correct"] + c["incorrect"] + c["false_refusal"]
    unanswerable = c["correct_refusal"] + c["false_answer"]
    answered = c["correct"] + c["incorrect"] + c["false_answer"]
    return {
        "n": n,
        "accuracy": _ratio(c["correct"] + c["correct_refusal"], n),
        "selective_accuracy": _ratio(c["correct"], answered),
        "coverage": _ratio(answered, n),
        "wrong_answer_rate": _ratio(c["incorrect"], answerable),
        "false_refusal_rate": _ratio(c["false_refusal"], answerable),
        "correct_refusal_rate": _ratio(c["correct_refusal"], unanswerable),
        "false_answer_rate": _ratio(c["false_answer"], unanswerable),
    }


def outcome_for(rec, system, retrieval_threshold=None, support_threshold=None):
    """Outcome label of one stored record for one system."""
    answerable = rec["answerable"]
    if system == "verified":
        s = rec["rag_hybrid"]
        abstain, _ = decide(
            rec["verification"]["strength"],
            rec["verification"]["supported_fraction"],
            s["text"],
            retrieval_threshold,
            support_threshold,
        )
        return classify(answerable, abstain, s["verdict"])
    s = rec[system]
    return classify(answerable, is_refusal(s["text"]), s["verdict"])


def sweep(records, grid_r=RETRIEVAL_GRID, grid_s=SUPPORT_GRID):
    """Summary for every threshold pair."""
    rows = []
    for rt in grid_r:
        for st in grid_s:
            outcomes = [outcome_for(r, "verified", rt, st) for r in records]
            rows.append({"retrieval_threshold": rt, "support_threshold": st, **summarize(outcomes)})
    return rows


def _nz(x):
    return 0.0 if (isinstance(x, float) and math.isnan(x)) else x


def choose_thresholds(dev_records):
    """Best pair on the dev rows: max accuracy, then lowest false-answer rate, then selective accuracy."""
    rows = sweep(dev_records)
    rows.sort(
        key=lambda r: (_nz(r["accuracy"]), -_nz(r["false_answer_rate"]), _nz(r["selective_accuracy"])),
        reverse=True,
    )
    best = rows[0]
    return best["retrieval_threshold"], best["support_threshold"]


def retrieval_hit_rate(records, system):
    hits = [r[system]["hit"] for r in records if r["answerable"] and r[system].get("hit") is not None]
    return _ratio(sum(hits), len(hits)), len(hits)


# =========================================================================
# Helpers
# =========================================================================
def pct(x):
    return "n/a" if (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def print_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def read_questions(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            q = (r.get("question") or "").strip()
            if not q:
                continue
            ans = (r.get("answerable") or "yes").strip().lower()
            page = (r.get("page") or "").strip()
            rows.append(
                {
                    "question": q,
                    "answer": (r.get("answer") or "").strip(),
                    "answerable": ans in ("yes", "y", "true", "1"),
                    "page": int(page) if page.isdigit() else None,
                    "file": (r.get("file") or "").strip(),
                    "split": (r.get("split") or "").strip().lower(),
                }
            )
    return rows


def judge_answer(judge, question, reference, system_answer):
    prompt = f"""You are grading a student-assistant answer against a reference answer.

Question: {question}
Reference answer (correct): {reference}
System answer: {system_answer}

The system answer may be longer or worded differently. It is "correct" only if it is
consistent with the reference answer and gives its key point. It is "incorrect" if it
contradicts the reference, misses the key point, or states something different.

Return ONLY JSON: {{"verdict": "correct"}} or {{"verdict": "incorrect"}}"""
    for _ in range(2):
        data = extract_json(safe_invoke(judge, prompt))
        if isinstance(data, dict) and data.get("verdict") in ("correct", "incorrect"):
            return data["verdict"]
    return "incorrect"  # unparseable twice; counted against the system (rare)


def page_hit(docs, row):
    if row["page"] is None:
        return None
    for d in docs:
        if d.metadata.get("page") == row["page"] and (
            not row["file"] or d.metadata.get("source") == row["file"]
        ):
            return True
    return False


# =========================================================================
# QA evaluation
# =========================================================================
def evaluate_one(chain, judge, row):
    q = row["question"]

    def verdict(text):
        if row["answerable"] and not is_refusal(text):
            return judge_answer(judge, q, row["answer"], text)
        return None

    llm_text = chain.ask_llm_only(q)
    vec_text, vec_docs = chain.ask_plain(q, mode="vector", use_history=False)
    hyb_text, hyb_docs = chain.ask_plain(q, mode="hybrid", use_history=False)
    strength, ver = chain.check_answer(q, hyb_text, hyb_docs)

    return {
        "question": q,
        "reference": row["answer"],
        "answerable": row["answerable"],
        "split": row["split"],
        "llm_only": {"text": llm_text, "verdict": verdict(llm_text)},
        "rag_vector": {"text": vec_text, "verdict": verdict(vec_text), "hit": page_hit(vec_docs, row)},
        "rag_hybrid": {"text": hyb_text, "verdict": verdict(hyb_text), "hit": page_hit(hyb_docs, row)},
        "verification": {
            "strength": strength,
            "supported_fraction": ver["supported_fraction"],
            "ok": ver["ok"],
        },
    }


def load_raw(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["question"]] = rec
    return done


def run_qa(args):
    from rag_chain import QAChain

    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "qa_raw.jsonl")
    rows = read_questions(args.questions)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No questions found in the CSV.")

    if args.fresh and os.path.exists(raw_path):
        os.remove(raw_path)
    done = load_raw(raw_path)

    print(f"Answering model: {main_model_name()} | judge model: {judge_model_name()}")
    chain = QAChain(history=[])
    if not chain.hybrid_available:
        print("WARNING: rank-bm25 is not installed, so 'hybrid' equals 'vector'. Run: pip install rank-bm25")
    judge = get_llm(judge_model_name(), temperature=0)

    for i, row in enumerate(rows, 1):
        if row["question"] in done:
            continue
        print(f"[{i}/{len(rows)}] {row['question'][:70]}")
        rec = evaluate_one(chain, judge, row)
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        done[row["question"]] = rec
        time.sleep(args.sleep)

    records = [done[r["question"]] for r in rows if r["question"] in done]
    report_qa(records, args.out)


def report_qa(records, out_dir):
    has_split = any(r["split"] in ("dev", "test") for r in records)
    if has_split:
        dev = [r for r in records if r["split"] == "dev"]
        test = [r for r in records if r["split"] == "test"]
        if not dev or not test:
            raise SystemExit("If you use the split column, include both dev and test rows.")
        rt, st = choose_thresholds(dev)
        eval_rows = test
        print(f"\nThresholds chosen on {len(dev)} dev questions: retrieval={rt}, support={st}")
        print(f"Results below are on {len(test)} test questions.\n")
    else:
        rt, st = choose_thresholds(records)
        eval_rows = records
        print(
            "\nNOTE: no dev/test split column, so thresholds were tuned on the same questions "
            "they are reported on. For the paper, split your questions into dev and test.\n"
        )

    systems = [
        ("LLM alone (no documents)", "llm_only", None),
        ("RAG, vector search (original CampusGPT)", "rag_vector", None),
        ("RAG, hybrid search", "rag_hybrid", None),
        (f"Verified RAG (hybrid, thresholds {rt}/{st})", "verified", (rt, st)),
        (f"Verified RAG (default thresholds {DEFAULT_RETRIEVAL_THRESHOLD}/{DEFAULT_SUPPORT_THRESHOLD})",
         "verified", (DEFAULT_RETRIEVAL_THRESHOLD, DEFAULT_SUPPORT_THRESHOLD)),
    ]

    headers = ["System", "Accuracy", "Selective acc.", "Coverage", "Wrong ans.", "False refusal",
               "Correct refusal", "False answer"]
    table, csv_rows = [], []
    for label, key, th in systems:
        outcomes = [outcome_for(r, key, *(th or (None, None))) for r in eval_rows]
        s = summarize(outcomes)
        table.append([label, pct(s["accuracy"]), pct(s["selective_accuracy"]), pct(s["coverage"]),
                      pct(s["wrong_answer_rate"]), pct(s["false_refusal_rate"]),
                      pct(s["correct_refusal_rate"]), pct(s["false_answer_rate"])])
        csv_rows.append({"system": label, **s})

    n_ans = sum(1 for r in eval_rows if r["answerable"])
    print(f"Questions: {len(eval_rows)} ({n_ans} answerable, {len(eval_rows) - n_ans} not answerable)\n")
    print_table(headers, table)
    print(
        "\nAccuracy = correct answers + correct refusals over all questions."
        "\nSelective acc. = correct answers over the answers actually given."
        "\nWrong ans. = wrong answers over answerable questions."
        "\nFalse answer = answered a question the document cannot answer."
    )

    hv, nv = retrieval_hit_rate(eval_rows, "rag_vector")
    hh, nh = retrieval_hit_rate(eval_rows, "rag_hybrid")
    if nv:
        print(f"\nRetrieval hit rate@4 (gold page among retrieved), {nv} questions with a page: "
              f"vector {pct(hv)} | hybrid {pct(hh)}")

    failures = sum(1 for r in records if r["answerable"] and not r["verification"]["ok"]
                   and not is_refusal(r["rag_hybrid"]["text"]))
    if failures:
        print(f"\n{failures} answer(s) could not be verified (verifier reply unparseable); "
              "they are never refused on that basis.")

    # save files
    with open(os.path.join(out_dir, "qa_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    sweep_rows = sweep(eval_rows)
    with open(os.path.join(out_dir, "qa_threshold_sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"\nSaved: {out_dir}/qa_summary.csv, qa_threshold_sweep.csv, qa_raw.jsonl")


# =========================================================================
# Quiz evaluation
# =========================================================================
def judge_mcq(judge, mcq, source_text):
    options = "\n".join(mcq.get("options", []))
    prompt = f"""You are reviewing a multiple-choice question written from a source text.
Judge it using ONLY the source text.

Source text:
{source_text}

Question: {mcq.get('question')}
Options:
{options}
Marked answer: {mcq.get('correct_answer')}
Explanation: {mcq.get('explanation')}

Return ONLY JSON:
{{"key_correct": true, "exactly_one_correct_option": true, "explanation_ok": true}}
key_correct: the marked answer is right according to the source text.
exactly_one_correct_option: no other option is also right according to the source text.
explanation_ok: every claim in the explanation appears in the source text."""
    data = extract_json(safe_invoke(judge, prompt))
    if not isinstance(data, dict) or not all(
        k in data for k in ("key_correct", "exactly_one_correct_option", "explanation_ok")
    ):
        return None  # unjudged
    return not (data["key_correct"] and data["exactly_one_correct_option"] and data["explanation_ok"])


def run_quiz(args):
    from loaders import load_file
    from quiz_generator import generate_mcqs

    path = args.file if os.path.exists(args.file) else os.path.join("data", args.file)
    docs = load_file(path)
    text = "\n\n".join(d.page_content for d in docs)
    windows = [text[i:i + 6000] for i in range(0, len(text), 6000) if len(text[i:i + 6000]) > 1500]
    windows = windows[: args.sections]
    if not windows:
        raise SystemExit("The document is too short to evaluate (need at least ~1500 characters).")

    judge = get_llm(judge_model_name(), temperature=0)
    groups = {"baseline": [], "verified_kept": [], "verified_rejected": []}
    requested = 0
    for wi, win in enumerate(windows, 1):
        print(f"Section {wi}/{len(windows)}: generating {args.n} MCQs without and with verification")
        base = generate_mcqs(win, args.n, "Medium")
        if isinstance(base, list):
            groups["baseline"] += [(m, win) for m in base]
        kept, rejected, _ = generate_verified_mcqs(win, args.n, "Medium")
        groups["verified_kept"] += [(m, win) for m in kept]
        groups["verified_rejected"] += [(m, win) for m in rejected]
        requested += args.n
        time.sleep(args.sleep)

    results, rows = {}, []
    for name, items in groups.items():
        flawed, judged = 0, 0
        for m, win in items:
            verdict = judge_mcq(judge, m, win)
            time.sleep(args.sleep)
            if verdict is None:
                continue
            judged += 1
            flawed += int(verdict)
            rows.append({"group": name, "question": m.get("question"), "flawed": int(verdict)})
        results[name] = (flawed, judged, len(items))

    print("\nQuiz verification results (judged by the independent judge model)\n")
    table = []
    for label, key in [("Without verification (all generated)", "baseline"),
                       ("With verification (kept)", "verified_kept"),
                       ("Rejected by verification", "verified_rejected")]:
        flawed, judged, total = results[key]
        table.append([label, total, judged, flawed, pct(_ratio(flawed, judged))])
    print_table(["Group", "MCQs", "Judged", "Flawed", "Flawed rate"], table)
    kept_n = results["verified_kept"][2]
    print(f"\nYield: {kept_n} verified MCQs kept out of {requested} requested ({pct(_ratio(kept_n, requested))}).")
    print("Flawed = wrong key, more than one correct option, or an explanation not in the source text.")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "quiz_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "question", "flawed"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")


# =========================================================================
def main():
    p = argparse.ArgumentParser(description="CampusGPT evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    qa = sub.add_parser("qa", help="evaluate question answering")
    qa.add_argument("--questions", required=True, help="CSV of test questions")
    qa.add_argument("--out", default="results")
    qa.add_argument("--limit", type=int, default=0, help="only the first N questions")
    qa.add_argument("--sleep", type=float, default=2.0, help="seconds to wait between questions")
    qa.add_argument("--fresh", action="store_true", help="ignore saved progress and start over")

    qz = sub.add_parser("quiz", help="evaluate MCQ verification")
    qz.add_argument("--file", required=True, help="file name inside data/ (or a full path)")
    qz.add_argument("--n", type=int, default=8, help="MCQs per section")
    qz.add_argument("--sections", type=int, default=3, help="number of 6000-character sections")
    qz.add_argument("--out", default="results")
    qz.add_argument("--sleep", type=float, default=1.0)

    args = p.parse_args()
    run_qa(args) if args.cmd == "qa" else run_quiz(args)


if __name__ == "__main__":
    main()
