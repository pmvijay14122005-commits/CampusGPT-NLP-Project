"""
stress.py
Stress test for the evidence verifier: plant KNOWN errors and measure how many
the verifier catches.

    python stress.py --questions eval_questions.csv --n 25
    python stress.py --questions eval_questions.csv --n 25 --file Morphology.pdf --mcq 8

Part A (answers): for each answerable question we take the correct reference answer and make
    clean         the reference answer as it is            (should NOT be flagged)
    injected      reference answer + one made-up sentence  (should be flagged)
    contradicted  the answer rewritten with one changed fact (should be flagged)
  The verifier checks each one against the real source page.

Part B (MCQs): for MCQs generated from your document we make
    clean         the MCQ as generated                      (should pass)
    wrong_key     the marked answer changed to another option (should be rejected)
    fake_expl     explanation + one claim not in the source  (should be rejected)

Detection rate = share of planted errors that were flagged.   False alarm rate = share of clean
items that were wrongly flagged.  The errors are written by the second (judge) model.

Output: results/stress_results.csv (every item, so you can spot-check by hand).
"""

import argparse
import csv
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()

from llm_utils import get_llm, judge_model_name, safe_invoke
from loaders import load_file
from verifier import DEFAULT_SUPPORT_THRESHOLD, verify_answer, verify_mcq


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ans = (r.get("answerable") or "").strip().lower()
            page = (r.get("page") or "").strip()
            if ans in ("yes", "y", "true", "1") and (r.get("answer") or "").strip() and page.isdigit() and r.get("file"):
                rows.append({"question": r["question"].strip(), "answer": r["answer"].strip(),
                             "page": page, "file": r["file"].strip()})
    return rows


def find_page(docs_cache, fname, page):
    if fname not in docs_cache:
        path = fname if os.path.exists(fname) else os.path.join("data", fname)
        docs_cache[fname] = load_file(path) if os.path.isfile(path) else []
    for d in docs_cache[fname]:
        if str(d.metadata.get("page")) == page:
            return d
    return None


def clean_text(s):
    return (s or "").strip().strip('"').strip()


def make_injected(writer, question, answer, page_text):
    prompt = f"""Below is a source text, a question and a correct answer.
Write ONE extra sentence that sounds like a natural part of the answer and is on the same topic,
but states a specific fact that is NOT in the source text and is not implied by it.
Return ONLY that sentence.

Source text:
{page_text[:2500]}

Question: {question}
Correct answer: {answer}"""
    extra = clean_text(safe_invoke(writer, prompt))
    return f"{answer} {extra}" if extra else None


def make_contradicted(writer, question, answer, page_text):
    prompt = f"""Rewrite the correct answer below so that it contains exactly one factual error:
change one key fact, term or number to a different, wrong one. Keep the same length and style.
The error must contradict the source text.
Return ONLY the rewritten answer.

Source text:
{page_text[:2500]}

Question: {question}
Correct answer: {answer}"""
    out = clean_text(safe_invoke(writer, prompt))
    return out if out and out != answer else None


def make_fake_expl(writer, mcq, source):
    prompt = f"""Add ONE sentence to the explanation below. The sentence must sound natural and relevant
but state a specific fact that is NOT in the source text.
Return ONLY the full new explanation.

Source text:
{source[:2500]}

Explanation: {mcq.get('explanation')}"""
    out = clean_text(safe_invoke(writer, prompt))
    return out if out and out != mcq.get("explanation") else None


def wrong_key(mcq, rng):
    letters = [o.strip()[0].upper() for o in mcq.get("options", []) if o.strip()]
    current = str(mcq.get("correct_answer", "")).strip()[:1].upper()
    others = [l for l in letters if l != current]
    return rng.choice(others) if others else None


def ratio(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def save(out_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stress_results.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["part", "type", "question", "text", "score", "flag_any", "flag_threshold"])
        w.writeheader()
        w.writerows(out_rows)
    return path


def part_a(args, writer, rng, out_rows):
    rows = read_rows(args.questions)
    if args.file:
        rows = [r for r in rows if r["file"] == args.file]
    rng.shuffle(rows)
    cache, used = {}, 0
    for r in rows:
        if used >= args.n:
            break
        d = find_page(cache, r["file"], r["page"])
        if d is None:
            continue
        used += 1
        print(f"Answers {used}/{args.n}")
        variants = {"clean": r["answer"]}
        try:
            inj = make_injected(writer, r["question"], r["answer"], d.page_content)
            con = make_contradicted(writer, r["question"], r["answer"], d.page_content)
        except Exception as e:  # noqa: BLE001
            print("  skipped (could not create errors):", e)
            continue
        if inj:
            variants["injected"] = inj
        if con:
            variants["contradicted"] = con
        for kind, text in variants.items():
            try:
                v = verify_answer(r["question"], text, [d])
            except Exception as e:  # noqa: BLE001
                print("  verify failed:", e)
                continue
            if not v["ok"]:
                continue
            frac = v["supported_fraction"]
            out_rows.append({"part": "answer", "type": kind, "question": r["question"], "text": text,
                             "score": round(frac, 3),
                             "flag_any": int(frac < 1.0),
                             "flag_threshold": int(frac < args.support)})
            save(out_rows, args.out)
            time.sleep(args.sleep)


def part_b(args, writer, rng, out_rows):
    from quiz_generator import generate_mcqs

    path = args.file if args.file and os.path.exists(args.file) else os.path.join("data", args.file or "")
    if not args.file or not os.path.isfile(path):
        print("Part B skipped: give --file NAME (a file inside data/) to test MCQs.")
        return
    text = "\n\n".join(d.page_content for d in load_file(path))
    windows = [text[i:i + 6000] for i in range(0, len(text), 6000) if len(text[i:i + 6000]) > 1500][:2]
    count = 0
    for win in windows:
        mcqs = generate_mcqs(win, args.mcq, "Medium")
        if isinstance(mcqs, dict):
            continue
        for m in mcqs:
            count += 1
            print(f"MCQ {count}")
            variants = {"clean": m}
            k = wrong_key(m, rng)
            if k:
                variants["wrong_key"] = dict(m, correct_answer=k)
            try:
                fe = make_fake_expl(writer, m, win)
            except Exception as e:  # noqa: BLE001
                fe = None
                print("  could not create fake explanation:", e)
            if fe:
                variants["fake_expl"] = dict(m, explanation=fe)
            for kind, mcq in variants.items():
                try:
                    v = verify_mcq(mcq, win)
                except Exception as e:  # noqa: BLE001
                    print("  verify failed:", e)
                    continue
                if not v["ok"]:
                    continue
                out_rows.append({"part": "mcq", "type": kind, "question": m.get("question"),
                                 "text": f"key={mcq.get('correct_answer')} | {mcq.get('explanation')}",
                                 "score": "", "flag_any": int(not v["verified"]),
                                 "flag_threshold": int(not v["verified"])})
                save(out_rows, args.out)
                time.sleep(args.sleep)


def report(out_rows, support):
    for part, title, order in [
        ("answer", "Part A - answers (verifier checks each sentence against the source page)",
         ["clean", "injected", "contradicted"]),
        ("mcq", "Part B - MCQs (verifier checks key, other options, explanation)",
         ["clean", "wrong_key", "fake_expl"]),
    ]:
        items = [r for r in out_rows if r["part"] == part]
        if not items:
            continue
        print(f"\n{title}\n")
        table_rows = []
        for kind in order:
            sub = [r for r in items if r["type"] == kind]
            if not sub:
                continue
            a = sum(r["flag_any"] for r in sub)
            t = sum(r["flag_threshold"] for r in sub)
            label = "False alarm rate (clean flagged)" if kind == "clean" else "Detection rate"
            if part == "answer":
                table_rows.append([kind, len(sub), label, ratio(a, len(sub)), ratio(t, len(sub))])
            else:
                table_rows.append([kind, len(sub), label, ratio(a, len(sub)), "-"])
        heads = ["Item type", "n", "Measure", "Any sentence unsupported" if part == "answer" else "Rejected",
                 f"Support < {support:.2f}" if part == "answer" else ""]
        table(heads, table_rows)


def main():
    p = argparse.ArgumentParser(description="Stress test for the CampusGPT verifier")
    p.add_argument("--questions", default="eval_questions.csv")
    p.add_argument("--n", type=int, default=25, help="answerable questions to use for Part A")
    p.add_argument("--file", default="", help="file in data/ (limits Part A to it, and is used for Part B)")
    p.add_argument("--mcq", type=int, default=6, help="MCQs per section for Part B (0 = skip)")
    p.add_argument("--support", type=float, default=DEFAULT_SUPPORT_THRESHOLD)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    writer = get_llm(judge_model_name(), temperature=0.7)  # a different model writes the errors
    out_rows = []
    try:
        part_a(args, writer, rng, out_rows)
        if args.mcq > 0:
            part_b(args, writer, rng, out_rows)
    except KeyboardInterrupt:
        print("\nStopped by you. Showing what was collected so far.")
    except Exception as e:  # noqa: BLE001 - e.g. the daily token limit; keep what we have
        print(f"\nStopped early: {str(e)[:200]}")
        print("Showing what was collected so far.")

    if not out_rows:
        raise SystemExit("Nothing was collected. If you saw a 429 / rate limit message, wait and run again.")
    report(out_rows, args.support)
    path = save(out_rows, args.out)
    print(f"\nSaved: {path}  (open it and spot-check a few planted errors by hand)")


if __name__ == "__main__":
    main()