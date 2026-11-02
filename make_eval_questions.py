"""
make_eval_questions.py
Drafts a test set for evaluate.py from your documents, so you only have to REVIEW
questions instead of writing them from scratch.

    python make_eval_questions.py                       (all files in data/)
    python make_eval_questions.py --file Morphology.pdf --per-file 20 --unanswerable 5

It writes eval_questions_draft.csv with the columns evaluate.py expects:
    question, answer, answerable, page, file, split

IMPORTANT - this is a DRAFT written by an LLM:
  * Open the CSV and check every row against your document. Fix or delete bad rows.
  * The "answerable = no" questions are written to be NOT covered, but the model can be
    wrong; if the document does answer one, change it to yes and add the answer, or delete it.
  * Save the reviewed file as eval_questions.csv and run: python evaluate.py qa --questions eval_questions.csv
  * In the paper, say the questions were drafted with an LLM and then checked by hand.
"""

import argparse
import csv
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()

from llm_utils import extract_json, get_llm, safe_invoke
from loaders import load_file


def pick_pages(docs, count, min_chars=300):
    """Pick `count` pages spread evenly across the document, skipping near-empty ones."""
    good = [d for d in docs if len(d.page_content.strip()) >= min_chars]
    if len(good) <= count:
        return good
    step = len(good) / count
    return [good[int(i * step)] for i in range(count)]


def choose_pages(docs, count, min_chars=None):
    """
    Pick pages for answerable questions. Normally a page needs 300+ characters, but
    slide decks have short slides, so we lower the bar (300 -> 150 -> 80) until there
    are enough pages. Pass min_chars to force a value.
    """
    if min_chars is not None:
        return pick_pages(docs, count, min_chars)
    for threshold in (300, 150, 80):
        pages = pick_pages(docs, count, threshold)
        if len(pages) >= count:
            return pages
    return pick_pages(docs, count, 80)


def draft_answerable(llm, page_doc, file_name):
    prompt = f"""Read the text below and write ONE exam-style question that this text fully answers,
plus the correct answer written only from this text (1 to 2 sentences).
The question must make sense on its own: do not say "the text", "the passage" or "the slide".

Text:
{page_doc.page_content[:3500]}

Return ONLY JSON: {{"question": "...", "answer": "..."}}"""
    data = extract_json(safe_invoke(llm, prompt))
    if not isinstance(data, dict) or not data.get("question") or not data.get("answer"):
        return None
    return {
        "question": str(data["question"]).strip(),
        "answer": str(data["answer"]).strip(),
        "answerable": "yes",
        "page": page_doc.metadata.get("page", ""),
        "file": file_name,
    }


def draft_unanswerable(llm, docs, count):
    excerpts = []
    for d in pick_pages(docs, 8, min_chars=100):
        excerpts.append(d.page_content.strip()[:500])
    digest = "\n---\n".join(excerpts)
    prompt = f"""Below are excerpts from a course document.
Write {count} questions a student might ask about this SAME subject that the document does NOT answer
(for example related topics, finer details, examples or comparisons it never covers).
They must sound like real study questions, not silly or off-topic ones.

Excerpts:
{digest}

Return ONLY a JSON list of {count} question strings."""
    data = extract_json(safe_invoke(llm, prompt))
    if not isinstance(data, list):
        return []
    return [
        {"question": str(q).strip(), "answer": "", "answerable": "no", "page": "", "file": ""}
        for q in data[:count]
        if str(q).strip()
    ]


def assign_splits(rows, seed=42):
    """Alternate dev/test within each answerable group, in shuffled order."""
    rng = random.Random(seed)
    for flag in ("yes", "no"):
        group = [r for r in rows if r["answerable"] == flag]
        rng.shuffle(group)
        for i, r in enumerate(group):
            r["split"] = "dev" if i % 2 == 0 else "test"
    return rows


def main():
    p = argparse.ArgumentParser(description="Draft evaluation questions from your documents")
    p.add_argument("--file", help="one file name inside data/ (default: every file)")
    p.add_argument("--per-file", type=int, default=20, help="answerable questions per file")
    p.add_argument("--unanswerable", type=int, default=5, help="not-answerable questions per file")
    p.add_argument("--min-chars", type=int, default=None,
                   help="minimum characters a page needs (default: automatic, lower for slide decks)")
    p.add_argument("--out", default="eval_questions_draft.csv")
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args()

    files = [args.file] if args.file else sorted(os.listdir("data"))
    llm = get_llm(temperature=0.3)
    rows = []

    for fname in files:
        path = fname if os.path.exists(fname) else os.path.join("data", fname)
        if not os.path.isfile(path):
            print(f"Skipping {fname}: not found")
            continue
        try:
            docs = load_file(path)
        except ValueError:
            print(f"Skipping {fname}: unsupported type")
            continue
        base = os.path.basename(path)
        pages = choose_pages(docs, args.per_file, args.min_chars)
        print(f"{base}: drafting {len(pages)} answerable questions")
        for d in pages:
            try:
                row = draft_answerable(llm, d, base)
            except Exception as e:  # noqa: BLE001 - keep going if one call fails
                print(f"  page {d.metadata.get('page')}: skipped ({e})")
                continue
            if row:
                rows.append(row)
            time.sleep(args.sleep)
        if args.unanswerable:
            print(f"{base}: drafting {args.unanswerable} not-answerable questions")
            try:
                rows += draft_unanswerable(llm, docs, args.unanswerable)
            except Exception as e:  # noqa: BLE001
                print(f"  not-answerable questions skipped ({e})")

    if not rows:
        raise SystemExit("No questions were drafted. Check that data/ has files and GROQ_API_KEY is set.")

    rows = assign_splits(rows)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer", "answerable", "page", "file", "split"])
        w.writeheader()
        w.writerows(rows)

    yes = sum(1 for r in rows if r["answerable"] == "yes")
    print(f"\nWrote {len(rows)} draft rows ({yes} answerable, {len(rows) - yes} not answerable) to {args.out}")
    print("NEXT: open it in Excel/Notepad, check every row against your document, fix or delete bad ones,")
    print("save as eval_questions.csv (CSV UTF-8), then run: python evaluate.py qa --questions eval_questions.csv")


if __name__ == "__main__":
    main()