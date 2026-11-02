"""
verifier.py
The research contribution of CampusGPT: an evidence-verification layer.

1. retrieval_strength()   how close the best retrieved chunk is to the question
2. verify_answer()        checks every sentence of an answer against the retrieved
                          chunks and returns which sentences are supported
3. verify_mcq()           checks that a generated MCQ's correct answer and
                          explanation are supported by the source text, and that
                          no other option is also correct
4. generate_verified_mcqs()  generates MCQs, keeps only the verified ones

Nothing here is tied to Streamlit, so evaluate.py can reuse it.
"""

import re

import numpy as np

from llm_utils import extract_json, get_llm, safe_invoke

NOT_FOUND_MSG = "Answer not found in uploaded material."

# Default decision thresholds. The app lets you change them, and evaluate.py
# sweeps them so you can pick values from data instead of guessing.
DEFAULT_RETRIEVAL_THRESHOLD = 0.25   # cosine similarity of the best chunk
DEFAULT_SUPPORT_THRESHOLD = 0.70     # share of answer sentences that must be supported


def is_refusal(text):
    t = (text or "").lower()
    return "not found in uploaded material" in t or "answer not found" in t


# ---------------------------------------------------------------------------
# 1. Retrieval strength
# ---------------------------------------------------------------------------
def retrieval_strength(embeddings, question, docs):
    """Highest cosine similarity between the question and any retrieved chunk (0 to 1)."""
    if not docs:
        return 0.0
    q = np.array(embeddings.embed_query(question), dtype=float)
    d = np.array(embeddings.embed_documents([x.page_content for x in docs]), dtype=float)
    sims = (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q) + 1e-9)
    return float(max(0.0, min(1.0, sims.max())))


# ---------------------------------------------------------------------------
# 2. Sentence-level answer verification
# ---------------------------------------------------------------------------
_LIST_MARKER = re.compile(r"^\s*(?:[-*•#>]+|\d+[.)])\s*")


def split_sentences(text):
    """Split an answer into checkable sentences (handles bullet and numbered lists)."""
    sentences = []
    for line in (text or "").splitlines():
        line = _LIST_MARKER.sub("", line).replace("**", "").strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if len(part.split()) >= 3:  # skip tiny fragments such as "Deployment"
                sentences.append(part)
    if not sentences and (text or "").strip():
        sentences = [text.strip()]
    return sentences


def _passages_block(docs, max_chars=1200):
    parts = []
    for i, d in enumerate(docs, 1):
        meta = f"{d.metadata.get('source')} p.{d.metadata.get('page')}"
        parts.append(f"[Passage {i}] ({meta})\n{d.page_content[:max_chars]}")
    return "\n\n".join(parts)


def verify_answer(question, answer, docs, llm=None):
    """
    Check every sentence of `answer` against the retrieved chunks `docs`.
    Returns {"ok": bool, "supported_fraction": float|None, "sentences": [...]}.
    ok=False means the verifier reply could not be parsed (treat as unverified).
    """
    llm = llm or get_llm()
    sentences = split_sentences(answer)
    if not sentences or not docs:
        return {"ok": False, "supported_fraction": None, "sentences": []}

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    prompt = f"""You are a strict fact checker. Use ONLY the source passages below.
For each numbered sentence, decide if the passages state it or directly imply it.
- Mark supported=true only if the passages back up the sentence's factual content.
- Sentences that make no factual claim (greetings, transitions) count as supported.
- Do NOT use outside knowledge, even if the sentence is true in the real world.

Question: {question}

Source passages:
{_passages_block(docs)}

Sentences to check:
{numbered}

Return ONLY a JSON list, one item per sentence, in this exact format:
[{{"id": 1, "supported": true, "evidence": 2}}]
"evidence" is the passage number that backs the sentence, or null if unsupported."""

    reply = safe_invoke(llm, prompt)
    data = extract_json(reply)
    if not isinstance(data, list):
        return {"ok": False, "supported_fraction": None, "sentences": []}

    by_id = {}
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            by_id[item["id"]] = item

    results = []
    for i, s in enumerate(sentences, 1):
        item = by_id.get(i, {})
        supported = bool(item.get("supported", False))
        ev = item.get("evidence")
        snippet, source = "", ""
        if supported and isinstance(ev, int) and 1 <= ev <= len(docs):
            snippet = docs[ev - 1].page_content.strip().replace("\n", " ")[:300]
            source = f"{docs[ev - 1].metadata.get('source')} (page {docs[ev - 1].metadata.get('page')})"
        results.append({"text": s, "supported": supported, "evidence": snippet, "source": source})

    fraction = sum(r["supported"] for r in results) / len(results)
    return {"ok": True, "supported_fraction": fraction, "sentences": results}


def decide(strength, supported_fraction, answer_text,
           retrieval_threshold=DEFAULT_RETRIEVAL_THRESHOLD,
           support_threshold=DEFAULT_SUPPORT_THRESHOLD):
    """
    The abstention rule. Returns (abstain: bool, reason: str).
    supported_fraction is None when the verifier could not run; we then do not
    abstain on that basis (the answer is shown but marked unverified).
    """
    if strength < retrieval_threshold:
        return True, "weak retrieval"
    if is_refusal(answer_text):
        return True, "model refused"
    if supported_fraction is not None and supported_fraction < support_threshold:
        return True, "low evidence support"
    return False, ""


# ---------------------------------------------------------------------------
# 3. MCQ verification
# ---------------------------------------------------------------------------
def verify_mcq(mcq, source_text, llm=None):
    """
    Is this MCQ grounded in `source_text`?
    verified = key is supported AND explanation is supported AND no other option is also correct.
    """
    llm = llm or get_llm()
    options = "\n".join(mcq.get("options", []))
    prompt = f"""You are checking a multiple-choice question against its source text.
Use ONLY the source text. Do not use outside knowledge.

Source text:
{source_text}

Question: {mcq.get('question')}
Options:
{options}
Marked correct answer: {mcq.get('correct_answer')}
Explanation given: {mcq.get('explanation')}

Answer these checks:
1. key_supported: does the source text support the marked correct answer?
2. other_correct: letters of any OTHER options that the source text would also make correct (empty list if none).
3. explanation_supported: is every claim in the explanation supported by the source text?

Return ONLY JSON in this format:
{{"key_supported": true, "other_correct": [], "explanation_supported": true, "reason": "one short sentence"}}"""

    data = extract_json(safe_invoke(llm, prompt))
    if not isinstance(data, dict):
        return {"ok": False, "verified": False, "reason": "verifier reply could not be parsed"}

    key_ok = bool(data.get("key_supported", False))
    others = data.get("other_correct") or []
    expl_ok = bool(data.get("explanation_supported", False))
    return {
        "ok": True,
        "verified": key_ok and not others and expl_ok,
        "key_supported": key_ok,
        "other_correct": others,
        "explanation_supported": expl_ok,
        "reason": str(data.get("reason", "")),
    }


def generate_verified_mcqs(content, num_questions=5, difficulty="Medium", max_rounds=2):
    """
    Generate MCQs and keep only verified ones. Over-generates a little so that
    after rejections we can still return `num_questions` when possible.
    Returns (kept_mcqs, rejected_mcqs, stats).
    """
    from quiz_generator import generate_mcqs  # imported here to avoid a circular import

    llm = get_llm()
    kept, rejected = [], []
    generated = 0
    for round_no in range(max_rounds):
        need = num_questions - len(kept)
        if need <= 0:
            break
        batch = generate_mcqs(content, need + 2, difficulty)
        if isinstance(batch, dict):  # parsing error from the generator
            break
        generated += len(batch)
        for mcq in batch:
            if len(kept) >= num_questions:
                break
            try:
                check = verify_mcq(mcq, content, llm)
            except Exception as e:  # noqa: BLE001
                check = {"ok": False, "verified": False, "reason": f"verification error: {e}"}
            mcq["verification"] = check
            (kept if check["verified"] else rejected).append(mcq)

    stats = {
        "requested": num_questions,
        "generated": generated,
        "kept": len(kept),
        "rejected": len(rejected),
    }
    return kept, rejected, stats
