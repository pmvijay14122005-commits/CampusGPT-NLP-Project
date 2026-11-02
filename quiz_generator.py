"""
quiz_generator.py
AI question generator: MCQs and descriptive (2/5/10 mark) questions as JSON.
For verified MCQs (checked against the source text) see verifier.generate_verified_mcqs.
"""

from llm_utils import extract_json, get_llm, safe_invoke


def _llm():
    return get_llm(temperature=0.4)  # created on use, so .env is already loaded


def _as_list(data):
    """Accept a list, or a dict that wraps the list (e.g. {"questions": [...]})."""
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
        return None
    return data if isinstance(data, list) else None


def generate_mcqs(content, num_questions=5, difficulty="Medium"):
    prompt = f"""Generate {num_questions} multiple-choice questions at {difficulty} difficulty
from the study material below. Every question must be answerable from the material alone.

Return ONLY valid JSON (a list), no extra text, in this exact format:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_answer": "A",
    "explanation": "..."
  }}
]

Study Material:
{content}
"""
    response = safe_invoke(_llm(), prompt)
    items = _as_list(extract_json(response))
    if items is None:
        return {"error": "Could not parse quiz JSON", "raw": response}
    needed = ("question", "options", "correct_answer", "explanation")
    good = [m for m in items if isinstance(m, dict) and all(k in m for k in needed)]
    if not good:
        return {"error": "Quiz JSON had no valid questions", "raw": response}
    return good


def generate_descriptive_questions(content, marks=5, difficulty="Medium", num_questions=3):
    prompt = f"""Generate {num_questions} descriptive/theory questions worth {marks} marks each,
at {difficulty} difficulty, from the study material below.

Return ONLY valid JSON (a list), no extra text, in this format:
[
  {{
    "question": "...",
    "marks": {marks},
    "expected_answer_points": ["point 1", "point 2", "point 3"]
  }}
]

Study Material:
{content}
"""
    response = safe_invoke(_llm(), prompt)
    items = _as_list(extract_json(response))
    if items is None:
        return {"error": "Could not parse question JSON", "raw": response}
    needed = ("question", "marks", "expected_answer_points")
    good = [q for q in items if isinstance(q, dict) and all(k in q for k in needed)]
    if not good:
        return {"error": "Question JSON had no valid questions", "raw": response}
    return good
