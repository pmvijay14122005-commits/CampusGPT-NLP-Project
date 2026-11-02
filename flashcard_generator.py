"""
flashcard_generator.py
Feature 5: AI Flashcard Generator.
Produces front/back flashcards as JSON for use in the Streamlit UI
(previous / next / random / mark-as-learned handled in app.py).
"""

import json
import re
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


def _extract_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else text


def generate_flashcards(content, num_cards=10):
    prompt = f"""Generate {num_cards} flashcards from the study material below.
Each flashcard should have a short "front" (a term or question) and a
concise "back" (the explanation/answer).

Return ONLY valid JSON (a list), no extra text, in this format:
[
  {{"front": "What is TF-IDF?", "back": "..."}}
]

Study Material:
{content}
"""
    response = llm.invoke(prompt).content
    try:
        cards = json.loads(_extract_json(response))
        for card in cards:
            card["learned"] = False
        return cards
    except json.JSONDecodeError:
        return {"error": "Could not parse flashcard JSON", "raw": response}

