"""
notes_generator.py
Feature 3: AI Study Notes Generator.
Takes raw text from an uploaded document and generates:
- Short summary
- Detailed summary
- Bullet-point notes
- Key concepts
- Important definitions
- Important topics
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

NOTES_PROMPT = PromptTemplate(
    template="""You are CampusGPT's notes generator. Read the study material
below and produce a well-structured revision guide with these exact sections:

## Short Summary
(3-4 sentences)

## Detailed Summary
(1-2 paragraphs)

## Bullet-Point Notes
(concise bullets covering all major points)

## Key Concepts
(list each concept with a 1-line explanation)

## Important Definitions
(term: definition format)

## Important Topics
(list topics likely to be asked in exams)

Study Material:
{content}
""",
    input_variables=["content"],
)


def chunk_text(text, max_chars=6000):
    """Split long documents into manageable pieces for the LLM context window."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def generate_notes(full_text):
    """
    Generates notes. For long documents, summarizes in pieces first,
    then merges into one final structured note-set.
    """
    pieces = chunk_text(full_text)

    if len(pieces) == 1:
        prompt = NOTES_PROMPT.format(content=pieces[0])
        return llm.invoke(prompt).content

    # Map step: summarize each piece briefly
    partial_summaries = []
    for piece in pieces:
        partial_prompt = f"Summarize the key points from this section in bullet points:\n\n{piece}"
        partial_summaries.append(llm.invoke(partial_prompt).content)

    merged_content = "\n\n".join(partial_summaries)

    # Reduce step: build final structured notes from merged summaries
    final_prompt = NOTES_PROMPT.format(content=merged_content)
    return llm.invoke(final_prompt).content
