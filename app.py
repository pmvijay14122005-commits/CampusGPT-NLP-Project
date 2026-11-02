"""
app.py
CampusGPT - main Streamlit application (6 tabs):
1. Upload & Management      4. Quiz / Question Generator
2. RAG Chatbot (verified)   5. Flashcards
3. Notes Generator          6. Dashboard

Run with:
    streamlit run app.py
"""

import os
import random

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # must run before the modules below create their LLM clients

from loaders import load_file
from ingest import build_vectorstore
from rag_chain import get_qa_chain, format_sources
from notes_generator import generate_notes
from quiz_generator import generate_mcqs, generate_descriptive_questions
from flashcard_generator import generate_flashcards
from verifier import (
    DEFAULT_RETRIEVAL_THRESHOLD,
    DEFAULT_SUPPORT_THRESHOLD,
    generate_verified_mcqs,
    is_refusal,
)
from theme import apply_theme, appearance_picker, hero, stat_card
from db_utils import (
    init_db, log_event, get_stats, get_recent_uploads,
    save_chat, load_chat, clear_chat,
)

init_db()

DATA_DIR = "data/"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="CampusGPT", page_icon="📚", layout="wide")
apply_theme()
hero("Welcome to CampusGPT",
     "Upload your study material. Get grounded answers, notes, quizzes and flashcards, checked against your own text.",
     chips=["🔎 Hybrid search", "✅ Verified answers", "📝 Smart quizzes", "🧠 Flashcards"])

# ---------------------------------------------------------------
# Sidebar: verification and retrieval settings
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Answer settings")
    verify_on = st.toggle("Evidence verification", value=True,
                          help="Check every answer sentence against the retrieved text and refuse weak answers.")
    mode = st.selectbox("Retrieval", ["hybrid", "vector"],
                        help="hybrid = keyword (BM25) + semantic search; vector = semantic only")
    retrieval_threshold = st.slider("Minimum retrieval strength", 0.0, 0.8,
                                    DEFAULT_RETRIEVAL_THRESHOLD, 0.01,
                                    help="Below this, the question is treated as not covered by your material.")
    support_threshold = st.slider("Minimum evidence support", 0.0, 1.0,
                                  DEFAULT_SUPPORT_THRESHOLD, 0.05,
                                  help="Share of answer sentences that must be backed by the material.")
    show_low = st.checkbox("Show low-evidence answers with a warning",
                           value=True,
                           help="When the evidence support is below the minimum, show the answer with a warning and the flagged sentences instead of refusing. Weak retrieval and 'not found' answers are still refused.")

with st.sidebar:
    appearance_picker()

tabs = st.tabs(
    ["📁 Upload", "💬 Chatbot", "📝 Notes Generator",
     "❓ Quiz Generator", "🧠 Flashcards", "📊 Dashboard"]
)


def confidence_badge(conf):
    pct = round(conf * 100)
    icon = "🟢" if conf >= 0.85 else "🟡" if conf >= 0.7 else "🔴"
    return f"{icon} Evidence support: {pct}%"


def render_entry(e):
    st.markdown(f"**You:** {e['question']}")
    if e["abstained"]:
        note = f" _(reason: {e['reason']})_" if e.get("reason") else ""
        st.warning(e["answer"] + note)
    else:
        st.markdown(f"**CampusGPT:** {e['answer']}")
        if e.get("confidence") is not None:
            st.caption(confidence_badge(e["confidence"]))
        elif verify_on:
            st.caption("⚪ Verification unavailable for this answer")
        if e.get("reason"):
            st.warning(e["reason"])
    if e.get("sentences"):
        with st.expander("Evidence check (sentence by sentence)"):
            for s in e["sentences"]:
                if s["supported"]:
                    st.markdown(f"✅ {s['text']}")
                    if s["evidence"]:
                        st.caption(f"“{s['evidence']}” — {s['source']}")
                else:
                    st.markdown(f"⚠️ {s['text']} _(not supported by the material)_")
    if e.get("sources"):
        with st.expander("Sources"):
            for s in e["sources"]:
                st.write("-", s)
    st.divider()


# ---------------------------------------------------------------
# TAB 1: Upload & Management
# ---------------------------------------------------------------
with tabs[0]:
    st.header("Study Material Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, PPTX, or DOCX files",
        type=["pdf", "pptx", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process & Index Files"):
        for f in uploaded_files:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            log_event("upload", f.name)
            st.success(f"Saved: {f.name}")

        with st.spinner("Extracting text and building index..."):
            num_chunks = build_vectorstore(data_path=DATA_DIR)
        st.success(f"Indexed {num_chunks} chunks. Ready for Q&A!")
        st.session_state.pop("qa_chain", None)  # reload with the new index

    st.subheader("Files currently uploaded")
    existing = os.listdir(DATA_DIR)
    if existing:
        for fname in existing:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f'<div class="cg-file">📄 {fname}</div>', unsafe_allow_html=True)
            if col2.button("🗑 Delete", key=f"del_{fname}"):
                os.remove(os.path.join(DATA_DIR, fname))
                st.session_state.pop("qa_chain", None)
                st.rerun()
        st.caption("Deleting a file removes it from disk. Click 'Process & Index Files' again afterward to rebuild the search index without it.")
    else:
        st.info("No files uploaded yet.")

# ---------------------------------------------------------------
# TAB 2: RAG Chatbot (verified)
# ---------------------------------------------------------------
with tabs[1]:
    st.header("Ask CampusGPT")

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = load_chat(50)  # history survives restarts

    if "qa_chain" not in st.session_state:
        if os.path.exists("db/"):
            pairs = [(c["question"], c["answer"]) for c in load_chat(5)]
            st.session_state.qa_chain = get_qa_chain(history=pairs, default_mode=mode)
        else:
            st.session_state.qa_chain = None

    chain = st.session_state.qa_chain
    if chain is None:
        st.warning("Upload and index at least one document first (see Upload tab).")
    else:
        if mode == "hybrid" and not chain.hybrid_available:
            st.info("Hybrid search needs the 'rank-bm25' package (pip install rank-bm25). Using vector search for now.")

        query = st.text_input("Your question:")
        col_ask, col_clear = st.columns([1, 6])
        if col_ask.button("Ask") and query:
            with st.spinner("Searching and verifying..." if verify_on else "Searching..."):
                if verify_on:
                    r = chain.ask_verified(
                        query, mode=mode,
                        retrieval_threshold=retrieval_threshold,
                        support_threshold=support_threshold,
                    )
                    answer_text, abstained, reason = r["answer"], r["abstain"], r["reason"]
                    if abstained and reason == "low evidence support" and show_low:
                        # show the answer, but make the weak evidence obvious
                        answer_text, abstained = r["raw_answer"], False
                        reason = ("Low evidence support: some sentences are not backed by your "
                                  "material. Check the ⚠️ ones in the evidence box below.")
                        if chain.history:
                            chain.history[-1] = (query, answer_text)
                    entry = {
                        "question": query, "answer": answer_text, "sources": r["sources"],
                        "confidence": r["confidence"], "abstained": abstained,
                        "reason": reason, "sentences": r["sentences"],
                    }
                else:
                    answer, docs = chain.ask_plain(query, mode=mode)
                    chain.history.append((query, answer))
                    entry = {
                        "question": query, "answer": answer, "sources": format_sources(docs),
                        "confidence": None, "abstained": is_refusal(answer),
                        "reason": "", "sentences": [],
                    }
            log_event("question", query)
            save_chat(entry["question"], entry["answer"], entry["sources"],
                      entry["confidence"], entry["abstained"], entry["reason"])
            st.session_state.chat_log.append(entry)

        if col_clear.button("Clear history"):
            clear_chat()
            st.session_state.chat_log = []
            chain.history = []
            st.rerun()

        for entry in reversed(st.session_state.chat_log):
            render_entry(entry)

# ---------------------------------------------------------------
# TAB 3: AI Study Notes Generator
# ---------------------------------------------------------------
with tabs[2]:
    st.header("Generate Study Notes")
    existing = os.listdir(DATA_DIR)
    if not existing:
        st.info("Upload a document first.")
    else:
        chosen_file = st.selectbox("Choose a document", existing, key="notes_file")
        if st.button("Generate Notes"):
            with st.spinner("Reading document and generating notes..."):
                docs = load_file(os.path.join(DATA_DIR, chosen_file))
                full_text = "\n\n".join(d.page_content for d in docs)
                notes = generate_notes(full_text)
            st.markdown(notes)
            st.download_button("Download Notes (.md)", notes, file_name="notes.md")

# ---------------------------------------------------------------
# TAB 4: Question & Quiz Generator
# ---------------------------------------------------------------
with tabs[3]:
    st.header("Generate Questions / Quiz")
    existing = os.listdir(DATA_DIR)
    if not existing:
        st.info("Upload a document first.")
    else:
        chosen_file = st.selectbox("Choose a document", existing, key="quiz_file")
        q_type = st.radio("Question type", ["MCQ", "Descriptive"])
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

        if q_type == "MCQ":
            num_q = st.slider("Number of MCQs", 1, 10, 5)
            verify_quiz = st.checkbox(
                "Verify each question against the source text (slower, rejects unsupported questions)",
                value=True,
            )
            if st.button("Generate MCQs"):
                docs = load_file(os.path.join(DATA_DIR, chosen_file))
                full_text = "\n\n".join(d.page_content for d in docs)[:6000]
                rejected, stats = [], None
                with st.spinner("Generating and verifying MCQs..." if verify_quiz else "Generating MCQs..."):
                    if verify_quiz:
                        mcqs, rejected, stats = generate_verified_mcqs(full_text, num_q, difficulty)
                    else:
                        mcqs = generate_mcqs(full_text, num_q, difficulty)
                log_event("quiz", f"{chosen_file} - {num_q} MCQs")
                if isinstance(mcqs, dict) and "error" in mcqs:
                    st.error(mcqs["error"])
                    st.text(mcqs["raw"])
                else:
                    if stats:
                        st.success(
                            f"Kept {stats['kept']} verified question(s); "
                            f"rejected {stats['rejected']} that the source text did not fully support."
                        )
                    if not mcqs:
                        st.warning("No question passed verification. Try again or use a longer section.")
                    for i, mcq in enumerate(mcqs, 1):
                        st.markdown(f"**Q{i}. {mcq['question']}**")
                        for opt in mcq["options"]:
                            st.write(opt)
                        with st.expander("Show Answer & Explanation"):
                            st.write("Correct Answer:", mcq["correct_answer"])
                            st.write("Explanation:", mcq["explanation"])
                            if mcq.get("verification", {}).get("verified"):
                                st.caption("✅ Verified against the source text")
                        st.divider()
                    if rejected:
                        with st.expander(f"Rejected questions ({len(rejected)})"):
                            for mcq in rejected:
                                st.markdown(f"**{mcq['question']}**")
                                st.caption(f"Why: {mcq.get('verification', {}).get('reason', 'not supported')}")
        else:
            marks = st.selectbox("Marks", [2, 5, 10])
            num_q = st.slider("Number of questions", 1, 10, 3)
            if st.button("Generate Descriptive Questions"):
                docs = load_file(os.path.join(DATA_DIR, chosen_file))
                full_text = "\n\n".join(d.page_content for d in docs)[:6000]
                with st.spinner("Generating questions..."):
                    questions = generate_descriptive_questions(full_text, marks, difficulty, num_q)
                log_event("quiz", f"{chosen_file} - {num_q} descriptive Qs")
                if isinstance(questions, dict) and "error" in questions:
                    st.error(questions["error"])
                    st.text(questions["raw"])
                else:
                    for i, q in enumerate(questions, 1):
                        st.markdown(f"**Q{i}. ({q['marks']} marks) {q['question']}**")
                        with st.expander("Expected answer points"):
                            for pt in q["expected_answer_points"]:
                                st.write("-", pt)
                        st.divider()

# ---------------------------------------------------------------
# TAB 5: Flashcard Generator
# ---------------------------------------------------------------
with tabs[4]:
    st.header("Flashcards")
    existing = os.listdir(DATA_DIR)
    if not existing:
        st.info("Upload a document first.")
    else:
        chosen_file = st.selectbox("Choose a document", existing, key="flash_file")
        num_cards = st.slider("Number of flashcards", 5, 20, 10)

        if st.button("Generate Flashcards"):
            docs = load_file(os.path.join(DATA_DIR, chosen_file))
            full_text = "\n\n".join(d.page_content for d in docs)[:6000]
            with st.spinner("Generating flashcards..."):
                cards = generate_flashcards(full_text, num_cards)
            if isinstance(cards, dict) and "error" in cards:
                st.error(cards["error"])
                st.text(cards["raw"])
            else:
                st.session_state.flashcards = cards
                st.session_state.card_index = 0
                log_event("flashcard", f"{chosen_file} - {num_cards} cards")

        if "flashcards" in st.session_state and st.session_state.flashcards:
            cards = st.session_state.flashcards
            idx = st.session_state.card_index
            card = cards[idx]

            st.markdown(f"### Card {idx + 1} / {len(cards)}")
            st.info(f"**Front:** {card['front']}")
            with st.expander("Show Back"):
                st.write(card["back"])

            col1, col2, col3, col4 = st.columns(4)
            if col1.button("⬅ Previous"):
                st.session_state.card_index = max(0, idx - 1)
                st.rerun()
            if col2.button("Next ➡"):
                st.session_state.card_index = min(len(cards) - 1, idx + 1)
                st.rerun()
            if col3.button("🎲 Random"):
                st.session_state.card_index = random.randint(0, len(cards) - 1)
                st.rerun()
            if col4.button("✅ Mark as Learned"):
                cards[idx]["learned"] = True
                st.success("Marked as learned!")

            learned_count = sum(1 for c in cards if c.get("learned"))
            st.progress(learned_count / len(cards))
            st.caption(f"{learned_count}/{len(cards)} cards learned")

# ---------------------------------------------------------------
# TAB 6: Dashboard
# ---------------------------------------------------------------
with tabs[5]:
    st.header("Student Learning Dashboard")
    stats = get_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card(stats["upload"], "Documents Uploaded", 1)
    with col2:
        stat_card(stats["question"], "Questions Asked", 2)
    with col3:
        stat_card(stats["quiz"], "Quizzes Generated", 3)
    with col4:
        stat_card(stats["flashcard"], "Flashcards Created", 4)

    st.write("")
    st.subheader("Activity overview")
    st.bar_chart(
        {"Count": {"Uploads": stats["upload"], "Questions": stats["question"],
                   "Quizzes": stats["quiz"], "Flashcards": stats["flashcard"]}},
        color="#6366F1",
    )

    st.subheader("Recent Uploads")
    recent = get_recent_uploads()
    if recent:
        for detail, ts in recent:
            st.write(f"- {detail}  _(uploaded {ts})_")
    else:
        st.info("No uploads yet.")