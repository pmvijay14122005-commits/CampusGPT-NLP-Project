"""
rag_chain.py
CampusGPT question answering.

Retrieval
  - "vector": semantic search in ChromaDB (the original CampusGPT behaviour)
  - "hybrid": BM25 keyword search + vector search merged with Reciprocal Rank Fusion

Answering
  - ask_plain():      retrieve + answer, no checking (the baseline RAG)
  - ask_verified():   retrieve + answer + sentence-level evidence verification,
                      with a retrieval gate and an evidence gate that can refuse
  - ask_llm_only():   the LLM with no documents at all (baseline for evaluation)

Conversation history is passed in by the app (loaded from SQLite), so it
survives restarts.
"""

import re

import numpy as np

from llm_utils import get_llm, safe_invoke
from verifier import (
    DEFAULT_RETRIEVAL_THRESHOLD,
    DEFAULT_SUPPORT_THRESHOLD,
    NOT_FOUND_MSG,
    decide,
    is_refusal,
    retrieval_strength,
    verify_answer,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # hybrid search is optional; we fall back to vector-only
    BM25Okapi = None

PERSIST_DIR = "db/"

SYSTEM_PROMPT = """You are CampusGPT, an academic assistant for students.
Answer the question using ONLY the context below.
If the answer is not present in the context, reply exactly:
"Answer not found in uploaded material."

Context:
{context}
"""

LLM_ONLY_PROMPT = """You are CampusGPT, an academic assistant for students.
You have NO access to any uploaded documents. Answer the question briefly,
using your general knowledge.
"""


def _tokenize(text):
    return re.findall(r"\w+", (text or "").lower())


def reciprocal_rank_fusion(rank_lists, k=60):
    """Merge several ranked lists of keys. Returns keys, best first."""
    scores = {}
    for ranked in rank_lists:
        for rank, key in enumerate(ranked, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [key for key, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def format_sources(docs):
    seen = []
    for d in docs:
        s = f"{d.metadata.get('source')} (page {d.metadata.get('page')})"
        if s not in seen:
            seen.append(s)
    return seen


class QAChain:
    def __init__(self, persist_dir=PERSIST_DIR, k=4, history=None,
                 default_mode="hybrid", embeddings=None, vectordb=None, llm=None):
        # embeddings / vectordb / llm can be injected, which keeps the logic testable
        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if vectordb is None:
            from langchain_chroma import Chroma
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        self.embeddings = embeddings
        self.vectordb = vectordb
        self.llm = llm or get_llm(temperature=0)
        self.k = k
        self.default_mode = default_mode
        self.history = list(history or [])  # list of (question, answer)
        self._build_bm25()

    # ------------------------------------------------------------ retrieval
    def _build_bm25(self):
        from langchain_core.documents import Document

        self.bm25 = None
        self.corpus_docs = []
        if BM25Okapi is None:
            return
        data = self.vectordb.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        self.corpus_docs = [
            Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas)
        ]
        if self.corpus_docs:
            self.bm25 = BM25Okapi([_tokenize(d.page_content) for d in self.corpus_docs])

    @property
    def hybrid_available(self):
        return self.bm25 is not None

    def retrieve(self, question, mode=None, k=None):
        mode = mode or self.default_mode
        k = k or self.k
        if mode == "vector" or self.bm25 is None:
            return self.vectordb.similarity_search(question, k=k)

        pool = max(k * 3, 10)
        vec_docs = self.vectordb.similarity_search(question, k=pool)
        scores = self.bm25.get_scores(_tokenize(question))
        order = np.argsort(scores)[::-1][:pool]
        bm_docs = [self.corpus_docs[i] for i in order if scores[i] > 0]

        by_key = {}
        for d in vec_docs + bm_docs:
            by_key.setdefault(d.page_content, d)
        fused = reciprocal_rank_fusion(
            [[d.page_content for d in vec_docs], [d.page_content for d in bm_docs]]
        )
        return [by_key[key] for key in fused[:k]]

    # ----------------------------------------------------------- generation
    def generate(self, question, docs, use_history=True):
        context = "\n\n".join(d.page_content for d in docs)
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        if use_history and self.history:
            history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history[-5:])
            messages.append(
                {"role": "system", "content": f"Previous conversation:\n{history_text}"}
            )
        messages.append({"role": "user", "content": question})
        return safe_invoke(self.llm, messages).strip()

    def ask_plain(self, question, mode=None, use_history=True):
        """Baseline RAG: retrieve and answer, no verification. Returns (answer, docs)."""
        docs = self.retrieve(question, mode)
        return self.generate(question, docs, use_history), docs

    def ask_llm_only(self, question):
        """No documents at all. Baseline for the evaluation."""
        messages = [
            {"role": "system", "content": LLM_ONLY_PROMPT},
            {"role": "user", "content": question},
        ]
        return safe_invoke(self.llm, messages).strip()

    # --------------------------------------------------------- verification
    def check_answer(self, question, answer, docs):
        """Returns (retrieval_strength, verification dict) for an existing answer."""
        strength = retrieval_strength(self.embeddings, question, docs)
        if docs and not is_refusal(answer):
            ver = verify_answer(question, answer, docs, self.llm)
        else:
            ver = {"ok": False, "supported_fraction": None, "sentences": []}
        return strength, ver

    def ask_verified(self, question, mode=None,
                     retrieval_threshold=DEFAULT_RETRIEVAL_THRESHOLD,
                     support_threshold=DEFAULT_SUPPORT_THRESHOLD,
                     short_circuit=True, use_history=True, remember=True):
        """
        Retrieve, answer, then verify the answer against the retrieved chunks.
        short_circuit=True skips the LLM entirely when retrieval is weak.
        """
        question = (question or "").strip()
        mode = mode or self.default_mode
        if len(question) < 3:
            return self._result(question, "", NOT_FOUND_MSG, True, "question too short",
                                0.0, None, False, [], [], mode)

        docs = self.retrieve(question, mode)
        strength = retrieval_strength(self.embeddings, question, docs)

        if short_circuit and strength < retrieval_threshold:
            raw, ver = NOT_FOUND_MSG, {"ok": False, "supported_fraction": None, "sentences": []}
            abstain, reason = True, "weak retrieval"
        else:
            raw = self.generate(question, docs, use_history)
            _, ver = self.check_answer(question, raw, docs)
            abstain, reason = decide(strength, ver["supported_fraction"], raw,
                                     retrieval_threshold, support_threshold)

        final = NOT_FOUND_MSG if abstain else raw
        if remember:
            self.history.append((question, final))
        return self._result(question, raw, final, abstain, reason, strength,
                            ver["supported_fraction"], ver["ok"], ver["sentences"], docs, mode)

    @staticmethod
    def _result(question, raw, final, abstain, reason, strength, fraction, ver_ok,
                sentences, docs, mode):
        return {
            "question": question,
            "raw_answer": raw,
            "answer": final,
            "abstain": abstain,
            "reason": reason,
            "strength": strength,
            "confidence": fraction,
            "verification_ok": ver_ok,
            "sentences": sentences,
            "docs": docs,
            "sources": format_sources(docs),
            "mode": mode,
        }

    # ------------------------------------------------------ old interface
    def invoke(self, inputs):
        """Kept for backward compatibility: plain RAG with history."""
        question = inputs["question"]
        answer, docs = self.ask_plain(question, use_history=True)
        self.history.append((question, answer))
        return {"answer": answer, "source_documents": docs}


def get_qa_chain(persist_dir=PERSIST_DIR, history=None, k=4, default_mode="hybrid"):
    return QAChain(persist_dir=persist_dir, k=k, history=history, default_mode=default_mode)


def ask_question(chain, query):
    """Kept for backward compatibility. Returns (answer, sources)."""
    result = chain.invoke({"question": query})
    return result["answer"], format_sources(result.get("source_documents", []))