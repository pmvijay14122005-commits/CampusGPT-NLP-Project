"""
ingest.py
Loads every file in data/, splits into chunks, embeds them,
and stores them in the ChromaDB vector store.

Every run does a full rebuild from what is currently in data/,
so deleted files do not linger as stale search results.

Run directly to (re)build the index:
    python ingest.py
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from loaders import load_directory

PERSIST_DIR = "db/"


def build_vectorstore(data_path="data/", persist_dir=PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clear the old index through Chroma itself instead of deleting files.
    # On Windows, db/chroma.sqlite3 stays locked while the chatbot has it
    # open, so shutil.rmtree() fails with WinError 32.
    if os.path.exists(persist_dir):
        try:
            old_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            old_db.delete_collection()
        except Exception as e:
            print(f"Could not clear old index (continuing): {e}")

    print("Loading documents...")
    docs = load_directory(data_path)
    print(f"Loaded {len(docs)} raw document sections.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    print(f"Indexed {len(chunks)} chunks into '{persist_dir}'.")
    return len(chunks)


if __name__ == "__main__":
    build_vectorstore()
