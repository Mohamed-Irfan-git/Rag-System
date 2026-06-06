from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

PERSIST_DIR = "vector_db"


# -------------------------
# EMBEDDINGS (UPDATED)
# -------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------------
# CREATE VECTOR DB
# -------------------------
def create_vector_db(chunks):
    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    return db


# -------------------------
# LOAD VECTOR DB
# -------------------------
def load_vector_db():
    if not os.path.exists(PERSIST_DIR):
        return None

    embeddings = get_embeddings()

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )