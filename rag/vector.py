from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings()

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    db.persist()
    return db