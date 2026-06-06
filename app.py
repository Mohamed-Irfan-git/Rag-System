from flask import Flask, request, jsonify

from rag.ingest import load_docs, split_docs
from rag.vector import create_vector_db
from rag.retriever import get_retriever
from rag.chain import create_chain

app = Flask(__name__)

docs = load_docs("data/files.pdf")
chunks = split_docs(docs)
db = create_vector_db(chunks)
retriever = get_retriever(db)
qa_chain = create_chain(retriever)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data["question"]

    result = qa_chain.invoke({"query": query})

    return jsonify({
        "answer": result["result"]
    })

if __name__ == "__main__":
    app.run(debug=True)