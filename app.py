from flask import Flask, request, jsonify
from flask_cors import CORS

from rag.vector import load_vector_db, get_embeddings
from rag.retriever import get_retriever
from rag.chain import create_chain

app = Flask(__name__)
CORS(app)

qa_chain = None


# -------------------------
# INIT RAG SYSTEM
# -------------------------
try:
    embeddings = get_embeddings()
    db = load_vector_db(embeddings)

    if db is None:
        raise ValueError("Vector DB not found. Run ingestion first.")

    retriever = get_retriever(db)
    qa_chain = create_chain(retriever)

    print("✅ RAG system initialized successfully")

except Exception as e:
    print(f"❌ Initialization error: {e}")
    qa_chain = None


# -------------------------
# HEALTH CHECK
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "RAG API running"
    })


# -------------------------
# ASK ENDPOINT
# -------------------------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        if qa_chain is None:
            return jsonify({
                "error": "RAG system not initialized"
            }), 500

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON body"}), 400

        query = data.get("question", "").strip()

        if not query:
            return jsonify({"error": "question is required"}), 400

        # -------------------------
        # RAG CALL
        # -------------------------
        result = qa_chain.invoke({
            "input": query
        })

        answer = (
            result.get("answer")
            or result.get("result")
            or str(result)
        )

        return jsonify({
            "question": query,
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )