from rag.ingest import load_and_split
from rag.vector import create_vector_db

chunks = load_and_split("data/files.pdf")

print(f"Chunks created: {len(chunks)}")

db = create_vector_db(chunks)

print("Vector DB created successfully!")