import shutil, os

PERSIST_DIR = "chroma_db"

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print("✅ Old DB cleared. You can now run ingest.py with new embeddings.")
