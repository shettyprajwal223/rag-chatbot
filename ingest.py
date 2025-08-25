import glob, os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_docs():
    docs = []
    for f in glob.glob("docs/*.pdf"):
        docs.extend(PyPDFLoader(f).load())
    for f in glob.glob("docs/*.txt"):
        docs.extend(TextLoader(f, encoding="utf-8").load())
    return docs

def main():
    print("📂 Loading documents...")
    docs = load_docs()
    if not docs:
        print("⚠️ No documents found in docs/ folder.")
        return

    print(f"✅ Loaded {len(docs)} docs. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    print("🔍 Generating embeddings (using HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    print(f"✅ Saved {len(chunks)} chunks into {PERSIST_DIR}")

if __name__ == "__main__":
    main()
