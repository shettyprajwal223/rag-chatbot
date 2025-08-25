Title & Description:
# 📄 Local RAG Chatbot (Ollama + LangChain + Chroma)
A local **Retrieval-Augmented Generation (RAG)** chatbot that works fully offline.  
Built with **Ollama (Mistral)**, **LangChain**, and **ChromaDB**.

Features:
- 🔹 Works 100% offline
- 🔹 Upload PDF/TXT files directly
- 🔹 Answers grounded in docs with citations [1], [2]
- 🔹 Vector DB with Chroma for fast retrieval
- 🔹 Adjustable retrieval size (`k`)
- 🔹 Interfaces:
  - Streamlit app (`app.py`)
  - CLI demo (`cli_demo.py`)

Architecture (Text Diagram):
Docs (PDF/TXT)


   ▼
   
Ingestion (split + embeddings with Mistral)
   
  
   ▼
   
Vector DB (ChromaDB)
   
   
   ▼
   
Retrieval (top-k chunks)
   
   
   ▼
   
Generation (Ollama + Mistral LLM)
   
   
   ▼
   
Answer + Citations

Setup & Installation:
1. Install Ollama: https://ollama.ai
   ollama pull mistral
   ollama pull llama3

2. Clone repo & create virtual environment
   git clone <your-repo-url>
   cd rag-bot
   python -m venv .venv
   .\.venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

Usage:
1. Add documents to docs/ folder
2. Ingest docs: python ingest.py
3. Run Streamlit app: streamlit run app.py
4. Optional CLI demo: python cli_demo.py
