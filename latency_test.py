import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

PERSIST_DIR = "chroma_db"

# Load database
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=OllamaEmbeddings(model="mistral")
)

# Prompt template
SYSTEM_PROMPT = "Answer using ONLY the context. Use [n] citations."
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Q: {question}\n\nContext:\n{context}\n\nAnswer:")
])

# Question for testing
QUESTION = "What is discussed in section 2?"

# Function to run test
def run_test(k_value):
    retriever = db.as_retriever(search_kwargs={"k": k_value})
    t0 = time.perf_counter()
    docs = retriever.get_relevant_documents(QUESTION)
    ctx = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    llm = ChatOllama(model="mistral", temperature=0)
    answer = llm.invoke(prompt.format_prompt(question=QUESTION, context=ctx).to_messages()).content
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n--- k={k_value} ---")
    print(answer)
    print(f"(latency: {elapsed:.0f} ms, chunks retrieved: {len(docs)})")

if __name__ == "__main__":
    run_test(2)   # fewer chunks, faster
    run_test(6)   # more chunks, slower but more complete
