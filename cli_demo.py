import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

PERSIST_DIR = "chroma_db"

def get_db():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=OllamaEmbeddings(model="mistral")
    )

SYSTEM_PROMPT = """Answer strictly from the context. Use [n] citations."""
prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT),
     ("human", "Q: {question}\n\nContext:\n{context}\n\nAnswer with [n] citations.")]
)

def format_docs(docs):
    out, i = [], 1
    seen = {}
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen[key] = i
            i += 1
        n = seen[key]
        out.append(f"[{n}] {d.page_content.strip()}")
    return "\n\n".join(out)

if __name__ == "__main__":
    db = get_db()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model="mistral", temperature=0.2)

    while True:
        q = input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        t0 = time.perf_counter()
        docs = retriever.get_relevant_documents(q)
        ctx = format_docs(docs)
        chain = ({"question": RunnablePassthrough(), "context": RunnablePassthrough()}
                 | prompt | llm)
        ans = chain.invoke({"question": q, "context": ctx}).content
        elapsed = (time.perf_counter() - t0) * 1000
        print("\n--- Answer ---")
        print(ans)
        print(f"\n(latency: {elapsed:.0f} ms)")
