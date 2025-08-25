import streamlit as st
import os, tempfile, shutil
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ------------------------
# Config
# ------------------------
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------------------------
# Helpers
# ------------------------
def reset_db():
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
        except PermissionError:
            st.warning("⚠️ Close any running processes using the DB and try again.")
        st.session_state.chat_history = []
        st.session_state.last_docs = []
        st.success("✅ Chroma DB cleared. You can now upload new documents.")
    else:
        st.info("ℹ️ No database found. You can start uploading documents.")

def load_and_split(path):
    if path.lower().endswith(".pdf"):
        docs = PyPDFLoader(path).load()
    elif path.lower().endswith(".txt"):
        docs = TextLoader(path, encoding="utf-8").load()
    else:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def add_to_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    db.add_documents(chunks)
    db.persist()

def get_answer(question):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k":5})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY from the context. Use citations like [1], [2]."),
        ("human", f"Q: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])
    llm = ChatOllama(model="mistral", temperature=0)
    answer = llm.invoke(prompt.format_prompt(question=question, context=context).to_messages()).content
    return answer, docs

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("📄 Local AI Q&A Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# Reset DB button
if st.button("🗑 Reset Database"):
    reset_db()

# Input + Upload row
st.markdown("---")
cols = st.columns([0.8, 0.2])

with cols[0]:
    question = st.text_input("", placeholder="Type your question here...", key="qinput")

with cols[1]:
    uploaded = st.file_uploader(
        "", type=["pdf","txt"], label_visibility="collapsed", key="upload_btn"
    )
    # Customize the button with CSS
    st.markdown(
        """
        <style>
        div[data-baseweb="file-uploader"] > div > label > div {
            background-color:black;
            color:white;
            border-radius:50%;
            width:40px;
            height:40px;
            display:flex;
            justify-content:center;
            align-items:center;
            font-size:24px;
        }
        div[data-baseweb="file-uploader"] input {
            display:none;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div style="position:absolute; top:-2px; left:0;">+</div>', unsafe_allow_html=True)

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        chunks = load_and_split(tmp_path)
        os.remove(tmp_path)
        if chunks:
            add_to_db(chunks)
            st.session_state.chat_history = []
            st.success(f"✅ Added {len(chunks)} chunks from {uploaded.name}")
        else:
            st.warning("⚠️ Could not extract content from the document.")

# Submit question when Enter is pressed
if question:
    try:
        answer, docs = get_answer(question.strip())
        st.session_state.chat_history.append(("You", question.strip()))
        st.session_state.chat_history.append(("Bot", answer))
        st.session_state.last_docs = docs
    except Exception as e:
        st.error(f"⚠️ Error: {e}")


# Display chat history
for speaker, msg in st.session_state.chat_history:
    color = "#141414" if speaker == "You" else "#141414"
    st.markdown(
        f"<div style='background-color:{color};padding:8px;border-radius:20px;margin:4px 0;'>"
        f"<b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True
    )

# Sources panel
if st.session_state.last_docs:
    st.markdown("### 📑 Sources")
    for i, d in enumerate(st.session_state.last_docs):
        with st.expander(f"Source [{i+1}]"):
            st.write(d.page_content[:600]+"...")
