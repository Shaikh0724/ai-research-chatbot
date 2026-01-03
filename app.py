import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load API Key from .env
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="centered")

# --- Custom Styling (Clean & Professional) ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-title { color: #1e3a8a; font-size: 38px; font-weight: bold; text-align: center; margin-bottom: 5px; }
    .sub-title { color: #4b5563; text-align: center; margin-bottom: 30px; font-style: italic; font-size: 18px; }
    .stChatMessage { border-radius: 15px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown("<div class='main-title'>🔬 AI Research Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Expert insights on AI Tools, Mechanisms, and Ethics</div>", unsafe_allow_html=True)

# --- Load Vector Database ---
@st.cache_resource
def load_db():
    if os.path.exists("./chroma_db"):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return None

db = load_db()

if db is None:
    st.error("⚠️ Database (chroma_db) not found! Please run 'python ingestion.py' first.")
    st.stop()

# --- RAG Chain Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# MMR Search optimized for diversity across 10 papers
retriever = db.as_retriever(
    search_type="mmr", 
    search_kwargs={'k': 7, 'fetch_k': 20}
)

# Professional Prompt Template
template = """You are a professional AI research assistant powered by GPT-4o-mini using RAG (Retrieval-Augmented Generation). 
Use the following academic context to answer the user's question accurately.
If the information is not in the context, say you don't know. 
Focus on explaining AI tools, their mechanisms, and their impacts.

Context: {context}

Question: {question}

Expert Answer:"""


prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    # Citation ke liye source metadata use karte hain
    return "\n\n".join(f"--- Source: {doc.metadata.get('source')} ---\n{doc.page_content}" for doc in docs)

# LCEL Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Settings
with st.sidebar:
    st.title("⚙️ Control Panel")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.info("Status: Connected to AI Knowledge Base (10 Papers)")

# Display previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input & AI Response ---
if query := st.chat_input("Ask me about AI, its tools, or its future..."):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing research papers..."):
            try:
                # Generate Answer
                response = rag_chain.invoke(query)
                st.markdown(response)
                
                # Retrieve sources for citation expander
                source_docs = retriever.invoke(query)
                with st.expander("📚 Evidence & Citations"):
                    # Unique sources list
                    sources = {f"{doc.metadata.get('source')} (Page {doc.metadata.get('page', 'N/A')})" for doc in source_docs}
                    for s in sources:
                        st.write(f"📍 {s}")

                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")


                