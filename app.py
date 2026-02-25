"""
app.py — TranZact ERP Support Chatbot
Streamlit web UI  +  RAG (ChromaDB)  +  Claude API

Run:  streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "tranzact_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K = 5          # number of knowledge entries to retrieve
MAX_HISTORY = 10   # messages to keep in context

SYSTEM_PROMPT = """You are a friendly and knowledgeable support agent for TranZact ERP software.
Your job is to help users understand how to use TranZact and solve their problems.

Guidelines:
- Answer based on the KNOWLEDGE BASE CONTEXT provided below.
- If the answer is clearly not in the context, say so honestly and suggest they contact TranZact support (10 AM – 7 PM, Mon–Sat).
- Keep answers clear, concise, and step-by-step where applicable.
- Use bullet points or numbered steps for multi-step instructions.
- Never make up features or steps that are not in the context.

KNOWLEDGE BASE CONTEXT:
{context}
"""

# ── Resource loading (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


# ── RAG search ────────────────────────────────────────────────────────────────
def search_knowledge(query: str, model, collection, top_k: int = TOP_K) -> list[str]:
    embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return results["documents"][0]


# ── Claude response ───────────────────────────────────────────────────────────
def get_response(
    user_query: str,
    context_docs: list[str],
    history: list[dict],
    api_key: str,
) -> str:
    context = "\n\n---\n\n".join(context_docs)
    system = SYSTEM_PROMPT.format(context=context)

    client = anthropic.Anthropic(api_key=api_key)

    # Build messages: history + current user query
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": user_query})

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TranZact Support Bot",
    page_icon="💬",
    layout="centered",
)

# Sidebar — API key
with st.sidebar:
    st.image("https://letstranzact.com/wp-content/uploads/2022/05/TranZact-Logo.svg", width=180)
    st.markdown("## Settings")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "This bot is trained on TranZact's internal support knowledge base "
        "containing 4900+ Q&A entries. Powered by Claude AI."
    )
    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header
st.title("💬 TranZact ERP Support")
st.caption("Ask me anything about TranZact — invoices, inventory, production, Tally integration & more.")

# Guard: API key required
if not api_key:
    st.warning("Please enter your Anthropic API key in the sidebar to start.")
    st.stop()

# Load vector DB + embedding model
try:
    model, collection = load_resources()
except Exception as e:
    st.error(
        f"Knowledge base not found. Please run `python ingest.py` first.\n\nError: {e}"
    )
    st.stop()

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Welcome message on first load
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I'm your TranZact ERP support assistant.\n\n"
            "I can help you with:\n"
            "- 📄 Invoices, Orders, Challans\n"
            "- 📦 Inventory & Stock Management\n"
            "- 🏭 Production & BOM\n"
            "- 🔗 Tally Integration\n"
            "- 📊 Reports & more\n\n"
            "What would you like help with today?"
        )

# Chat input
if prompt := st.chat_input("e.g. How do I generate an e-invoice?"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search knowledge base
    with st.spinner("Searching knowledge base..."):
        context_docs = search_knowledge(prompt, model, collection)

    # Get Claude response
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            history = st.session_state.messages[-MAX_HISTORY:]
            try:
                answer = get_response(prompt, context_docs, history, api_key)
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.markdown(answer)

    # Save to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
