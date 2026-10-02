import streamlit as st
import requests
import uuid
import json
import os
import logging
from dotenv import load_dotenv


load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- Simple logging setup ---
# This is intentionally lightweight compared to the backend's logger —
# a Streamlit app is a single-user local script, not a multi-user server,
# so there's no need for request IDs or daily rotation. Just a plain file
# log, useful for debugging issues after the fact.
logging.basicConfig(
    filename="frontend.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar: Upload + Document Management ---
with st.sidebar:
    st.header("📄 Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner("Uploading and embedding... this may take a while for large PDFs"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload/", files=files, timeout=600)
                    if response.status_code == 200:
                        st.success(f"Uploaded! {response.json()['chunks_created']} chunks created.")
                    else:
                        error_detail = response.json().get("error", "Unknown error")
                        st.error(f"Upload failed: {error_detail}")
                        logger.warning(f"Upload failed for '{uploaded_file.name}': {error_detail}")

                # Catching specific exception types gives the user a much
                # clearer message than one generic "something went wrong"
                except requests.exceptions.Timeout:
                    st.error("Upload timed out. The file may be too large, or the server is busy.")
                    logger.error(f"Upload timeout for '{uploaded_file.name}'")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend. Is it running?")
                    logger.error("Backend connection failed during upload")
                except requests.exceptions.RequestException as e:
                    st.error(f"Unexpected error during upload: {e}")
                    logger.exception("Unexpected error during upload")

    st.divider()

    try:
        docs_response = requests.get(f"{BACKEND_URL}/documents/", timeout=10)
        documents = docs_response.json().get("documents", [])
    except requests.exceptions.ConnectionError:
        documents = []
        st.warning("Backend not reachable — is it running?")
        logger.error("Backend connection failed while listing documents")
    except requests.exceptions.RequestException as e:
        documents = []
        st.warning(f"Could not load documents: {e}")
        logger.exception("Unexpected error while listing documents")

    if documents:
        st.caption(f"{len(documents)} document(s) uploaded")
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"{doc['filename']} ({doc['chunk_count']} chunks)")
            with col2:
                if st.button("🗑️", key=f"del_{doc['document_id']}"):
                    try:
                        requests.delete(f"{BACKEND_URL}/documents/{doc['document_id']}", timeout=30)
                    except requests.exceptions.RequestException:
                        st.error("Failed to delete — backend not reachable.")
                        logger.exception(f"Failed to delete document_id={doc['document_id']}")
                    st.rerun()
    else:
        st.caption("No documents uploaded yet")

    st.divider()

    doc_options = {"All documents": None}
    doc_options.update({doc["filename"]: doc["document_id"] for doc in documents})
    selected_doc_name = st.selectbox("Search scope", options=list(doc_options.keys()))
    selected_document_id = doc_options[selected_doc_name]


# --- Streaming helper ---

def stream_chat_response(question: str, session_id: str, document_id):
    """
    Calls the backend's streaming endpoint and yields each token as it
    arrives, instead of waiting for the full response.
    """
    payload = {
        "question": question,
        "session_id": session_id,
        "document_id": document_id,
    }

    full_answer = ""
    sources = []

    try:
        response = requests.post(
            f"{BACKEND_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=120,
        )

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            raw = line[len("data: "):]
            if raw == "[DONE]":
                break

            event = json.loads(raw)

            if "token" in event:
                full_answer += event["token"]
                yield full_answer, sources

            if "sources" in event:
                sources = event["sources"]
                yield full_answer, sources

            if "error" in event:
                logger.warning(f"Backend reported a stream error: {event['error']}")
                yield f"⚠️ {event['error']}", sources
                break

    except requests.exceptions.Timeout:
        logger.error(f"Chat request timed out | question='{question}'")
        yield "⚠️ The request took too long and timed out. Please try again.", sources
    except requests.exceptions.ConnectionError:
        logger.error("Backend connection failed during chat")
        yield "⚠️ Could not connect to the backend. Is it running?", sources
    except requests.exceptions.RequestException as e:
        logger.exception("Unexpected error during chat")
        yield f"⚠️ Unexpected error: {e}", sources


# --- Main chat area ---

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📄 Sources: {', '.join(msg['sources'])}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_answer = ""
        final_sources = []

        for partial_answer, sources in stream_chat_response(
            question, st.session_state.session_id, selected_document_id
        ):
            final_answer = partial_answer
            final_sources = sources
            placeholder.markdown(final_answer + "▌")

        placeholder.markdown(final_answer)
        if final_sources:
            st.caption(f"📄 Sources: {', '.join(final_sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "sources": final_sources,
    })