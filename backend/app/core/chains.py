from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


from app.core.logger import get_logger
from app.core.config import settings
from app.core.vectorstore import get_vectorstore
from app.core.memory import get_session_history

logger = get_logger(__name__)

def build_retriever(document_id: Optional[str] = None):
    """
    Builds a HYBRID retriever that combines two different search strategies:
 
    1. Semantic search (Chroma / embeddings) — finds chunks with similar
       MEANING to the query, even if the exact words differ.
    2. Keyword search (BM25) — finds chunks that contain the exact WORDS
       from the query, scored by term frequency. This is what semantic
       search alone misses — e.g. a query like "table of contents" was
       matching random chapter-heading chunks (semantically "similar")
       instead of the actual contents page, because BM25 wasn't in play to
       weight the page where those exact words appear most densely.
 
    EnsembleRetriever runs both searches and merges their results using
    Reciprocal Rank Fusion (RRF) — chunks that rank highly in BOTH lists
    float to the top of the combined result.
 
    If document_id is given, both retrievers are scoped to only that
    document's chunks
    """
    vectorstore = get_vectorstore()

    # --- Fetch the chunk pool BM25 will search over ---
    # BM25Retriever works entirely in-memory over a fixed set of documents
    # (unlike Chroma, it has no built-in metadata filtering), so we pull
    # the relevant chunks out of Chroma first — either all of them (global
    # search) or just one document's (scoped search) — and hand them to
    # BM25Retriever directly.
    

    where_filter = {"document_id":document_id} if document_id else None

    raw = vectorstore.get(include=["documents", "metadatas"], where=where_filter)

    all_chunks = [
        Document(page_content=text, metadata = meta)
        for text, meta in zip(raw["documents"],raw["metadatas"])
    ]

    if not all_chunks:
        logger.warning(
            f"No Chunks found for BM25 index (document_id={document_id}) - "
            "falling back to semantic-only search."
        )
        search_kwargs = {"k":settings.retrieval_k}
        if document_id:
            search_kwargs['filter'] = {"document_id":document_id}
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = settings.retrieval_k

    search_kwargs = {"k":settings.retrieval_k}
    if document_id:
        search_kwargs["filter"] = {"document_id":document_id}
    semantic_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    hybrid_retriever = EnsembleRetriever(
        retrievers = [semantic_retriever,bm25_retriever],
        weights = [0.5,0.5]
    )
    return hybrid_retriever


def build_conversational_chain(document_id: Optional[str] = None):
    """
    Builds the full conversational RAG chain:

    1. History-aware retriever: rewrites follow-up questions into
       standalone questions (using chat history), THEN retrieves relevant
       chunks for that rewritten question.
    2. Question-answer chain: takes the retrieved chunks + question and
       generates an answer, grounded only in that context.
    3. Wrapped with RunnableWithMessageHistory so chat history is
       automatically loaded and updated per session_id — we never have to
       manually pass history in.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        # NOTE: gemini-3.6-flash deprecates the `temperature` sampling
        # parameter — the API silently ignores it now. Determinism is
        # instead controlled via explicit instructions in the system
        # prompt below ("be precise and consistent"), which is Google's
        # current recommended approach for this model generation.
    )
    retriever = build_retriever(document_id)

    # --- Step 1: Query rewriting ("history-aware retriever") ---
    # Given the chat history + the latest question, the LLM rewrites the
    # question to be standalone (e.g. "what's its solution?" -> "what is
    # the solution to overfitting?"), THEN that rewritten question is used
    # to retrieve chunks. If the question is already standalone, it's
    # passed through unchanged.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might "
         "reference context in the chat history, formulate a standalone "
         "question which can be understood without the chat history. "
         "Do NOT answer the question, just reformulate it if needed and "
         "otherwise return it as is. Be precise and consistent — given "
         "the same input, always produce the same rewritten question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # --- Step 2: Answer generation ---
    # Same grounding rules as our Simple RAG prompt: context-only answers,
    # moderate length, no hallucination.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful study assistant. Answer the question using "
         "ONLY the context below.\n\n"
         "Guidelines:\n"
         "- Give a well-rounded answer in about 4-6 sentences\n"
         "- If the answer is not present in the context, say clearly that "
         "it is not available in the notes — do not make up an answer\n"
         "- Respond in the same language style as the question (Hinglish "
         "question -> Hinglish answer, English question -> English answer)\n"
         "- Be precise and consistent: for the same question and context, "
         "always give the same core answer — avoid unnecessary variation "
         "in wording or emphasis between repeated runs\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combines the history-aware retriever + answer chain into one pipeline:
    # rewrite question -> retrieve -> answer.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # --- Step 3: Wrap with automatic memory management ---
    # This makes the chain automatically look up (and update) chat history
    # for whatever session_id is passed in at call time — we never touch
    # history manually.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain