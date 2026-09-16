from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.config import settings
from app.core.vectorstore import get_vectorstore
from app.core.memory import get_session_history


def build_retriever(document_id: Optional[str] = None):
    """
    Builds a retriever over the vectorstore

    If document_id is given, retrieval is SCOPED to only that document's
    chunks (using Chroma's metadata filter). If document_id is None,
    retrieval searches across ALL uploaded documnets (global search).

    This is "Hybrid" retrieval behavior: global by default, scoped on request.
    """

    vectorstore = get_vectorstore()
    search_kwargs = {"k" : settings.retrieval_k}
    if document_id:
        # Chroma's metadata filter  - only chunks whose metadata.document_id
        # matches this value will be considered during search.
        search_kwargs["filter"] = {"document_id":document_id}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def build_conversational_chain(document_id: Optional[str] = None):
    """
    Builds the full conversational RAG Chain:

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
        model = settings.llm_model,
        google_api_key = settings.google_api_key,
        temperature = 0
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
         "Given a chat history and the latest user question which might"
         "reference context in the chat history, formulate a standalone"
         "question which can be understood without the chat history."
         "Do NOT answer the question, just reformulate it if needed and"
         "otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),
        
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm,retriever,contextualize_prompt
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
         "question -> Hinglish answer, English question -> English answer)\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

    #combines the history-aware retriever + answer chain into one pipeline
    # rewrite question -> retrieve ->answer

    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

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


