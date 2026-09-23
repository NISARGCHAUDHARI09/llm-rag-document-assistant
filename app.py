from pathlib import Path
import hashlib
import sys
import tempfile

import streamlit as st


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TOP_K,
    VECTOR_DB_PATH,
)

from src.document_loader import load_pdf
from src.embeddings import get_embedding_model
from src.llm import get_llm
from src.rag_pipeline import generate_rag_response
from src.text_splitter import split_documents
from src.utils import calculate_file_hash
from src.vector_store import (
    add_documents_to_vector_store,
    get_indexed_documents,
    get_vector_store_count,
    load_vector_store,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .source-card {
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 0.5rem;
    }

    .metric-card {
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_hashes" not in st.session_state:
    st.session_state.indexed_hashes = set()


# ============================================================
# Cached resources
# ============================================================

@st.cache_resource
def load_embeddings():
    return get_embedding_model()


@st.cache_resource
def load_llm():
    return get_llm(
        model_name="openai/gpt-oss-20b",
        temperature=0.0,
    )


@st.cache_resource
def load_database(_embedding_model):
    return load_vector_store(
        _embedding_model
    )


# ============================================================
# Load core resources
# ============================================================

try:

    embedding_model = load_embeddings()

    llm = load_llm()

    vector_store = load_database(
        embedding_model
    )

except Exception as error:

    st.error(
        "Failed to initialize the RAG system."
    )

    st.exception(error)

    st.stop()


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">📚 RAG Document Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload documents, ask questions, and receive
    grounded answers with document and page citations.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Retrieval Settings")

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help=(
            "Number of chunks retrieved for each "
            "retrieval query."
        ),
    )

    absolute_threshold = st.slider(
        "Absolute Distance Threshold",
        min_value=0.50,
        max_value=1.20,
        value=1.00,
        step=0.05,
    )

    relative_margin = st.slider(
        "Strict Relative Margin",
        min_value=0.00,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )

    recovery_margin = st.slider(
        "Recovery Margin",
        min_value=0.05,
        max_value=0.20,
        value=0.10,
        step=0.01,
    )

    minimum_relevant_documents = st.slider(
        "Minimum Relevant Chunks",
        min_value=1,
        max_value=4,
        value=3,
        step=1,
    )

    redundancy_threshold = st.slider(
        "Redundancy Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.80,
        step=0.05,
    )

    max_context_documents = st.slider(
        "Maximum Context Chunks",
        min_value=1,
        max_value=8,
        value=min(4, top_k),
        step=1,
    )

    st.divider()

    st.header("📊 Knowledge Base")

    indexed_count = get_vector_store_count(
        vector_store
    )

    st.metric(
        "Indexed Chunks",
        f"{indexed_count:,}",
    )

    indexed_documents = get_indexed_documents(
        vector_store
    )

    st.metric(
        "Documents",
        len(indexed_documents),
    )

    st.divider()

    st.header("📄 Indexed Documents")

    if indexed_documents:

        for document in sorted(
            indexed_documents,
            key=lambda item: item.get(
                "source_file",
                "",
            ),
        ):

            st.caption(
                f"• {document.get('source_file', 'Unknown')}"
            )

    else:

        st.caption(
            "No documents indexed."
        )

    st.divider()

    st.header("🧩 Chunking")

    st.caption(
        f"Chunk size: {CHUNK_SIZE}"
    )

    st.caption(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# Document upload
# ============================================================

st.subheader("📤 Add Documents")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        "Uploaded PDFs are split into chunks and "
        "added to the local Chroma vector database."
    ),
)


if uploaded_files:

    if st.button(
        "➕ Index Uploaded Documents",
        type="primary",
        use_container_width=True,
    ):

        progress_bar = st.progress(
            0
        )

        status_text = st.empty()

        total_files = len(
            uploaded_files
        )

        indexed_now = 0

        skipped_now = 0

        failed_now = 0

        for file_index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):

            status_text.write(
                f"Processing "
                f"{file_index}/{total_files}: "
                f"{uploaded_file.name}"
            )

            try:

                file_bytes = (
                    uploaded_file.getvalue()
                )

                document_hash = (
                    calculate_file_hash(
                        file_bytes
                    )
                )

                # ------------------------------------------------
                # Duplicate detection
                # ------------------------------------------------

                existing_hashes = set()

                for document in get_indexed_documents(
                    vector_store
                ):

                    existing_hash = document.get(
                        "document_hash"
                    )

                    if existing_hash:
                        existing_hashes.add(
                            existing_hash
                        )

                if document_hash in existing_hashes:

                    st.info(
                        f"Skipped duplicate: "
                        f"{uploaded_file.name}"
                    )

                    skipped_now += 1

                    progress_bar.progress(
                        file_index / total_files
                    )

                    continue

                # ------------------------------------------------
                # Temporary PDF
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temporary_file:

                    temporary_file.write(
                        file_bytes
                    )

                    temporary_path = (
                        Path(
                            temporary_file.name
                        )
                    )

                try:

                    # --------------------------------------------
                    # Load
                    # --------------------------------------------

                    documents = load_pdf(
                        temporary_path
                    )

                    # --------------------------------------------
                    # Add metadata
                    # --------------------------------------------

                    for document in documents:

                        document.metadata[
                            "source_file"
                        ] = uploaded_file.name

                        document.metadata[
                            "source"
                        ] = uploaded_file.name

                        document.metadata[
                            "document_hash"
                        ] = document_hash

                    # --------------------------------------------
                    # Split
                    # --------------------------------------------

                    chunks = split_documents(
                        documents
                    )

                    # --------------------------------------------
                    # Deterministic IDs
                    # --------------------------------------------

                    chunk_ids = []

                    for chunk_index, chunk in enumerate(
                        chunks
                    ):

                        chunk_id = (
                            f"{document_hash}"
                            f"_chunk_{chunk_index}"
                        )

                        chunk.metadata[
                            "chunk_id"
                        ] = chunk_id

                        chunk.metadata[
                            "chunk_index"
                        ] = chunk_index

                        chunk_ids.append(
                            chunk_id
                        )

                    # --------------------------------------------
                    # Add to Chroma
                    # --------------------------------------------

                    add_documents_to_vector_store(
                        vector_store,
                        chunks,
                        ids=chunk_ids,
                    )

                    indexed_now += 1

                    st.success(
                        f"Indexed "
                        f"{uploaded_file.name} "
                        f"({len(chunks):,} chunks)"
                    )

                finally:

                    if temporary_path.exists():
                        temporary_path.unlink()

            except Exception as error:

                failed_now += 1

                st.error(
                    f"Failed to index "
                    f"{uploaded_file.name}: "
                    f"{error}"
                )

            progress_bar.progress(
                file_index / total_files
            )

        status_text.empty()

        st.success(
            f"Indexing finished — "
            f"{indexed_now} added, "
            f"{skipped_now} skipped, "
            f"{failed_now} failed."
        )

        st.cache_resource.clear()

        st.rerun()


# ============================================================
# Current knowledge base status
# ============================================================

indexed_count = get_vector_store_count(
    vector_store
)

if indexed_count == 0:

    st.warning(
        "The knowledge base is empty. "
        "Upload a PDF to begin."
    )

    st.stop()


# ============================================================
# Chat history display
# ============================================================

st.subheader("💬 Ask Your Documents")

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message[
                    "sources"
                ]:

                    st.markdown(
                        f"""
                        **{source['source']}**  
                        Page {source['page']}  
                        Distance: {source['distance']:.4f}
                        """
                    )


# ============================================================
# User question
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Prepare conversation history
    # --------------------------------------------------------

    chat_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages[
            :-1
        ]
    ]

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = generate_rag_response(
                    question=question,
                    vector_store=vector_store,
                    llm=llm,
                    top_k=top_k,
                    chat_history=chat_history,
                    absolute_threshold=absolute_threshold,
                    relative_margin=relative_margin,
                    recovery_margin=recovery_margin,
                    minimum_relevant_documents=(
                        minimum_relevant_documents
                    ),
                    redundancy_threshold=(
                        redundancy_threshold
                    ),
                    max_context_documents=(
                        max_context_documents
                    ),
                    enable_query_expansion=True,
                    max_retrieval_queries=4,
                )

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                st.markdown(
                    answer
                )

                # ------------------------------------------------
                # Source information
                # ------------------------------------------------

                source_data = []

                for document, distance in result.get(
                    "retrieval_scores",
                    [],
                ):

                    source_data.append(
                        {
                            "source":
                                document.metadata.get(
                                    "source_file",
                                    document.metadata.get(
                                        "source",
                                        "Unknown",
                                    ),
                                ),

                            "page":
                                int(
                                    document.metadata.get(
                                        "page",
                                        0,
                                    )
                                ) + 1,

                            "distance":
                                distance,
                        }
                    )

                if source_data:

                    with st.expander(
                        "📚 Retrieved Sources"
                    ):

                        for index, source in enumerate(
                            source_data,
                            start=1,
                        ):

                            st.markdown(
                                f"""
                                **Source {index}**  
                                📄 {source['source']}  
                                📖 Page {source['page']}  
                                📏 Distance: {source['distance']:.4f}
                                """
                            )

                # ------------------------------------------------
                # Diagnostics
                # ------------------------------------------------

                with st.expander(
                    "🔍 Retrieval Diagnostics"
                ):

                    col1, col2, col3, col4 = st.columns(
                        4
                    )

                    with col1:

                        st.metric(
                            "Retrieved",
                            len(
                                result.get(
                                    "all_retrieved_results",
                                    [],
                                )
                            ),
                        )

                    with col2:

                        st.metric(
                            "Relevant",
                            len(
                                result.get(
                                    "retrieval_scores",
                                    [],
                                )
                            ),
                        )

                    with col3:

                        st.metric(
                            "Rejected",
                            len(
                                result.get(
                                    "rejected_documents",
                                    [],
                                )
                            ),
                        )

                    with col4:

                        st.metric(
                            "Redundant",
                            len(
                                result.get(
                                    "redundant_results",
                                    [],
                                )
                            ),
                        )

                    st.write(
                        "Retrieval mode:",
                        result.get(
                            "retrieval_mode",
                            "unknown",
                        ),
                    )

                    st.write(
                        "Detected document:",
                        result.get(
                            "detected_source_file",
                            "None",
                        ),
                    )

                    st.write(
                        "Best distance:",
                        result.get(
                            "best_distance",
                            None,
                        ),
                    )

                    st.write(
                        "Allowed distance:",
                        result.get(
                            "allowed_distance",
                            None,
                        ),
                    )

                    st.write(
                        "LLM context used:",
                        result.get(
                            "used_llm_context",
                            False,
                        ),
                    )

                    st.write(
                        "Rewritten question:",
                        result.get(
                            "standalone_question",
                            "",
                        ),
                    )

                    retrieval_queries = result.get(
                        "retrieval_queries",
                        [],
                    )

                    if retrieval_queries:

                        st.write(
                            "Retrieval queries:"
                        )

                        for query in retrieval_queries:

                            st.caption(
                                f"• {query}"
                            )

                    citation_verification = (
                        result.get(
                            "citation_verification",
                            [],
                        )
                    )

                    if citation_verification:

                        st.write(
                            "Citation verification:"
                        )

                        for citation in (
                            citation_verification
                        ):

                            st.caption(
                                f"Source "
                                f"{citation.get('source_number')}: "
                                f"{citation.get('status')}"
                            )

                # ------------------------------------------------
                # Save assistant message
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_data,
                    }
                )

            except Exception as error:

                error_message = (
                    "An error occurred while "
                    "processing your question."
                )

                st.error(
                    error_message
                )

                st.exception(
                    error
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )