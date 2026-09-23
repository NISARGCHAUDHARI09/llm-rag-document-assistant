from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import (
    COLLECTION_NAME,
    VECTOR_DB_PATH,
)


def create_vector_store(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Create a persistent ChromaDB vector store from documents.
    """

    if not documents:
        raise ValueError(
            "No documents provided for vector store creation."
        )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
    )

    return vector_store


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Load an existing persistent ChromaDB vector store.
    """

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=VECTOR_DB_PATH,
    )

    return vector_store


def add_documents_to_vector_store(
    vector_store: Chroma,
    documents: List[Document],
    ids: List[str] | None = None,
) -> None:
    """
    Add document chunks to an existing ChromaDB vector store.
    """

    if not documents:
        raise ValueError(
            "No documents provided for indexing."
        )

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )


def get_vector_store_count(
    vector_store: Chroma,
) -> int:
    """
    Return the total number of chunks stored in ChromaDB.
    """

    return vector_store._collection.count()


def get_indexed_document_hashes(
    vector_store: Chroma,
) -> set[str]:
    """
    Retrieve all document hashes currently stored in ChromaDB.

    Because the hashes are stored inside Chroma metadata,
    this information survives Streamlit restarts.
    """

    collection = vector_store._collection

    if collection.count() == 0:
        return set()

    result = collection.get(
        include=["metadatas"]
    )

    metadatas = result.get(
        "metadatas",
        [],
    )

    hashes = set()

    for metadata in metadatas:

        if not metadata:
            continue

        document_hash = metadata.get(
            "document_hash"
        )

        if document_hash:
            hashes.add(
                document_hash
            )

    return hashes


def get_indexed_documents(
    vector_store: Chroma,
) -> List[dict]:
    """
    Return a unique list of indexed documents.

    Each document contains:
        - source_file
        - document_hash
    """

    collection = vector_store._collection

    if collection.count() == 0:
        return []

    result = collection.get(
        include=["metadatas"]
    )

    metadatas = result.get(
        "metadatas",
        [],
    )

    documents = {}
    
    for metadata in metadatas:

        if not metadata:
            continue

        source_file = metadata.get(
            "source_file",
            metadata.get(
                "source",
                "Unknown",
            ),
        )

        document_hash = metadata.get(
            "document_hash"
        )

        if document_hash:

            documents[
                document_hash
            ] = {
                "source_file": source_file,
                "document_hash": document_hash,
            }

        elif source_file != "Unknown":

            # Fallback for older records that may not
            # contain a document hash.
            documents[
                source_file
            ] = {
                "source_file": source_file,
                "document_hash": None,
            }

    return list(
        documents.values()
    )


def clear_vector_store(
    vector_store: Chroma,
) -> None:
    """
    Delete all documents from the current ChromaDB collection.
    """

    collection = vector_store._collection

    result = collection.get()

    ids = result.get(
        "ids",
        [],
    )

    if ids:

        collection.delete(
            ids=ids
        )