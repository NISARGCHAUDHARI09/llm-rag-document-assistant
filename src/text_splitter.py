from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def split_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Split LangChain documents into smaller text chunks.

    Each chunk retains the original document metadata
    and receives a deterministic chunk ID when the
    document hash is available.
    """

    if not documents:
        raise ValueError(
            "No documents provided for splitting."
        )

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # Track chunk number separately for each document.
    document_chunk_counts = {}

    for chunk in chunks:

        # --------------------------------------------------
        # Get source information
        # --------------------------------------------------

        source = Path(
            chunk.metadata.get(
                "source",
                "unknown",
            )
        ).name

        source_file = chunk.metadata.get(
            "source_file",
            source,
        )

        page = chunk.metadata.get(
            "page",
            "unknown",
        )

        document_hash = chunk.metadata.get(
            "document_hash"
        )

        # --------------------------------------------------
        # Keep uploaded filename
        # --------------------------------------------------

        chunk.metadata[
            "source_file"
        ] = source_file

        chunk.metadata[
            "source"
        ] = source_file

        # --------------------------------------------------
        # Determine chunk index within document
        # --------------------------------------------------

        if document_hash:

            current_index = (
                document_chunk_counts.get(
                    document_hash,
                    0,
                )
            )

            document_chunk_counts[
                document_hash
            ] = current_index + 1

            chunk.metadata[
                "chunk_index"
            ] = current_index

            chunk.metadata[
                "chunk_id"
            ] = (
                f"{document_hash}"
                f"_chunk_{current_index}"
            )

        else:

            # Fallback for documents that do not
            # contain a document hash.

            current_index = (
                len(document_chunk_counts)
            )

            chunk.metadata[
                "chunk_id"
            ] = (
                f"{source_file}"
                f"_page_{page}"
                f"_chunk_{current_index}"
            )

    return chunks