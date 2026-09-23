import hashlib
from pathlib import Path


def calculate_file_hash(
    file_bytes: bytes,
) -> str:
    """
    Calculate a SHA-256 hash for a file.

    The hash is used to identify whether the exact
    same document has already been indexed.
    """

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def create_chunk_id(
    document_hash: str,
    chunk_index: int,
) -> str:
    """
    Create a deterministic ID for a document chunk.
    """

    return (
        f"{document_hash}_chunk_{chunk_index}"
    )


def get_file_extension(
    filename: str,
) -> str:
    """
    Return the lowercase file extension.
    """

    return Path(
        filename
    ).suffix.lower()