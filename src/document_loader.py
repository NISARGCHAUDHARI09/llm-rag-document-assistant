from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a PDF document and return its pages as LangChain Documents.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of LangChain Document objects.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {file_path.suffix}"
        )

    loader = PyPDFLoader(str(file_path))

    documents = loader.load()

    return documents


def load_pdfs_from_directory(
    directory: str | Path
) -> List[Document]:
    """
    Load all PDF files from a directory.

    Args:
        directory: Directory containing PDF files.

    Returns:
        A combined list of LangChain Document objects.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {directory}"
        )

    all_documents = []

    for pdf_file in pdf_files:
        documents = load_pdf(pdf_file)
        all_documents.extend(documents)

    return all_documents