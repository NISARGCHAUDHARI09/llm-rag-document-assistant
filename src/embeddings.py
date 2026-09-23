from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL_NAME


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return the Hugging Face embedding model.

    Returns:
        HuggingFaceEmbeddings: Configured embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embedding_model