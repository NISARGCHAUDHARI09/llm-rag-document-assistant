import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Project Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------
# LLM Configuration
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# Embedding Configuration
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Text Splitting Configuration
# -----------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# -----------------------------
# Retrieval Configuration
# -----------------------------
TOP_K = 4

# -----------------------------
# Vector Database Configuration
# -----------------------------
VECTOR_DB_PATH = str(PROJECT_ROOT / "vectorstore")
COLLECTION_NAME = "rag_documents"