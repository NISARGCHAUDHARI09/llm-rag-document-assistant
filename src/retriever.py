from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from config.settings import TOP_K


def create_retriever(
    vector_store,
    top_k: int = TOP_K,
):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
        },
    )


def retrieve_documents(
    query: str,
    vector_store,
    top_k: int = TOP_K,
    source_file: str | None = None,
) -> List[Document]:

    if not query.strip():
        return []

    if source_file:
        return vector_store.similarity_search(
            query,
            k=top_k,
            filter={
                "source_file": source_file,
            },
        )

    return vector_store.similarity_search(
        query,
        k=top_k,
    )


def retrieve_documents_with_scores(
    query: str,
    vector_store,
    top_k: int = TOP_K,
    source_file: str | None = None,
) -> List[Tuple[Document, float]]:

    if not query.strip():
        return []

    if source_file:
        return vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={
                "source_file": source_file,
            },
        )

    return vector_store.similarity_search_with_score(
        query,
        k=top_k,
    )


def get_indexed_source_files(
    vector_store,
) -> List[str]:

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

    source_files = set()

    for metadata in metadatas:

        if not metadata:
            continue

        source_file = metadata.get(
            "source_file"
        )

        if not source_file:
            source_file = metadata.get(
                "source"
            )

        if source_file:
            source_files.add(
                Path(source_file).name
            )

    return sorted(source_files)


def _normalize_document_text(
    text: str,
) -> str:

    text = text.lower()

    replacements = {
        "_": " ",
        "-": " ",
        ".pdf": " ",
        ".": " ",
        "/": " ",
        "\\": " ",
        "(": " ",
        ")": " ",
        ",": " ",
        ":": " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return " ".join(
        word
        for word in text.split()
        if word.strip()
    )


def _meaningful_document_tokens(
    source_file: str,
) -> List[str]:

    normalized = _normalize_document_text(
        source_file
    )

    ignored_words = {
        "eis",
        "pdf",
        "document",
        "project",
        "plan",
        "final",
        "draft",
        "report",
        "environmental",
        "statement",
        "impact",
        "for",
        "the",
        "and",
        "of",
        "site",
        "october",
        "2023",
        "508",
    }

    return [
        token
        for token in normalized.split()
        if token not in ignored_words
        and len(token) >= 3
    ]


def detect_explicit_source_file(
    question: str,
    vector_store,
) -> str | None:
    """
    Detect whether the question explicitly refers to one of the
    documents currently indexed in Chroma.
    """

    if not question.strip():
        return None

    source_files = get_indexed_source_files(
        vector_store
    )

    if not source_files:
        return None

    normalized_question = (
        _normalize_document_text(
            question
        )
    )

    question_tokens = set(
        normalized_question.split()
    )

    candidates = []

    for source_file in source_files:

        source_tokens = (
            _meaningful_document_tokens(
                source_file
            )
        )

        if not source_tokens:
            continue

        matched_tokens = [
            token
            for token in source_tokens
            if token in question_tokens
        ]

        coverage = (
            len(matched_tokens)
            / len(source_tokens)
        )

        candidates.append(
            {
                "source_file": source_file,
                "source_tokens": source_tokens,
                "matched_tokens": matched_tokens,
                "coverage": coverage,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["coverage"],
            len(item["matched_tokens"]),
        ),
        reverse=True,
    )

    best = candidates[0]

    matched_count = len(
        best["matched_tokens"]
    )

    coverage = best["coverage"]

    if (
        matched_count >= 2
        and coverage >= 0.40
    ):
        return best["source_file"]

    if (
        matched_count == 1
        and len(best["source_tokens"]) == 1
    ):
        return best["source_file"]

    return None


def retrieve_multi_query_documents_with_scores(
    queries: List[str],
    vector_store,
    top_k: int = TOP_K,
    source_file: str | None = None,
) -> List[
    Tuple[
        Document,
        float,
        int,
        int,
    ]
]:

    if not queries:
        return []

    merged: Dict[str, dict] = {}

    for query_index, query in enumerate(
        queries
    ):

        query = query.strip()

        if not query:
            continue

        results = retrieve_documents_with_scores(
            query=query,
            vector_store=vector_store,
            top_k=top_k,
            source_file=source_file,
        )

        for rank, (
            document,
            distance,
        ) in enumerate(results):

            chunk_id = document.metadata.get(
                "chunk_id"
            )

            if not chunk_id:
                chunk_id = (
                    f"{document.metadata.get('source_file', '')}"
                    f"__"
                    f"{document.metadata.get('page', '')}"
                    f"__"
                    f"{hash(document.page_content)}"
                )

            if chunk_id not in merged:

                merged[chunk_id] = {
                    "document": document,
                    "best_distance": float(
                        distance
                    ),
                    "query_hits": 1,
                    "best_rank": rank,
                    "query_indexes": {
                        query_index
                    },
                }

            else:

                item = merged[
                    chunk_id
                ]

                item[
                    "best_distance"
                ] = min(
                    item["best_distance"],
                    float(distance),
                )

                if (
                    query_index
                    not in item[
                        "query_indexes"
                    ]
                ):
                    item[
                        "query_hits"
                    ] += 1

                    item[
                        "query_indexes"
                    ].add(
                        query_index
                    )

                item[
                    "best_rank"
                ] = min(
                    item["best_rank"],
                    rank,
                )

    merged_items = list(
        merged.values()
    )

    merged_items.sort(
        key=lambda item: (
            item["best_distance"],
            -item["query_hits"],
            item["best_rank"],
        )
    )

    return [
        (
            item["document"],
            item["best_distance"],
            item["query_hits"],
            item["best_rank"],
        )
        for item in merged_items
    ]


def filter_documents_by_threshold(
    results: List[
        Tuple[Document, float]
    ],
    threshold: float,
) -> Tuple[
    List[Tuple[Document, float]],
    List[Tuple[Document, float]],
]:

    relevant = []
    rejected = []

    for document, distance in results:

        if distance <= threshold:
            relevant.append(
                (
                    document,
                    distance,
                )
            )
        else:
            rejected.append(
                (
                    document,
                    distance,
                )
            )

    return relevant, rejected


def filter_documents_by_relative_threshold(
    results: List[
        Tuple[Document, float]
    ],
    relative_margin: float = 0.05,
) -> Tuple[
    List[Tuple[Document, float]],
    List[Tuple[Document, float]],
    float,
]:

    if not results:
        return [], [], 0.0

    best_distance = min(
        distance
        for _, distance in results
    )

    allowed_distance = (
        best_distance
        + relative_margin
    )

    relevant = []
    rejected = []

    for document, distance in results:

        if distance <= allowed_distance:
            relevant.append(
                (
                    document,
                    distance,
                )
            )
        else:
            rejected.append(
                (
                    document,
                    distance,
                )
            )

    return (
        relevant,
        rejected,
        allowed_distance,
    )


def adaptive_filter_documents(
    results: List[
        Tuple[Document, float]
    ],
    absolute_threshold: float = 1.00,
    strict_margin: float = 0.05,
    recovery_margin: float = 0.10,
    minimum_relevant_documents: int = 3,
) -> Tuple[
    List[Tuple[Document, float]],
    List[Tuple[Document, float]],
    float,
    float,
    str,
]:

    if not results:
        return (
            [],
            [],
            float("inf"),
            absolute_threshold,
            "none",
        )

    results = sorted(
        results,
        key=lambda item: item[1],
    )

    best_distance = results[0][1]

    strict_allowed = min(
        absolute_threshold,
        best_distance + strict_margin,
    )

    strict_relevant = [
        item
        for item in results
        if item[1] <= strict_allowed
    ]

    strict_rejected = [
        item
        for item in results
        if item[1] > strict_allowed
    ]

    if len(strict_relevant) >= minimum_relevant_documents:

        return (
            strict_relevant,
            strict_rejected,
            best_distance,
            strict_allowed,
            "strict",
        )

    recovery_allowed = min(
        absolute_threshold,
        best_distance + recovery_margin,
    )

    recovery_relevant = [
        item
        for item in results
        if item[1] <= recovery_allowed
    ]

    recovery_rejected = [
        item
        for item in results
        if item[1] > recovery_allowed
    ]

    if recovery_relevant:

        return (
            recovery_relevant,
            recovery_rejected,
            best_distance,
            recovery_allowed,
            "recovery",
        )

    return (
        [],
        results,
        best_distance,
        absolute_threshold,
        "none",
    )


def _tokenize_for_redundancy(
    text: str,
) -> set[str]:

    normalized = (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
    )

    return {
        token
        for token in normalized.split()
        if len(token) > 2
    }


def _jaccard_similarity(
    first: set[str],
    second: set[str],
) -> float:

    if not first or not second:
        return 0.0

    intersection = len(
        first & second
    )

    union = len(
        first | second
    )

    if union == 0:
        return 0.0

    return intersection / union


def rank_and_deduplicate_documents(
    results: List[
        Tuple[Document, float]
    ],
    redundancy_threshold: float = 0.80,
    max_documents: int | None = None,
) -> Tuple[
    List[Tuple[Document, float]],
    List[Tuple[Document, float]],
]:

    if not results:
        return [], []

    sorted_results = sorted(
        results,
        key=lambda item: item[1],
    )

    selected = []
    redundant = []
    selected_tokens = []

    for document, distance in sorted_results:

        current_tokens = (
            _tokenize_for_redundancy(
                document.page_content
            )
        )

        is_redundant = False

        for existing_tokens in selected_tokens:

            similarity = _jaccard_similarity(
                current_tokens,
                existing_tokens,
            )

            if (
                similarity
                >= redundancy_threshold
            ):
                is_redundant = True
                break

        if is_redundant:

            redundant.append(
                (
                    document,
                    distance,
                )
            )

            continue

        selected.append(
            (
                document,
                distance,
            )
        )

        selected_tokens.append(
            current_tokens
        )

        if (
            max_documents is not None
            and len(selected)
            >= max_documents
        ):
            break

    return (
        selected,
        redundant,
    )


def format_retrieved_documents(
    documents: List[Document],
) -> str:

    formatted_chunks = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source_file = document.metadata.get(
            "source_file",
            document.metadata.get(
                "source",
                "Unknown",
            ),
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        try:
            display_page = int(page) + 1
        except (
            TypeError,
            ValueError,
        ):
            display_page = page

        formatted_chunks.append(
            f"[Source {index}: "
            f"{Path(source_file).name}, "
            f"Page {display_page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        formatted_chunks
    )