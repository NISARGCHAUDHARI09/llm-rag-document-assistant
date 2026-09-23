import re
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.retriever import (
    adaptive_filter_documents,
    detect_explicit_source_file,
    format_retrieved_documents,
    rank_and_deduplicate_documents,
    retrieve_documents_with_scores,
    retrieve_multi_query_documents_with_scores,
)


SYSTEM_PROMPT = """
You are a grounded document question-answering assistant.

Your job is to answer questions ONLY using the supplied document
excerpts.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the supplied excerpts do not contain enough information to
   answer the question, clearly say that the information could
   not be found in the provided documents.
4. Every factual claim based on the retrieved documents must use
   an exact source citation.
5. Use this exact citation format:

   [Source N: document_name.pdf, Page 123]

6. Do not create source numbers that do not exist in the supplied
   context.
7. Do not cite a source merely because it is related to the topic.
   The cited excerpt must actually support the statement.
8. When multiple documents are present, use only the document
   evidence relevant to the question.
"""


QUERY_EXPANSION_PROMPT = """
Generate retrieval queries for the user's question.

The goal is to improve semantic retrieval from a document collection.

Rules:

- Generate 3 to 5 concise retrieval queries.
- Preserve the meaning of the original question.
- Preserve explicit document names, project names, acronyms,
  organizations, locations, and technical terms.
- Do not answer the question.
- Do not introduce facts that are not present in the question.
- If the question already identifies a document, preserve that
  document identity in the generated queries.
- Return one query per line.
- Do not number the queries.
"""


def rewrite_question(
    question: str,
    chat_history: List[dict] | None = None,
    llm=None,
) -> str:
    """
    Convert a conversational follow-up into a standalone question.

    If there is no chat history, the original question is returned.
    """

    if not question.strip():
        return question

    if not chat_history:
        return question.strip()

    history_lines = []

    for message in chat_history:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        ).strip()

        if content:
            history_lines.append(
                f"{role}: {content}"
            )

    if not history_lines:
        return question.strip()

    history_text = "\n".join(
        history_lines[-8:]
    )

    rewrite_prompt = f"""
Rewrite the user's latest question into a standalone retrieval
question.

Conversation:
{history_text}

Latest question:
{question}

Rules:
- Preserve the exact topic from the conversation.
- Preserve any document, project, location, or subject already
  established in the conversation.
- Resolve pronouns such as "these", "this", "they", and "it".
- Do not answer the question.
- Return only the rewritten question.
"""

    try:

        response = llm.invoke(
            rewrite_prompt
        )

        rewritten = response.content.strip()

        if rewritten:
            return rewritten

    except Exception:
        pass

    return question.strip()


def format_chat_history(
    chat_history: List[dict] | None,
) -> str:
    if not chat_history:
        return ""

    lines = []

    for message in chat_history:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        ).strip()

        if content:
            lines.append(
                f"{role}: {content}"
            )

    return "\n".join(lines)


def generate_retrieval_queries(
    question: str,
    llm,
    max_queries: int = 4,
) -> List[str]:
    """
    Generate multiple retrieval queries.

    The original question is always retained.
    """

    if not question.strip():
        return []

    queries = [
        question.strip()
    ]

    try:

        response = llm.invoke(
            QUERY_EXPANSION_PROMPT
            + "\n\nUser question:\n"
            + question
        )

        generated_text = (
            response.content.strip()
        )

        generated_lines = (
            generated_text.splitlines()
        )

        for line in generated_lines:

            cleaned = re.sub(
                r"^\s*[-*•\d.)]+\s*",
                "",
                line,
            ).strip()

            if not cleaned:
                continue

            if cleaned.lower() in {
                query.lower()
                for query in queries
            }:
                continue

            queries.append(
                cleaned
            )

            if len(queries) >= max_queries:
                break

    except Exception:
        pass

    return queries[:max_queries]


def format_context(
    documents: List[Document],
) -> str:
    return format_retrieved_documents(
        documents
    )


def build_rag_prompt(
    question: str,
    context: str,
    chat_history: List[dict] | None = None,
) -> str:
    history_text = (
        format_chat_history(
            chat_history
        )
    )

    if history_text:
        history_section = f"""
Conversation history:
{history_text}
"""
    else:
        history_section = ""

    return f"""
{SYSTEM_PROMPT}

{history_section}

Retrieved document excerpts:
{context}

Current question:
{question}

Answer the current question using only the retrieved excerpts.

If the excerpts do not provide enough evidence, say that the
information could not be found in the provided documents.

Include exact source citations for factual statements.
"""


def extract_citation_source_numbers(
    answer: str,
) -> List[int]:
    matches = re.findall(
        r"\[Source\s+(\d+)\s*:",
        answer,
        flags=re.IGNORECASE,
    )

    return sorted(
        {
            int(number)
            for number in matches
        }
    )


def verify_citations(
    answer: str,
    documents: List[Document],
) -> List[dict]:
    """
    Verify that citations refer to real supplied sources and that
    the exact citation text exists in the answer.
    """

    verification = []

    cited_numbers = (
        extract_citation_source_numbers(
            answer
        )
    )

    source_expectations = {}

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

        citation_text = (
            f"[Source {index}: "
            f"{source_file}, "
            f"Page {display_page}]"
        )

        source_expectations[index] = {
            "source_file": source_file,
            "page": display_page,
            "citation": citation_text,
        }

    for source_number in cited_numbers:

        if source_number not in source_expectations:

            verification.append(
                {
                    "source_number": source_number,
                    "status": "invalid",
                    "expected_citation": None,
                }
            )

            continue

        expected = source_expectations[
            source_number
        ]

        if expected["citation"] in answer:

            verification.append(
                {
                    "source_number": source_number,
                    "status": "used",
                    "expected_citation":
                        expected["citation"],
                }
            )

        else:

            verification.append(
                {
                    "source_number": source_number,
                    "status": "citation_mismatch",
                    "expected_citation":
                        expected["citation"],
                }
            )

    for source_number, expected in (
        source_expectations.items()
    ):

        if source_number not in cited_numbers:

            verification.append(
                {
                    "source_number": source_number,
                    "status": "not_cited",
                    "expected_citation":
                        expected["citation"],
                }
            )

    return verification


def _convert_multi_query_results(
    results,
) -> List:
    """
    Convert multi-query retrieval results into the two-value
    structure expected by the adaptive filtering layer.

    Query agreement remains available separately in diagnostics.
    """

    return [
        (
            document,
            distance,
        )
        for (
            document,
            distance,
            _query_hits,
            _best_rank,
        ) in results
    ]


def generate_rag_response(
    question: str,
    vector_store,
    llm,
    top_k: int = 4,
    chat_history: List[dict] | None = None,
    absolute_threshold: float = 1.00,
    relative_margin: float = 0.05,
    recovery_margin: float = 0.10,
    minimum_relevant_documents: int = 3,
    redundancy_threshold: float = 0.80,
    max_context_documents: int | None = None,
    enable_query_expansion: bool = True,
    max_retrieval_queries: int = 4,
) -> Dict[str, Any]:
    """
    Complete grounded RAG pipeline.

    Pipeline:

        question
            ↓
        conversation rewrite
            ↓
        explicit document detection
            ↓
        query expansion
            ↓
        multi-query retrieval
            ↓
        adaptive filtering
            ↓
        redundancy removal
            ↓
        grounded LLM generation
            ↓
        citation verification
    """

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if vector_store is None:
        raise ValueError(
            "Vector store is required."
        )

    if llm is None:
        raise ValueError(
            "LLM is required."
        )

    # ---------------------------------------------------------
    # 1. Rewrite conversational question
    # ---------------------------------------------------------

    standalone_question = rewrite_question(
        question=question,
        chat_history=chat_history,
        llm=llm,
    )

    # ---------------------------------------------------------
    # 2. Detect explicit document reference
    # ---------------------------------------------------------

    detected_source_file = (
        detect_explicit_source_file(
            standalone_question,
            vector_store,
        )
    )

    # ---------------------------------------------------------
    # 3. Query expansion
    # ---------------------------------------------------------

    if enable_query_expansion:

        retrieval_queries = (
            generate_retrieval_queries(
                question=standalone_question,
                llm=llm,
                max_queries=max_retrieval_queries,
            )
        )

    else:

        retrieval_queries = [
            standalone_question
        ]

    # Make sure the standalone question is always first.
    retrieval_queries = [
        standalone_question
    ] + [
        query
        for query in retrieval_queries
        if query.strip().lower()
        != standalone_question.strip().lower()
    ]

    retrieval_queries = (
        retrieval_queries[
            :max_retrieval_queries
        ]
    )

    # ---------------------------------------------------------
    # 4. Retrieve
    # ---------------------------------------------------------

    multi_query_results = (
        retrieve_multi_query_documents_with_scores(
            queries=retrieval_queries,
            vector_store=vector_store,
            top_k=top_k,
            source_file=detected_source_file,
        )
    )

    # If an explicit document was detected but retrieval
    # unexpectedly returned nothing, perform one safe fallback
    # against that same document using the original question.
    #
    # This never expands beyond the requested document.
    if (
        detected_source_file
        and not multi_query_results
    ):

        fallback_results = (
            retrieve_documents_with_scores(
                query=standalone_question,
                vector_store=vector_store,
                top_k=top_k,
                source_file=detected_source_file,
            )
        )

        multi_query_results = [
            (
                document,
                distance,
                1,
                rank,
            )
            for rank, (
                document,
                distance,
            ) in enumerate(
                fallback_results
            )
        ]

    # ---------------------------------------------------------
    # 5. Preserve query-agreement diagnostics
    # ---------------------------------------------------------

    query_agreement_metadata = []

    for (
        document,
        distance,
        query_hits,
        best_rank,
    ) in multi_query_results:

        query_agreement_metadata.append(
            {
                "document": document,
                "distance": distance,
                "query_hits": query_hits,
                "best_rank": best_rank,
            }
        )

    # Adaptive filtering works with:
    # (Document, distance)

    filtering_results = (
        _convert_multi_query_results(
            multi_query_results
        )
    )

    # ---------------------------------------------------------
    # 6. Adaptive relevance filtering
    # ---------------------------------------------------------

    (
        relevant_results,
        rejected_results,
        best_distance,
        allowed_distance,
        retrieval_mode,
    ) = adaptive_filter_documents(
        results=filtering_results,
        absolute_threshold=absolute_threshold,
        strict_margin=relative_margin,
        recovery_margin=recovery_margin,
        minimum_relevant_documents=minimum_relevant_documents,
    )

    # ---------------------------------------------------------
    # 7. Refuse if no evidence survives
    # ---------------------------------------------------------

    if not relevant_results:

        answer = (
            "I’m sorry, but the provided documents do not "
            "contain enough relevant information to answer "
            "this question."
        )

        return {
            "answer": answer,
            "source_documents": [],
            "standalone_question":
                standalone_question,
            "retrieval_scores": [],
            "rejected_documents": [
                document
                for document, _ in rejected_results
            ],
            "rejected_scores": rejected_results,
            "all_retrieved_results":
                filtering_results,
            "relevance_threshold":
                allowed_distance,
            "absolute_threshold":
                absolute_threshold,
            "relative_margin":
                relative_margin,
            "recovery_margin":
                recovery_margin,
            "minimum_relevant_documents":
                minimum_relevant_documents,
            "best_distance":
                best_distance,
            "allowed_distance":
                allowed_distance,
            "retrieval_mode":
                "none",
            "used_llm_context":
                False,
            "ranked_results": [],
            "redundant_results": [],
            "citation_verification": [],
            "redundancy_threshold":
                redundancy_threshold,
            "max_context_documents":
                max_context_documents,
            "retrieval_queries":
                retrieval_queries,
            "detected_source_file":
                detected_source_file,
            "query_agreement_metadata":
                query_agreement_metadata,
        }

    # ---------------------------------------------------------
    # 8. Rank and remove redundant chunks
    # ---------------------------------------------------------

    ranked_results, redundant_results = (
        rank_and_deduplicate_documents(
            results=relevant_results,
            redundancy_threshold=redundancy_threshold,
            max_documents=max_context_documents,
        )
    )

    # Safety fallback.
    if not ranked_results:

        ranked_results = [
            relevant_results[0]
        ]

        redundant_results = []

    context_documents = [
        document
        for document, _ in ranked_results
    ]

    # ---------------------------------------------------------
    # 9. Build grounded context
    # ---------------------------------------------------------

    context = format_context(
        context_documents
    )

    prompt = build_rag_prompt(
        question=standalone_question,
        context=context,
        chat_history=chat_history,
    )

    # ---------------------------------------------------------
    # 10. Generate answer
    # ---------------------------------------------------------

    response = llm.invoke(
        prompt
    )

    answer = response.content.strip()

    # ---------------------------------------------------------
    # 11. Verify citations
    # ---------------------------------------------------------

    citation_verification = (
        verify_citations(
            answer=answer,
            documents=context_documents,
        )
    )

    return {
        "answer": answer,
        "source_documents":
            context_documents,
        "standalone_question":
            standalone_question,
        "retrieval_scores":
            ranked_results,
        "rejected_documents": [
            document
            for document, _ in rejected_results
        ],
        "rejected_scores":
            rejected_results,
        "all_retrieved_results":
            filtering_results,
        "relevance_threshold":
            allowed_distance,
        "absolute_threshold":
            absolute_threshold,
        "relative_margin":
            relative_margin,
        "recovery_margin":
            recovery_margin,
        "minimum_relevant_documents":
            minimum_relevant_documents,
        "best_distance":
            best_distance,
        "allowed_distance":
            allowed_distance,
        "retrieval_mode":
            retrieval_mode,
        "used_llm_context":
            True,
        "ranked_results":
            ranked_results,
        "redundant_results":
            redundant_results,
        "citation_verification":
            citation_verification,
        "redundancy_threshold":
            redundancy_threshold,
        "max_context_documents":
            max_context_documents,
        "retrieval_queries":
            retrieval_queries,
        "detected_source_file":
            detected_source_file,
        "query_agreement_metadata":
            query_agreement_metadata,
    }