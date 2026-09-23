from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY


def get_llm(
    model_name: str = "openai/gpt-oss-20b",
    temperature: float = 0.0,
):
    """
    Create and return the Groq chat model.

    Args:
        model_name: Groq model to use.
        temperature: Controls response randomness.

    Returns:
        Configured ChatGroq instance.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file."
        )

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
    )

    return llm