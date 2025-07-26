from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from src.utils.logger import logger
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Default model instances
llm_2_0 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1,
    google_api_key=GOOGLE_API_KEY
)


# Default embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def get_llm(model_name: str, api_key: str = None) -> ChatGoogleGenerativeAI:
    """
    Get LLM instance based on model name and optional API key.

    Args:
        model_name: Name of the model to use
        api_key: Optional API key for authentication

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Raises:
        ValueError: If model name is not supported
    """
    if api_key:
        logger.warning("Using custom API key")
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=1, google_api_key=api_key
        )

    if model_name == "gemini-2.0-flash":
        return llm_2_0
    
    raise ValueError(f"Unknown model: {model_name}")
