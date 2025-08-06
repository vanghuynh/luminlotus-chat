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

#### OpenAI LLM Configuration
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from dotenv import load_dotenv
# from src.utils.logger import logger
# import os

# # Load biến môi trường từ file .env
# load_dotenv()

# # Lấy API key từ biến môi trường
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # ===========================
# # ✅ Embedding Model (cho vector search, nếu có)
# # ===========================
# embeddings = OpenAIEmbeddings(
#     openai_api_key=OPENAI_API_KEY
# )

# # ===========================
# # ✅ Hàm khởi tạo ChatOpenAI
# # ===========================
# def get_llm(model_name: str = "gpt-3.5-turbo", api_key: str = None) -> ChatOpenAI:
#     """
#     Khởi tạo LLM từ OpenAI với mô hình và API key cụ thể.

#     Args:
#         model_name (str): Tên mô hình, ví dụ: "gpt-3.5-turbo", "gpt-4", "gpt-4o"
#         api_key (str): API key tùy chọn. Nếu không truyền, sẽ dùng OPENAI_API_KEY từ .env

#     Returns:
#         ChatOpenAI: Instance của LangChain ChatOpenAI
#     """
#     key_to_use = api_key or OPENAI_API_KEY

#     if not key_to_use:
#         raise ValueError("❌ Thiếu OpenAI API Key. Vui lòng cấu hình biến OPENAI_API_KEY trong file .env.")

#     logger.info(f"[LLM] ✅ Khởi tạo ChatOpenAI với model: {model_name}")

#     return ChatOpenAI(
#         model=model_name,
#         temperature=0.7,
#         openai_api_key=key_to_use
#     )
# # ===========================
# # ✅ Mặc định sử dụng mô hình gpt-3.5-turbo