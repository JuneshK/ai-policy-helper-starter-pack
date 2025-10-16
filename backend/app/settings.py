from pydantic import BaseModel
import os

class Settings(BaseModel):
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "local-384")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # stub | openai | ollama
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")  # qdrant | memory
    collection_name: str = os.getenv("COLLECTION_NAME", "policy_helper")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    data_dir: str = os.getenv("DATA_DIR", "/app/data")

settings = Settings()
