from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"

class Settings(BaseSettings):
    """
    Centralized, type-safe application settings.

    pydantic-settings automatically reads value from .env and validates their type
    if a required setting is missing, the app will fail to start with clear error.
    """

    # --- API Keys---
    google_api_key : str

    # ---Paths---
    chroma_persist_dir:str = "chroma_db"
    upload_dir :str = "uploads"

    # --chunking settings ---
    chunk_size : int = 1000
    chunk_overlap :int = 200

    # --- Model Settings ---
    llm_model:str = "gemini-2.5-flash"
    embeddings_model: str = "models/gemini-embedding-001"

    # --Retrieval Settings ---
    retrieval_k:int = 6

    # This tells pydantic - settings to load variables from a .env file
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH,env_file_encoding="utf-8")

# Single shared settings instance — import this everywhere instead of
# creating new Settings() objects, so the .env file is only read once.
settings = Settings()