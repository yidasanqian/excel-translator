"""应用配置管理模块."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """应用配置类."""

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    target_language: str = Field(default="english", env="TARGET_LANGUAGE")
    max_batch_size: int = Field(default=50, env="MAX_BATCH_SIZE", ge=1, le=100)
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT", ge=1, le=300)
    preserve_format: bool = Field(default=True, env="PRESERVE_FORMAT")
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    output_dir: str = Field(default="output", env="OUTPUT_DIR")
    max_file_size: int = Field(
        default=10485760, env="MAX_FILE_SIZE", ge=1024, le=104857600
    )
    app_name: str = Field(default="Excel Translator", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        """Pydantic配置."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
