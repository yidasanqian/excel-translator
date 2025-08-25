"""应用配置管理模块."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类."""

    openai_base_url: str = Field(..., env="OPENAI_BASE_URL")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    max_batch_size: int = Field(default=50, env="MAX_BATCH_SIZE", ge=1, le=100)
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT", ge=1, le=300)
    preserve_format: bool = Field(default=True, env="PRESERVE_FORMAT")
    # 批量翻译设置
    max_tokens: int = Field(default=4096, env="MAX_TOKENS", ge=10, le=128000)
    token_buffer: int = Field(default=500, env="TOKEN_BUFFER", ge=10, le=500)
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    output_dir: str = Field(default="output", env="OUTPUT_DIR")
    max_file_size: int = Field(
        default=10485760, env="MAX_FILE_SIZE", ge=1024, le=104857600
    )
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        """Pydantic配置."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
