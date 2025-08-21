"""API数据模型定义."""

from pydantic import BaseModel
from typing import Dict, Optional


class TranslateRequest(BaseModel):
    """翻译请求数据模型."""

    source_language: str
    target_language: str
    model: Optional[str] = None
    domain_terms: Optional[Dict[str, Dict[str, str]]] = None


class ProgressMessage(BaseModel):
    """进度消息数据模型."""

    type: str  # "progress", "complete", "error", "file"
    progress: Optional[float] = None
    message: Optional[str] = None
    filename: Optional[str] = None
    content: Optional[str] = None  # Base64编码的文件内容


class SSEMessageType:
    """SSE消息类型常量."""

    PROGRESS = "progress"
    COMPLETE = "complete"
    ERROR = "error"
    FILE = "file"
