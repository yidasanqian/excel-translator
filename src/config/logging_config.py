"""日志配置模块."""

import logging
import logging.config
from typing import Optional
from .settings import settings


def setup_logging(level: Optional[str] = None, format_str: Optional[str] = None):
    """
    设置日志配置.

    Args:
        level: 日志级别，默认从配置中读取
        format_str: 日志格式，默认使用标准格式
    """
    log_level = level or getattr(
        logging,
        settings.log_level.upper() if hasattr(settings, "log_level") else "INFO",
    )
    log_format = format_str or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=log_level, format=log_format, handlers=[logging.StreamHandler()]
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger实例.

    Args:
        name: logger名称，通常使用__name__

    Returns:
        配置好的logger实例
    """
    return logging.getLogger(name)
