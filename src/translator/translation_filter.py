"""翻译过滤器 - 用于判断文本是否需要翻译."""

import re


def needs_translation(text: str, target_language: str) -> bool:
    """
    判断文本是否需要翻译.

    Args:
        text: 待检查的文本
        target_language: 目标语言

    Returns:
        bool: 如果需要翻译返回True，否则返回False
    """
    if not text:
        return False

    # 检查是否为纯数字或数字与符号的组合
    if _is_numeric_or_symbolic(text):
        return False

    # 如果目标语言是英语
    if target_language.lower() in ["english", "en", "en-us"]:
        # 检查是否为纯英文、纯数字或英文与数字混合
        if _is_english_mixed(text):
            return False

    # 如果目标语言是中文
    elif target_language.lower() in ["chinese", "zh", "zh-cn"]:
        # 检查是否为纯中文、纯数字或中文与数字混合
        if _is_chinese_mixed(text):
            return False

    return True


def _is_numeric_or_symbolic(text: str) -> bool:
    """检查文本是否为纯数字或数字与符号的组合."""
    # 匹配纯数字、小数、负数或包含常见符号的数字表达式
    numeric_pattern = r"^[0-9\-\+\.,\s]+$"
    return bool(re.match(numeric_pattern, text))


def _is_english_mixed(text: str) -> bool:
    """检查文本是否为纯英文、纯数字或英文与数字混合."""
    # 检查是否只包含英文字母、数字和空格
    english_mixed_pattern = r'^[a-zA-Z0-9\s\-\+\.,:;!?()"\']*$'
    return bool(re.match(english_mixed_pattern, text))


def _is_chinese_mixed(text: str) -> bool:
    """检查文本是否为纯中文、纯数字或中文与数字混合."""
    # 检查是否只包含中文字符、数字和空格
    chinese_mixed_pattern = r'^[\u4e00-\u9fff0-9\s\-\+\.,:;!?()"\']*$'
    return bool(re.match(chinese_mixed_pattern, text))
