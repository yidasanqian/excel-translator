"""术语管理器 - 管理和提供术语翻译服务."""

import json
from typing import Dict, Optional
from config.logging_config import get_logger

logger = get_logger(__name__)


class TerminologyManager:
    """术语管理器."""

    def __init__(self, domain_terms: Dict[str, Dict[str, str]]):
        """
        初始化术语管理器.

        Args:
            custom_terms: 自定义术语字典，格式为 {domain: {term: translation}}
        """
        self.terminology_cache = {}
        if domain_terms is None:
            domain_terms = {}
        self.domain_terms = domain_terms

    def get_term_translation(self, text: str, domain: str) -> Optional[str]:
        """
        获取术语翻译.

        Args:
            text: 原始文本
            domain: 领域

        Returns:
            翻译后的文本，如果未找到则返回 None
        """
        if not text or not text.strip():
            return None
        text_clean = str(text).strip()
        if all(ord(c) < 128 for c in text_clean):
            return text_clean
        if domain in self.domain_terms and text_clean in self.domain_terms[domain]:
            return self.domain_terms[domain][text_clean]
        return None

    def get_relevant_terms(self, text: str, domain: str) -> Dict[str, str]:
        """
        获取与文本相关的术语翻译映射.

        Args:
            text: 原始文本
            domain: 领域

        Returns:
            相关术语的字典映射
        """
        if not text or not text.strip():
            return {}
        text_clean = str(text).strip()
        relevant_terms = {}
        if all(ord(c) < 128 for c in text_clean):
            return {}
        if domain in self.domain_terms:
            for cn_term, en_term in self.domain_terms[domain].items():
                if cn_term in text_clean:
                    relevant_terms[cn_term] = en_term
        return relevant_terms

    def add_term(self, original: str, translated: str, domain: str):
        """
        添加新术语.

        Args:
            original: 原始术语
            translated: 翻译后的术语
            domain: 领域
        """
        if domain not in self.domain_terms:
            self.domain_terms[domain] = {}
        self.domain_terms[domain][original] = translated

    def add_domain_terms(self, domain: str, terms: Dict[str, str]):
        """
        添加整个领域的术语.

        Args:
            domain: 领域
            terms: 术语字典
        """
        if domain not in self.domain_terms:
            self.domain_terms[domain] = {}
        self.domain_terms[domain].update(terms)

    def get_current_domain(self) -> str:
        """
        获取当前领域.

        Returns:
            当前领域
        """
        return self.domain_terms.keys()[0] if self.domain_terms else "general"

    def get_domain_terms(self, domain: str) -> Dict[str, str]:
        """
        获取特定领域的所有术语.

        Args:
            domain: 领域

        Returns:
            术语字典
        """
        return self.domain_terms.get(domain, {})

    def get_all_terms(self) -> Dict[str, Dict[str, str]]:
        """
        获取所有领域的术语.

        Returns:
            所有术语的字典
        """
        return self.domain_terms.copy()

    def update_terms_from_dict(self, terms_dict: Dict[str, Dict[str, str]]):
        """
        从字典更新术语.

        Args:
            terms_dict: 术语字典，格式为 {domain: {term: translation}}
        """
        for domain, terms in terms_dict.items():
            if domain not in self.domain_terms:
                self.domain_terms[domain] = {}
            self.domain_terms[domain].update(terms)

    def update_terms_from_json(self, json_str: str):
        """
        从JSON字符串更新术语.

        Args:
            json_str: JSON格式的术语字符串
        """
        try:
            terms_dict = json.loads(json_str)
            self.update_terms_from_dict(terms_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON terms: {e}")
            raise

    def clear_cache(self):
        """清除缓存."""
        self.terminology_cache.clear()
