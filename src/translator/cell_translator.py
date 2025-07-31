"""OpenAI translation service."""

import asyncio
from typing import Dict, List
from openai import AsyncOpenAI
from config.settings import settings
from config.logging_config import get_logger
from tqdm.asyncio import tqdm_asyncio

logger = get_logger(__name__)


class ExcelCellTranslator:
    """OpenAI-based translation service."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url if settings.openai_base_url else None,
        )
        self.model = settings.openai_model
        self.target_language = settings.target_language
        self.max_batch_size = settings.max_batch_size
        self.timeout = settings.request_timeout

    async def detect_language(self, text: str) -> str:
        """Detect the source language of the text."""
        try:
            prompt = f"Detect the language of this text: '{text[:100]}...'. Return only the language name in English."
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a language detection expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                timeout=self.timeout,
            )
            detected = response.choices[0].message.content.strip().lower()
            return detected
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "chinese"

    async def translate_text(self, text: str, target_lang: str = None) -> str:
        """Translate single text string."""
        if not text or not text.strip():
            return text
        target = target_lang or self.target_language
        try:
            source_lang = await self.detect_language(text)
            if (
                source_lang == "chinese"
                and target == "chinese"
                or (source_lang == "english" and target == "english")
            ):
                return text
            prompt = f"Translate the following text from {source_lang} to {target}:\n\n{text}"
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate accurately and maintain the technical meaning. Return only the translated text without explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                timeout=self.timeout,
            )
            translated = response.choices[0].message.content.strip()
            return translated
        except Exception as e:
            logger.error(f"Translation failed for text: {text[:50]}... Error: {e}")
            return text

    async def translate_batch(
        self, texts: List[str], target_lang: str = None
    ) -> List[str]:
        """Translate a batch of texts efficiently."""
        if not texts:
            return []
        target = target_lang or self.target_language
        batches = [
            texts[i : i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]
        results = []
        for batch in tqdm_asyncio(batches, desc="翻译中", unit="批次"):
            batch_results = await self._translate_batch_async(batch, target)
            results.extend(batch_results)
        return results

    async def _translate_batch_async(
        self, texts: List[str], target_lang: str
    ) -> List[str]:
        """Translate a single batch asynchronously."""
        if not texts:
            return []
        tasks = [self.translate_text(text, target_lang) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch translation error for item {i}: {result}")
                final_results.append(texts[i])
            else:
                final_results.append(result)
        return final_results

    async def translate_excel_data(
        self, data: Dict[str, List[str]], target_lang: str = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Translate Excel data organized by sheets.

        Args:
            data: Dict with sheet names as keys and lists of texts as values
            target_lang: Target language for translation

        Returns:
            Dict with translations organized by sheet and original text
        """
        target = target_lang or self.target_language
        translations = {}
        for sheet_name, texts in data.items():
            if not texts:
                translations[sheet_name] = {}
                continue
            unique_texts = list(set(texts))
            print(f"\n正在翻译工作表: {sheet_name}")
            translated_texts = await self.translate_batch(unique_texts, target)
            translation_map = dict(zip(unique_texts, translated_texts))
            full_translations = {
                text: translation_map.get(text, text) for text in texts
            }
            translations[sheet_name] = full_translations
        return translations

    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "chinese",
            "english",
            "japanese",
            "korean",
            "french",
            "german",
            "spanish",
        ]
