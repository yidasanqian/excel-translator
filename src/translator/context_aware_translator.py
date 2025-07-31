"""上下文感知的Excel翻译服务."""

import pandas as pd
from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
from openai import AsyncOpenAI
from config.settings import settings
from config.logging_config import get_logger
from translator.data_type_config import DataTypeConfig
from tqdm.asyncio import tqdm_asyncio


logger = get_logger(__name__)


@dataclass
class TranslationContext:
    """翻译上下文信息."""

    row_context: Dict[str, str]
    col_context: Dict[str, str]
    sheet_context: Dict[str, Any]
    domain: str = "general"


@dataclass
class TranslationUnit:
    """翻译单元."""

    original_text: str
    context: TranslationContext
    row_id: int
    col_name: str
    cache_key: str


class TableStructureAnalyzer:
    """表格结构分析器."""

    def analyze_table_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析表格结构."""
        structure = {
            "columns": list(df.columns),
            "row_count": len(df),
            "col_count": len(df.columns),
            "data_types": self._infer_data_types(df),
            "domain": self._detect_domain(df),
            "patterns": self._extract_patterns(df),
        }
        return structure

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_type_config = DataTypeConfig()

    def _infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """推断列的数据类型."""
        return self.data_type_config.infer_data_types(df)

    def _detect_domain(self, df: pd.DataFrame) -> str:
        """检测专业领域."""
        all_text = " ".join(df.astype(str).values.flatten())
        domain_keywords = {
            "mechanical": ["发动机", "液压", "制动", "转向", "机械", "故障"],
            "electrical": ["电路", "电压", "电流", "电池", "电机", "电控"],
            "software": ["程序", "系统", "软件", "代码", "bug", "错误"],
            "medical": ["症状", "诊断", "治疗", "药物", "手术", "患者"],
        }
        for domain, keywords in domain_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                return domain
        else:
            return "general"

    def _extract_patterns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """提取数据模式."""
        patterns = defaultdict(list)
        for col in df.columns:
            column_data = df[col]
            if isinstance(column_data, pd.DataFrame):
                column_data = column_data.iloc[:, 0]
            values = column_data.dropna().astype(str)
            if len(values) > 0:
                common_values = values.value_counts().head(3).index.tolist()
                patterns[col] = common_values
        return dict(patterns)


class TerminologyManager:
    """术语管理器."""

    def __init__(self):
        self.terminology_cache = {}
        self.domain_terms = {
            "mechanical": {
                "发动机": "engine",
                "液压系统": "hydraulic system",
                "制动系统": "braking system",
                "转向系统": "steering system",
                "故障": "malfunction",
                "异常": "abnormal",
                "检查": "inspect",
                "更换": "replace",
            },
            "electrical": {
                "电路": "circuit",
                "电压": "voltage",
                "电流": "current",
                "电池": "battery",
                "电机": "motor",
                "短路": "short circuit",
                "断路": "open circuit",
            },
        }

    def get_term_translation(self, text: str, domain: str) -> Optional[str]:
        """获取术语翻译."""
        if not text or not text.strip():
            return None
        text_clean = str(text).strip()
        if all(ord(c) < 128 for c in text_clean):
            return text_clean
        if domain in self.domain_terms and text_clean in self.domain_terms[domain]:
            return self.domain_terms[domain][text_clean]
        return None

    def get_relevant_terms(self, text: str, domain: str) -> Dict[str, str]:
        """获取与文本相关的术语翻译映射."""
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
        """添加新术语."""
        if domain not in self.domain_terms:
            self.domain_terms[domain] = {}
        self.domain_terms[domain][original] = translated


class SmartBatcher:
    """智能分批器."""

    def __init__(self, max_batch_size: int = 10):
        self.max_batch_size = max_batch_size

    def create_batches(self, df: pd.DataFrame) -> List[List[TranslationUnit]]:
        """创建智能翻译批次."""
        analyzer = TableStructureAnalyzer()
        structure = analyzer.analyze_table_structure(df)
        batches = []
        current_batch = []
        for idx, row in df.iterrows():
            for col_name in df.columns:
                original_value = row[col_name]
                if pd.isna(original_value) or str(original_value).strip() == "":
                    continue
                text = str(original_value).strip()
                if all(ord(c) < 128 for c in text):
                    continue
                context = self._build_context(df, idx, col_name, structure)
                unit = TranslationUnit(
                    original_text=text,
                    context=context,
                    row_id=idx,
                    col_name=col_name,
                    cache_key=self._generate_cache_key(text, context),
                )
                current_batch.append(unit)
                if len(current_batch) >= self.max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
        if current_batch:
            batches.append(current_batch)
        return batches

    def _build_context(
        self, df: pd.DataFrame, row_idx: int, col_name: str, structure: Dict[str, Any]
    ) -> TranslationContext:
        """构建翻译上下文."""
        row_data = df.iloc[row_idx].to_dict()
        col_type = structure["data_types"].get(col_name, "text")
        return TranslationContext(
            row_context=row_data,
            col_context={
                "column_name": col_name,
                "data_type": col_type,
                "common_values": structure["patterns"].get(col_name, []),
            },
            sheet_context={
                "domain": structure["domain"],
                "columns": structure["columns"],
                "total_rows": structure["row_count"],
            },
            domain=structure["domain"],
        )

    def _generate_cache_key(self, text: str, context: TranslationContext) -> str:
        """生成缓存键."""
        key_data = {
            "text": text,
            "domain": context.domain,
            "col_type": context.col_context.get("data_type", "text"),
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


class ContextAwareTranslator:
    """上下文感知翻译器."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url if settings.openai_base_url else None,
        )
        self.model = settings.openai_model
        self.target_language = settings.target_language
        self.timeout = settings.request_timeout
        self.analyzer = TableStructureAnalyzer()
        self.terminology = TerminologyManager()
        self.batcher = SmartBatcher()
        self.translation_cache = {}
        self.context_cache = {}

    async def translate_dataframe(
        self, df: pd.DataFrame, target_lang: str = None
    ) -> pd.DataFrame:
        """翻译整个DataFrame."""
        target = target_lang or self.target_language
        structure = self.analyzer.analyze_table_structure(df)
        logger.info(f"Detected domain: {structure['domain']}")
        translated_columns = []
        column_name_mapping = {}
        for col_name in df.columns:
            if pd.isna(col_name) or col_name is None:
                translated_columns.append(col_name)
                column_name_mapping[col_name] = col_name
            else:
                col_name_str = str(col_name)
                # 处理空列名或"Unnamed"列名
                if not col_name_str.strip() or col_name_str.startswith("Unnamed:"):
                    translated_columns.append("")
                    column_name_mapping[col_name] = ""
                elif col_name_str.strip() and (
                    not all(ord(c) < 128 for c in col_name_str)
                ):
                    translated_col = await self._translate_single_text(
                        col_name_str, target, structure["domain"]
                    )
                    translated_columns.append(translated_col)
                    column_name_mapping[col_name] = translated_col
                else:
                    translated_columns.append(col_name_str)
                    column_name_mapping[col_name] = col_name_str
        batches = self.batcher.create_batches(df)
        logger.info(
            f"Total {len(df)} units, created {len(batches)} translation batches, each with up to {self.batcher.max_batch_size} units."
        )
        translated_df = df.copy()
        translated_df.columns = translated_columns
        translation_map = {}
        for batch in tqdm_asyncio(batches, desc="翻译中", unit="批次"):
            translated_batch = await self._translate_batch(
                batch, target, structure["domain"]
            )
            for unit, translated_text in zip(batch, translated_batch):
                translation_map[unit.row_id, unit.col_name] = translated_text
        for (row_id, col_name), translated_text in translation_map.items():
            translated_col_name = column_name_mapping[col_name]
            translated_df.iloc[
                row_id, translated_df.columns.get_loc(translated_col_name)
            ] = translated_text
        translated_df = translated_df.fillna("")
        return translated_df

    async def _translate_batch(
        self, batch: List[TranslationUnit], target_lang: str, domain: str
    ) -> List[str]:
        """翻译批次."""
        tasks = []
        for unit in batch:
            if unit.cache_key in self.translation_cache:
                tasks.append(
                    asyncio.create_task(self._get_cached_translation(unit.cache_key))
                )
            else:
                term_translation = self.terminology.get_term_translation(
                    unit.original_text, domain
                )
                if term_translation:
                    self.translation_cache[unit.cache_key] = term_translation
                    tasks.append(
                        asyncio.create_task(
                            self._get_cached_translation(unit.cache_key)
                        )
                    )
                else:
                    tasks.append(
                        asyncio.create_task(
                            self._translate_with_context(unit, target_lang)
                        )
                    )
        results = await asyncio.gather(*tasks)
        for unit, translated in zip(batch, results):
            self.translation_cache[unit.cache_key] = translated
        return results

    async def _get_cached_translation(self, cache_key: str) -> str:
        """获取缓存的翻译."""
        return self.translation_cache[cache_key]

    async def _translate_with_context(
        self, unit: TranslationUnit, target_lang: str
    ) -> str:
        """使用上下文进行翻译."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                context_prompt = self._build_context_prompt(unit, target_lang)
                logger.debug(
                    f"Translating text with context: '{unit.original_text}' (attempt {attempt + 1})"
                )
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator specializing in technical documents. Translate accurately while maintaining consistency with the context provided.",
                        },
                        {"role": "user", "content": context_prompt},
                    ],
                    max_tokens=500,
                    timeout=self.timeout,
                )
                translated = response.choices[0].message.content.strip()
                logger.debug(f"Translation result: '{translated}'")
                return translated
            except Exception as e:
                logger.warning(
                    f"Context translation attempt {attempt + 1} failed for text '{unit.original_text}' in column '{unit.col_name}' (row {unit.row_id}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Context translation failed for text '{unit.original_text}' in column '{unit.col_name}' (row {unit.row_id}) after {max_retries} attempts: {str(e)}"
                    )
                    logger.error(f"Context: {context_prompt}")
                    logger.error(f"Domain: {unit.context.domain}")
                    return unit.original_text
                else:
                    await asyncio.sleep(1)
        return None

    def _build_context_prompt(self, unit: TranslationUnit, target_lang: str) -> str:
        """构建上下文翻译提示."""
        context = unit.context
        row_context_str = ", ".join(
            [
                f"{k}: {v}"
                for k, v in context.row_context.items()
                if str(v).strip() and str(v) != unit.original_text
            ]
        )
        col_name = context.col_context.get("column_name", "")
        col_type = context.col_context.get("data_type", "text")
        domain = context.domain
        relevant_terms = self.terminology.get_relevant_terms(unit.original_text, domain)
        terms_prompt = ""
        if relevant_terms:
            terms_str = "\n".join(
                [f"- {cn}: {en}" for cn, en in relevant_terms.items()]
            )
            terms_prompt = f"\n\nRelevant terminology:\n{terms_str}"
        prompt = f"Translate the following Chinese text to complete {target_lang}. \nDo not mix languages.\nContext:\n- Column: {col_name} ({col_type})\n- Row context: {row_context_str}\n- Domain: {domain}{terms_prompt}\nChinese text: {unit.original_text}\nRequirements:\n1. Translate the entire text to {target_lang}, do not leave any Chinese characters\n2. Use the provided terminology consistently\n3. Maintain the original meaning and context\n4. Return only the translated text without any explanations\n5. Ensure complete translation, no partial translation allowed"
        return prompt

    def clear_cache(self):
        """清除缓存."""
        self.translation_cache.clear()
        self.context_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计."""
        return {
            "translation_cache_size": len(self.translation_cache),
            "context_cache_size": len(self.context_cache),
        }

    async def _translate_single_text(
        self, text: str, target_lang: str, domain: str
    ) -> str:
        """翻译单个文本."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                term_translation = self.terminology.get_term_translation(text, domain)
                if term_translation:
                    logger.debug(
                        f"Term translation found: '{text}' -> '{term_translation}'"
                    )
                    return term_translation
                prompt = f"Translate the following Chinese text to complete {target_lang}.\nDo not mix languages.\nContext:\n- Domain: {domain}\nChinese text: {text}\nRequirements:\n1. Translate the entire text to {target_lang}, do not leave any Chinese characters\n2. Maintain the original meaning and context\n3. Return only the translated text without any explanations\n4. Ensure complete translation, no partial translation allowed"
                logger.debug(
                    f"Translating single text: '{text}' in domain '{domain}' (attempt {attempt + 1})"
                )
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator specializing in technical documents. Translate accurately while maintaining consistency with the context provided.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    timeout=self.timeout,
                )
                translated = response.choices[0].message.content.strip()
                logger.debug(f"Single text translation result: '{translated}'")
                return translated
            except Exception as e:
                logger.warning(
                    f"Single text translation attempt {attempt + 1} failed for text '{text}' in domain '{domain}': {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Single text translation failed for text '{text}' in domain '{domain}' after {max_retries} attempts: {str(e)}"
                    )
                    return text
                else:
                    await asyncio.sleep(1)
        return None
