"""批量翻译器 - 支持上下文感知的批量Excel翻译."""

import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import json
from dataclasses import dataclass
from collections import defaultdict
import tiktoken
from openai import AsyncOpenAI
from config.settings import settings
from config.logging_config import get_logger
from translator.translation_filter import needs_translation
from translator.terminology_manager import TerminologyManager

logger = get_logger(__name__)


@dataclass
class TranslationBatch:
    """翻译批次数据结构."""

    # 批次中的文本单元
    texts: List[str]

    # 批次的位置信息 (行索引, 列名)
    positions: List[Tuple[int, str]]

    # 批次的token数量
    token_count: int

    # 批次的上下文信息
    context_info: Dict[str, Any]


class TokenManager:
    """Token管理器，负责计算和管理token数量."""

    def __init__(self, model):
        if "/" in model:
            model = model.split("/")[-1]  # 处理可能的路径格式
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.cache = {}
        self.max_cache_size = 10000

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量."""
        if text in self.cache:
            return self.cache[text]

        token_count = len(self.encoding.encode(text))

        # 简单的缓存管理
        if len(self.cache) < self.max_cache_size:
            self.cache[text] = token_count

        return token_count

    def count_tokens_in_list(self, texts: List[str]) -> List[int]:
        """计算文本列表中每个文本的token数量."""
        return [self.count_tokens(text) for text in texts]

    def count_total_tokens(self, texts: List[str]) -> int:
        """计算文本列表的总token数量."""
        return sum(self.count_tokens_in_list(texts))


class ContextAwareBatchContextBuilder:
    """批量翻译上下文构建器."""

    def __init__(self):
        return

    def build_table_context(
        self,
        df: pd.DataFrame,
        domain: str,
        data_types: Dict[str, str],
        patterns: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """构建表格级上下文."""
        return {
            "domain": domain,
            "columns": list(df.columns),
            "data_types": data_types,
            "patterns": patterns,
        }

    def build_batch_context(
        self,
        batch_id: int,
        total_batches: int,
        positions: List[Tuple[int, str]],
        original_texts: List[str],
    ) -> Dict[str, Any]:
        """构建批次级上下文."""
        position_info = []
        for i, (row, col) in enumerate(positions):
            position_info.append(
                {"row": row, "column": col, "original_text": original_texts[i]}
            )

        return {
            "batch_id": batch_id,
            "total_batches": total_batches,
            "position_info": position_info,
        }


class ContextAwareBatchTranslator:
    """批量翻译器，支持上下文感知的批量Excel翻译."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        token_buffer: int = 1000,
    ):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.token_buffer = token_buffer

        # 初始化组件
        self.token_manager = TokenManager(self.model)
        self.context_builder = ContextAwareBatchContextBuilder()

        # 统计信息
        self.stats = {"batches_processed": 0, "texts_translated": 0, "api_calls": 0}

    async def translate_dataframe(
        self,
        df: pd.DataFrame,
        source_lang: str,
        target_lang: str,
        domain_terms: Optional[Dict[str, Dict[str, str]]] = None,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        批量翻译整个DataFrame.

        Args:
            df: 要翻译的DataFrame
            source_lang: 源语言
            target_lang: 目标语言
            domain_terms: 领域术语字典，格式为 {domain: {term: translation}}
            progress_callback: 进度回调函数，用于报告翻译进度

        Returns:
            翻译后的DataFrame
        """
        logger.info(
            f"Starting batch translation for DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )

        # 实例化术语管理器
        self.terminology_manager = TerminologyManager(domain_terms)

        # 处理列名
        translated_columns = await self._translate_column_names(
            df, source_lang, target_lang
        )

        # 分析表格结构
        structure = self._analyze_table_structure(df)
        domain = structure["domain"]
        data_types = structure["data_types"]
        patterns = structure["patterns"]

        # 构建表格上下文
        table_context = self.context_builder.build_table_context(
            df, domain, data_types, patterns
        )

        # 提取需要翻译的文本
        texts, positions = self._extract_texts_and_positions(df, target_lang, domain)
        logger.info(f"Extracted {len(texts)} texts for translation")

        if not texts:
            result_df = df.copy()
            result_df.columns = translated_columns
            return result_df

        # 创建翻译批次
        batches = self.create_translation_batches(texts, positions, table_context)
        logger.info(f"Created {len(batches)} translation batches")

        # 翻译所有批次
        translated_texts = []
        total_batches = len(batches)

        # 真正逐个执行批次翻译任务，确保进度事件实时发送
        for i, batch in enumerate(batches):
            try:
                # 报告开始翻译批次的进度（如果有进度回调函数）
                if progress_callback:
                    progress = (i / total_batches) * 100
                    await progress_callback(
                        progress, f"正在翻译批次 {i + 1}/{total_batches}"
                    )

                # 执行批次翻译（不预创建任务，直接调用）
                batch_result = await self.translate_batch_with_context(
                    i + 1, total_batches, batch, table_context, source_lang, target_lang
                )
                translated_texts.extend(batch_result)
                self.stats["batches_processed"] += 1

                # 报告完成翻译批次的进度（如果有进度回调函数）
                if progress_callback:
                    progress = ((i + 1) / total_batches) * 100
                    await progress_callback(
                        progress, f"已完成批次 {i + 1}/{total_batches}"
                    )

            except Exception as e:
                logger.error(f"Batch {i + 1} translation failed: {str(e)}")
                # 即使某个批次失败，也继续处理其他批次
                if progress_callback:
                    progress = ((i + 1) / total_batches) * 100
                    await progress_callback(
                        progress, f"批次 {i + 1} 翻译失败: {str(e)}"
                    )

        # 应用翻译结果到DataFrame
        translated_df = self._apply_translations_to_dataframe(
            df, translated_texts, positions
        )

        # 设置翻译后的列名
        translated_df.columns = translated_columns

        # 处理可能由pandas自动生成的"Unnamed:*"列名
        final_columns = []
        for col in translated_df.columns:
            if isinstance(col, str) and col.startswith("Unnamed:"):
                final_columns.append("")
            else:
                final_columns.append(col)
        translated_df.columns = final_columns

        # 确保所有未翻译的单元格保留其原始值，而不是变成NaN
        translated_df = translated_df.fillna(df)

        logger.info("Batch translation completed successfully")
        return translated_df

    def _analyze_table_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析表格结构."""
        # 检测专业领域
        domain = self.terminology_manager.get_current_domain()

        # 推断数据类型
        data_types = self._infer_data_types(df)

        # 提取数据模式
        patterns = self._extract_patterns(df)

        return {"domain": domain, "data_types": data_types, "patterns": patterns}

    def _infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """推断列的数据类型."""
        types = {}
        for col in df.columns:
            column_data = df[col]
            if isinstance(column_data, pd.DataFrame):
                column_data = column_data.iloc[:, 0]
            values = column_data.dropna().astype(str)
            if values.empty:
                types[col] = "empty"
                continue
            try:
                values.astype(float)
                types[col] = "numeric"
                continue
            except ValueError:
                pass
            types[col] = "text"
        return types

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

    def _extract_texts_and_positions(
        self, df: pd.DataFrame, target_language: str = None, domain: str = "general"
    ) -> Tuple[List[str], List[Tuple[int, str]]]:
        """提取需要翻译的文本和位置信息."""
        texts = []
        positions = []

        for idx, row in df.iterrows():
            for col_name in df.columns:
                original_value = row[col_name]
                if pd.isna(original_value) or str(original_value).strip() == "":
                    continue
                text = str(original_value).strip()
                # 检查是否需要翻译
                if not needs_translation(text, target_language):
                    continue
                # 检查是否有术语翻译
                term_translation = self.terminology_manager.get_term_translation(
                    text, domain
                )
                if term_translation:
                    # 如果有术语翻译，直接使用术语翻译结果
                    texts.append(term_translation)
                else:
                    texts.append(text)
                positions.append((idx, col_name))
        # 添加调试信息
        logger.info(f"Extracted {len(texts)} texts for translation")
        if len(texts) > 0:
            logger.info(f"First few extracted texts: {texts[:5]}")

        return texts, positions

    async def _translate_single_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """翻译单个文本."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"Translate the following {source_lang} text to {target_lang}. Do not mix languages.\n\n{text}"
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator. Translate accurately.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                )
                translated = response.choices[0].message.content.strip()
                return translated
            except Exception as e:
                logger.warning(
                    f"Single text translation attempt {attempt + 1} failed for text '{text}': {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Single text translation failed for text '{text}' after {max_retries} attempts: {str(e)}"
                    )
                    return text
                else:
                    await asyncio.sleep(1)
        return text

    async def _translate_column_names(
        self, df: pd.DataFrame, source_lang: str, target_lang: str
    ) -> List[str]:
        """翻译列名."""
        logger.info(
            f"Translating column names for DataFrame with {len(df.columns)} columns"
        )
        translated_columns = []
        for col_name in df.columns:
            if pd.isna(col_name) or col_name is None:
                logger.debug(f"Column name is NaN or None: {col_name}")
                translated_columns.append(col_name)
            else:
                col_name_str = str(col_name)
                # 处理空列名
                if not col_name_str.strip():
                    logger.debug(f"Column name is empty: '{col_name_str}'")
                    translated_columns.append("")
                # 对于"Unnamed"列名，替换为空字符串（优先处理，不受翻译过滤器影响）
                elif col_name_str.startswith("Unnamed:"):
                    logger.debug(
                        f"Column name starts with 'Unnamed:', replacing with empty string: '{col_name_str}'"
                    )
                    translated_columns.append("")
                # 检查列名是否需要翻译
                elif not needs_translation(col_name_str, target_lang):
                    translated_columns.append(col_name_str)
                    continue
                else:
                    logger.debug(f"Processing column name: '{col_name_str}'")
                    logger.debug(f"Translating column name: '{col_name_str}'")
                    # 使用现有的翻译方法翻译列名
                    translated_name = await self._translate_single_text(
                        col_name_str, source_lang, target_lang
                    )
                    logger.debug(
                        f"Translated column name: '{col_name_str}' -> '{translated_name}'"
                    )
                    translated_columns.append(translated_name)
        logger.info(f"Finished translating column names. Result: {translated_columns}")
        return translated_columns

    def create_translation_batches(
        self,
        texts: List[str],
        positions: List[Tuple[int, str]],
        table_context: Dict[str, Any],
    ) -> List[TranslationBatch]:
        """
        创建适合模型上下文限制的翻译批次.

        Args:
            texts: 要翻译的文本列表
            positions: 文本的位置信息
            table_context: 表格上下文信息

        Returns:
            翻译批次列表
        """
        max_tokens = self.max_tokens - self.token_buffer

        # 计算上下文token数量
        context_str = json.dumps(table_context, ensure_ascii=False)
        context_tokens = self.token_manager.count_tokens(context_str)

        batches = []
        current_batch_texts = []
        current_batch_positions = []
        current_token_count = context_tokens

        for text, position in zip(texts, positions):
            text_tokens = self.token_manager.count_tokens(text)
            # 预留一些额外的token空间用于格式化
            total_tokens_with_text = current_token_count + text_tokens + 50

            # 如果添加当前文本会超过token限制，则创建新批次
            if total_tokens_with_text > max_tokens and current_batch_texts:
                batches.append(
                    TranslationBatch(
                        texts=current_batch_texts.copy(),
                        positions=current_batch_positions.copy(),
                        token_count=current_token_count,
                        context_info=table_context,
                    )
                )
                # 重置当前批次
                current_batch_texts = [text]
                current_batch_positions = [position]
                current_token_count = context_tokens + text_tokens + 50
            else:
                # 添加文本到当前批次
                current_batch_texts.append(text)
                current_batch_positions.append(position)
                current_token_count = total_tokens_with_text

        # 添加最后一个批次
        if current_batch_texts:
            batches.append(
                TranslationBatch(
                    texts=current_batch_texts,
                    positions=current_batch_positions,
                    token_count=current_token_count,
                    context_info=table_context,
                )
            )

        return batches

    async def translate_batch_with_context(
        self,
        batch_id: int,
        total_batches: int,
        batch: TranslationBatch,
        table_context: Dict[str, Any],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """
        使用上下文进行批量翻译.

        Args:
            batch: 翻译批次
            table_context: 表格上下文信息
            target_lang: 目标语言

        Returns:
            翻译结果列表
        """
        if not batch.texts:
            return []

        # 构建批次上下文
        batch_context = self.context_builder.build_batch_context(
            batch_id, total_batches, batch.positions, batch.texts
        )

        # 构建翻译提示
        prompt = self._build_batch_translation_prompt(
            table_context, batch_context, batch.texts, source_lang, target_lang
        )

        # 调用API进行翻译
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Translating batch with {len(batch.texts)} texts (attempt {attempt + 1})"
                )
                # 记录API调用开始
                api_start_time = asyncio.get_event_loop().time()
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator specializing in technical documents. Translate accurately while maintaining consistency with the context provided.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=min(4096, self.max_tokens // 2),
                )
                api_end_time = asyncio.get_event_loop().time()
                api_duration = api_end_time - api_start_time
                self.stats["api_calls"] += 1
                logger.debug(
                    f"API call completed in {api_duration:.2f}s with {response.usage.total_tokens if response.usage else 'unknown'} tokens"
                )

                translated_content = response.choices[0].message.content.strip()
                translations = self._parse_batch_translation_result(
                    translated_content, len(batch.texts)
                )

                logger.debug(
                    f"Batch translation completed with {len(translations)} results"
                )
                self.stats["texts_translated"] += len(batch.texts)
                return translations

            except Exception as e:
                logger.warning(
                    f"Batch translation attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Batch translation failed after {max_retries} attempts: {str(e)}"
                    )
                    # 回退到逐个翻译
                    return await self._fallback_to_individual_translation(
                        batch.texts, source_lang, target_lang
                    )
                else:
                    await asyncio.sleep(1)

        return batch.texts  # 如果所有重试都失败，返回原文本

    def _build_batch_translation_prompt(
        self,
        table_context: Dict[str, Any],
        batch_context: Dict[str, Any],
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> str:
        """构建批量翻译提示."""
        domain = table_context.get("domain", "general")

        # 获取领域相关术语
        all_terms = {}
        for text in texts:
            terms = self.terminology_manager.get_relevant_terms(text, domain)
            all_terms.update(terms)

        # 构建表格上下文描述
        table_context_str = f"""Table context:
        - Domain: {domain}
        - Columns: {table_context.get("columns", [])}
        - Data types: {table_context.get("data_types", {})}
        """

        # 构建术语信息
        terms_prompt = ""
        if all_terms:
            terms_str = "\n".join([f"- {cn}: {en}" for cn, en in all_terms.items()])
            terms_prompt = f"\n\nRelevant terminology:\n{terms_str}"

        # 构建批次上下文描述
        batch_context_str = """Batch context:
        - Position information:
        """
        for i, info in enumerate(batch_context.get("position_info", [])):
            batch_context_str += f'  - Row {info["row"]}, Column "{info["column"]}": {info["original_text"]}\n'

        # 构建待翻译文本列表
        texts_list_str = f"{source_lang} texts to translate:\n"
        for i, text in enumerate(texts, 1):
            texts_list_str += f"{i}. {text}\n"

        prompt = f"""Translate the following {source_lang} texts to {target_lang}. 
        Do not mix languages.

        {table_context_str}
        {batch_context_str}
        {terms_prompt}

        Translation requirements:
        1. Translate all texts to {target_lang}, do not leave any {source_lang} characters
        2. Maintain consistency with the context provided
        3. Use the provided terminology consistently
        4. Return translations in the same order as the original texts
        5. Return only the translated texts without any explanations
        6. Ensure complete translation, no partial translation allowed
        7. Each translation should be on a separate line
        8. Do not include the numbering (e.g., "1.", "2.") in the translations
        9. if it is not a valid or complete {source_lang} text, return original text

        {texts_list_str}

        Translated texts in {target_lang}:
        """

        return prompt

    def _parse_batch_translation_result(
        self, result: str, expected_count: int
    ) -> List[str]:
        """解析批量翻译结果."""
        # 如果期望只有一个翻译结果，直接返回整个结果
        if expected_count == 1:
            return [result.strip()]

        # 按行分割结果
        lines = result.strip().split("\n")

        # 过滤空行并去除行首的序号
        translations = []
        for line in lines:
            if line.strip():
                # 去除行首的序号（如 "1. "、"2. " 等）
                cleaned_line = line.strip()
                if ". " in cleaned_line and cleaned_line.split(". ", 1)[0].isdigit():
                    cleaned_line = cleaned_line.split(". ", 1)[1]
                translations.append(cleaned_line)

        # 如果结果数量不匹配，尝试其他解析方法
        if len(translations) != expected_count:
            # 如果只有一行，按句号分割
            if len(lines) == 1:
                translations = [t.strip() for t in result.split(".") if t.strip()]

            # 如果仍然不匹配，返回原文本
            if len(translations) != expected_count:
                logger.warning(
                    f"Translation count mismatch: expected {expected_count}, got {len(translations)}"
                )
                # 返回适当数量的空字符串或原文本
                if len(translations) < expected_count:
                    # 补齐数量
                    translations.extend([""] * (expected_count - len(translations)))
                else:
                    # 截断到期望数量
                    translations = translations[:expected_count]

        return translations

    async def _fallback_to_individual_translation(
        self, texts: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """回退到逐个翻译."""
        logger.info("Falling back to individual translation")
        translations = []
        for text in texts:
            try:
                prompt = f"Translate the following {source_lang} text to {target_lang}. Do not mix languages.\n\n{text}"
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator. Translate accurately.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                )
                translated = response.choices[0].message.content.strip()
                translations.append(translated)
            except Exception as e:
                logger.error(
                    f"Individual translation failed for text '{text[:50]}...': {str(e)}"
                )
                translations.append(text)  # 如果翻译失败，返回原文本
        return translations

    def _apply_translations_to_dataframe(
        self,
        df: pd.DataFrame,
        translations: List[str],
        positions: List[Tuple[int, str]],
    ) -> pd.DataFrame:
        """将翻译结果应用到DataFrame."""
        translated_df = df.copy()

        for translation, (row_idx, col_name) in zip(translations, positions):
            translated_df.iloc[row_idx, df.columns.get_loc(col_name)] = translation

        return translated_df

    def get_stats(self) -> Dict[str, Any]:
        """获取翻译统计信息."""
        return self.stats.copy()
