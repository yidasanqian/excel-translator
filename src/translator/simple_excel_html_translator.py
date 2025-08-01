from translator.excel2html import excel_to_html_with_format
from translator.html2excel import html_to_excel_with_format
from translator.translation_filter import needs_translation
from openai import AsyncOpenAI
from config.settings import settings
from config.logging_config import get_logger
import os
from bs4 import BeautifulSoup
import json
import asyncio
import tiktoken
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import re

logger = get_logger(__name__)


@dataclass
class TranslationBatch:
    """翻译批次数据结构."""

    # 批次中的文本单元
    texts: List[str]

    # 批次的位置信息 (索引)
    positions: List[int]

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


class SimpleExcelHTMLTranslator:
    """简化版的Excel HTML翻译器，只负责Excel到HTML的转换和翻译."""

    def __init__(
        self,
        model: str = None,
        target_language: str = None,
        max_tokens: int = None,
        token_buffer: int = 1000,
    ):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url if settings.openai_base_url else None,
        )
        self.model = model or settings.openai_model
        self.target_language = target_language or settings.target_language
        self.timeout = settings.request_timeout
        self.max_tokens = max_tokens or 8192
        self.token_buffer = token_buffer

        # 初始化组件
        self.token_manager = TokenManager(self.model)

    def _detect_domain(self, texts: List[str]) -> str:
        """检测专业领域."""
        all_text = " ".join(texts)
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

    def create_translation_batches(
        self,
        texts: List[str],
        positions: List[int],
        domain: str,
    ) -> List[TranslationBatch]:
        """
        创建适合模型上下文限制的翻译批次.

        Args:
            texts: 要翻译的文本列表
            positions: 文本的位置信息
            domain: 领域信息

        Returns:
            翻译批次列表
        """
        max_tokens = self.max_tokens - self.token_buffer

        # 构建上下文信息
        context_info = {"domain": domain}

        # 计算上下文token数量
        context_str = json.dumps(context_info, ensure_ascii=False)
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
                        context_info=context_info,
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
                    context_info=context_info,
                )
            )

        return batches

    def _build_batch_translation_prompt(
        self,
        batch_id: int,
        total_batches: int,
        batch: TranslationBatch,
        target_lang: str,
    ) -> str:
        """构建批量翻译提示."""
        # 构建上下文描述
        context_str = f"Domain: {batch.context_info.get('domain', 'general')}\n"
        batch_info_str = f"Batch: {batch_id}/{total_batches}\n"

        # 构建待翻译文本列表
        texts_list_str = "Texts to translate:\n"
        for i, text in enumerate(batch.texts, 1):
            texts_list_str += f"{i}. {text}\n"

        prompt = f"""Translate the following texts to {target_lang}.
Do not mix languages.

Context information:
{context_str}
{batch_info_str}

Translation requirements:
1. Translate all texts to {target_lang}, do not leave any source language characters
2. Maintain consistency with the context provided
3. Return translations in the same order as the original texts
4. Return only the translated texts without any explanations
5. Ensure complete translation, no partial translation allowed
6. Each translation should be on a separate line

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

        # 过滤空行并处理可能的序号
        translations = []
        for line in lines:
            line = line.strip()
            if line:
                # 移除行首的序号（如 "1. " 或 "94. "）
                # 使用正则表达式匹配并移除序号

                # 匹配以数字和点开头的序号，例如 "1. ", "94. ", "123. "
                line = re.sub(r"^\d+\.\s*", "", line)
                translations.append(line)

        # 如果结果数量不匹配，尝试其他解析方法
        if len(translations) != expected_count:
            # 如果只有一行，按句号分割
            if len(lines) == 1:
                translations = [t.strip() for t in result.split(".") if t.strip()]
                # 对每个分割后的翻译也移除可能的序号

                translations = [re.sub(r"^\d+\.\s*", "", t) for t in translations]

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

    async def translate_batch_with_context(
        self,
        batch_id: int,
        total_batches: int,
        batch: TranslationBatch,
        target_lang: str,
    ) -> List[str]:
        """
        使用上下文进行批量翻译.

        Args:
            batch_id: 批次ID
            total_batches: 总批次数
            batch: 翻译批次
            target_lang: 目标语言

        Returns:
            翻译结果列表
        """
        if not batch.texts:
            return []

        # 构建翻译提示
        prompt = self._build_batch_translation_prompt(
            batch_id, total_batches, batch, target_lang
        )

        # 调用API进行翻译
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Translating batch with {len(batch.texts)} texts (attempt {attempt + 1})"
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
                    max_tokens=min(4096, self.max_tokens // 2),
                    timeout=self.timeout,
                )

                translated_content = response.choices[0].message.content.strip()
                translations = self._parse_batch_translation_result(
                    translated_content, len(batch.texts)
                )

                logger.debug(
                    f"Batch translation completed with {len(translations)} results"
                )
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
                        batch.texts, target_lang
                    )
                else:
                    await asyncio.sleep(1)

        return batch.texts  # 如果所有重试都失败，返回原文本

    async def _fallback_to_individual_translation(
        self, texts: List[str], target_lang: str
    ) -> List[str]:
        """回退到逐个翻译."""
        logger.info("Falling back to individual translation")
        translations = []
        for text in texts:
            try:
                prompt = f"Translate the following text to {target_lang}. Do not mix languages.\n\n{text}"
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
                    timeout=self.timeout,
                )
                translated = response.choices[0].message.content.strip()
                translations.append(translated)
            except Exception as e:
                logger.error(
                    f"Individual translation failed for text '{text[:50]}...': {str(e)}"
                )
                translations.append(text)  # 如果翻译失败，返回原文本
        return translations

    async def translate_html_content(self, html_content: str, target_lang: str) -> str:
        """翻译HTML内容."""
        try:
            # 使用BeautifulSoup解析HTML以提取文本内容进行翻译
            soup = BeautifulSoup(html_content, "html.parser")

            # 提取需要翻译的文本
            text_elements = []
            original_texts = []

            # 优先提取表格中的文本内容
            table_cells = soup.find_all(["td", "th"])
            for cell in table_cells:
                text = cell.get_text().strip()
                if text and needs_translation(text, target_lang):
                    text_elements.append(cell)
                    original_texts.append(text)

            # 如果没有找到表格文本，则提取其他重要文本
            if not original_texts:
                # 提取标题和段落文本
                for element in soup.find_all(
                    ["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"]
                ):
                    text = element.get_text().strip()
                    if text and needs_translation(text, target_lang):
                        text_elements.append(element)
                        original_texts.append(text)

            # 最后提取所有其他文本作为备选
            if not original_texts:
                # 提取所有需要翻译的文本节点
                for element in soup.find_all(string=True):
                    parent = element.parent
                    # 跳过脚本和样式标签中的文本
                    if parent.name in ["script", "style"]:
                        continue
                    text = element.strip()
                    if text and needs_translation(text, target_lang):
                        text_elements.append(parent)
                        original_texts.append(text)

            if not original_texts:
                logger.info("No text found to translate")
                return html_content

            # 检测领域
            domain = self._detect_domain(original_texts)
            logger.info(f"Detected domain: {domain}")

            # 创建位置信息
            positions = list(range(len(original_texts)))

            # 创建翻译批次
            batches = self.create_translation_batches(original_texts, positions, domain)
            logger.info(f"Created {len(batches)} translation batches")

            # 翻译所有批次
            translated_texts = []
            total_batches = len(batches)

            # 使用 tqdm 进度条显示翻译进度
            batch_tasks = [
                self.translate_batch_with_context(
                    i + 1, total_batches, batch, target_lang
                )
                for i, batch in enumerate(batches)
            ]

            # 使用 tqdm_asyncio.gather 来显示进度条
            batch_results = await tqdm_asyncio.gather(
                *batch_tasks, desc="翻译进度", unit="批次"
            )

            # 处理翻译结果
            for batch_translations in batch_results:
                translated_texts.extend(batch_translations)

            # 将翻译后的文本替换回HTML
            for i, element in enumerate(text_elements):
                if i < len(translated_texts):
                    # 如果是表格单元格，直接替换文本内容
                    if element.name in ["td", "th"]:
                        element.string = translated_texts[i]
                    # 如果是其他HTML元素，更新其文本内容
                    else:
                        element.string = translated_texts[i]

            return str(soup)

        except Exception as e:
            logger.error(f"HTML translation failed: {str(e)}")
            return html_content

    async def translate_excel_to_excel(
        self, input_excel: str, output_excel: str = None, target_lang: str = None
    ) -> str:
        """
        将Excel文件翻译为另一种语言的Excel文件.

        Args:
            input_excel: 输入Excel文件路径
            output_excel: 输出Excel文件路径
            target_lang: 目标语言

        Returns:
            翻译后的Excel文件路径
        """
        target = target_lang or self.target_language
        # 确保输出目录存在
        output_path = "output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        base_name = os.path.splitext(os.path.basename(input_excel))[0]
        if output_excel is None:
            output_excel = os.path.join(output_path, f"{base_name}_translated.xlsx")

        logger.info(f"Starting translation for Excel file: {input_excel}")

        # 步骤1: Excel转HTML
        base_name = os.path.splitext(os.path.basename(input_excel))[0]
        temp_html = os.path.join(output_path, f"{base_name}_temp.html")
        metadata_file = os.path.join(output_path, f"{base_name}_metadata.json")

        logger.info("Step 1: Converting Excel to HTML...")
        _ = excel_to_html_with_format(input_excel, temp_html, metadata_file)
        logger.info("✓ Excel to HTML conversion completed")

        # 步骤2: 读取HTML内容并翻译
        logger.info("Step 2: Translating HTML content...")
        with open(temp_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        translated_html_content = await self.translate_html_content(
            html_content, target
        )

        # 保存翻译后的HTML
        translated_html = os.path.join(output_path, f"{base_name}_translated.html")
        with open(translated_html, "w", encoding="utf-8") as f:
            f.write(translated_html_content)
        logger.info("✓ HTML translation completed")

        # 步骤3: HTML转回Excel
        logger.info("Step 3: Converting HTML back to Excel...")
        html_to_excel_with_format(translated_html, output_excel, metadata_file)
        logger.info("✓ HTML to Excel conversion completed")

        logger.info(f"Translation completed successfully. Output file: {output_excel}")
        return output_excel


# 保持向后兼容的函数接口
async def simple_excel_html_translation_workflow(
    input_excel, output_excel=None, target_language=None
):
    """
    简化版的Excel到Excel翻译工作流程
    """
    # 确保输出目录存在
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    base_name = os.path.splitext(os.path.basename(input_excel))[0]
    if output_excel is None:
        output_excel = os.path.join(output_path, f"{base_name}_translated.xlsx")

    # 使用简化版的翻译器
    translator = SimpleExcelHTMLTranslator(target_language=target_language)
    result_file = await translator.translate_excel_to_excel(input_excel, output_excel)

    print(f"Translation completed. File saved as: {result_file}")
    return result_file
