"""集成翻译器 - 将上下文感知翻译集成到现有系统."""

import pandas as pd
from typing import Dict
from translator.context_aware_translator import ContextAwareTranslator
from translator.excel_handler import ExcelHandler
from translator.enhanced_excel_handler import EnhancedExcelHandler
from config.logging_config import get_logger

logger = get_logger(__name__)


class IntegratedTranslator:
    """集成翻译器，提供统一的翻译接口."""

    def __init__(
        self,
        use_context_aware: bool = True,
        preserve_format: bool = True,
        batch_translation_enabled: bool = True,
    ):
        """
        初始化集成翻译器.
        :param use_context_aware: 是否使用上下文感知翻译
        :param preserve_format: 是否保留Excel格式
        :param batch_translation_enabled: 是否启用批量翻译
        """
        self.use_context_aware = use_context_aware
        self.preserve_format = preserve_format
        self.excel_handler = (
            EnhancedExcelHandler() if preserve_format else ExcelHandler()
        )
        if use_context_aware:
            self.translator = ContextAwareTranslator(batch_translation_enabled)
        else:
            from translator.cell_translator import ExcelCellTranslator

            self.translator = ExcelCellTranslator()

    async def translate_excel_file(
        self, file_path: str, output_path: str = None, target_language: str = "english"
    ) -> str:
        """翻译Excel文件."""
        try:
            if self.preserve_format:
                excel_data_with_info = self.excel_handler.read_excel_with_merge_info(
                    file_path
                )
                excel_data = excel_data_with_info["data"]
                merge_info = excel_data_with_info["merge_info"]
                style_info = excel_data_with_info["style_info"]
            else:
                excel_data = self.excel_handler.read_excel(file_path)
                merge_info = None
                style_info = None
            logger.info(f"Reading Excel file: {file_path}")
            translated_data = {}
            for sheet_name, df in excel_data.items():
                logger.info(f"Translating sheet: {sheet_name}")
                if self.use_context_aware:
                    translated_df = await self.translator.translate_dataframe(
                        df, target_language
                    )
                else:
                    texts = self.excel_handler.extract_text_for_translation(
                        {sheet_name: df}
                    )[sheet_name]
                    translations = await self.translator.translate_excel_data(
                        {sheet_name: texts}, target_language
                    )
                    translated_df = self.excel_handler.apply_translations(
                        {sheet_name: df}, translations
                    )[sheet_name]
                translated_data[sheet_name] = translated_df
            if not output_path:
                output_path = file_path.replace(
                    ".xlsx", f"_translated_{target_language}.xlsx"
                )
            else:
                file_name = file_path.split("/")[-1]
                output_path = f"{output_path}/translated_{target_language}_{file_name}"
            if self.preserve_format:
                result_path = self.excel_handler.write_excel_with_merge_info(
                    file_path, translated_data, output_path, merge_info, style_info
                )
            else:
                result_path = self.excel_handler.write_excel(
                    translated_data, output_path
                )
            logger.info(f"Translation completed: {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    async def translate_excel_data(
        self, data: Dict[str, pd.DataFrame], target_language: str = "english"
    ) -> Dict[str, pd.DataFrame]:
        """翻译Excel数据."""
        translated_data = {}
        for sheet_name, df in data.items():
            logger.info(f"Translating sheet: {sheet_name}")
            if self.use_context_aware:
                translated_df = await self.translator.translate_dataframe(
                    df, target_language
                )
            else:
                texts = self.excel_handler.extract_text_for_translation(
                    {sheet_name: df}
                )[sheet_name]
                translations = await self.translator.translate_excel_data(
                    {sheet_name: texts}, target_language
                )
                translated_df = self.excel_handler.apply_translations(
                    {sheet_name: df}, translations
                )[sheet_name]
            translated_data[sheet_name] = translated_df
        return translated_data

    def get_translation_stats(self) -> Dict[str, any]:
        """获取翻译统计信息."""
        if self.use_context_aware:
            return {
                "translator_type": "context_aware",
                "cache_stats": self.translator.get_cache_stats(),
            }
        return {"translator_type": "legacy", "cache_stats": {}}


async def translate_excel_with_context(
    file_path: str,
    output_path: str = None,
    target_language: str = "english",
    preserve_format: bool = True,
) -> str:
    """使用上下文感知翻译翻译Excel文件."""
    translator = IntegratedTranslator(
        use_context_aware=True, preserve_format=preserve_format
    )
    return await translator.translate_excel_file(
        file_path, output_path, target_language
    )


async def translate_excel_without_context(
    file_path: str,
    output_path: str = None,
    target_language: str = "english",
    preserve_format: bool = True,
) -> str:
    """使用传统翻译方法翻译Excel文件."""
    translator = IntegratedTranslator(
        use_context_aware=False, preserve_format=preserve_format
    )
    return await translator.translate_excel_file(
        file_path, output_path, target_language
    )
