import unittest
import pandas as pd
from translator.context_aware_batch_translator import ContextAwareBatchTranslator


class TestUnnamedColumnHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.translator = ContextAwareBatchTranslator()

    async def test_translate_column_names_with_unnamed_columns(self):
        # 创建一个包含"Unnamed:"列名的DataFrame
        df = pd.DataFrame(
            {
                "Name": ["张三", "李四"],
                "Unnamed: 1": ["年龄", "30"],
                "Unnamed: 2": ["城市", "北京"],
                "Description": ["工程师", "设计师"],
            }
        )

        # 翻译列名
        translated_columns = await self.translator._translate_column_names(
            df, "english"
        )

        # 验证"Unnamed:"列名被替换为空字符串
        expected_columns = [
            "Name",
            "",
            "",
            "Description",
        ]  # 假设Name和Description被翻译为英文
        self.assertEqual(len(translated_columns), len(expected_columns))

        # 验证Unnamed列被替换为空字符串
        self.assertEqual(translated_columns[1], "")
        self.assertEqual(translated_columns[2], "")

        print("Translated columns:", translated_columns)

    async def test_translate_column_names_with_mixed_columns(self):
        # 创建一个包含混合列名的DataFrame
        df = pd.DataFrame(
            {
                "姓名": ["张三", "李四"],
                "Unnamed: 1": ["年龄", "30"],
                "年龄": ["25", "30"],
                "": ["城市", "职业"],
            }
        )

        # 翻译列名
        translated_columns = await self.translator._translate_column_names(
            df, "english"
        )

        # 验证"Unnamed:"列名被替换为空字符串
        self.assertEqual(translated_columns[1], "")

        # 验证空列名仍然为空字符串
        self.assertEqual(translated_columns[3], "")

        print("Translated columns:", translated_columns)


if __name__ == "__main__":
    unittest.main()
