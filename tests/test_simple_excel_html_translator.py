import unittest
import os
from translator.simple_excel_html_translator import SimpleExcelHTMLTranslator


class TestSimpleExcelHTMLTranslator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        model = "azure-openai/gpt-4o"
        self.translator = SimpleExcelHTMLTranslator(model)
        self.test_file = "docs/案例5.xlsx"
        self.output_path = "output"

        # 确保输出目录存在
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    async def test_translate_excel_to_excel(self):
        """测试简化版的Excel到Excel翻译工作流程"""
        # 检查测试文件是否存在
        if not os.path.exists(self.test_file):
            self.skipTest(f"测试文件 {self.test_file} 不存在，跳过测试")

        # 生成输出文件路径
        base_name = os.path.splitext(os.path.basename(self.test_file))[0]
        output_file = os.path.join(
            self.output_path, f"{base_name}_workflow_translated.xlsx"
        )

        # 执行翻译工作流程
        result_file = await self.translator.translate_excel_to_excel(
            self.test_file, output_file, "chinese", "english"
        )

        # 验证结果
        self.assertEqual(result_file, output_file)
        self.assertTrue(os.path.exists(result_file), "翻译后的文件应该存在")

        print(f"工作流程翻译完成，文件已保存为: {result_file}")


if __name__ == "__main__":
    # 创建测试套件
    suite = unittest.TestSuite()

    # 添加测试用例
    suite.addTest(TestSimpleExcelHTMLTranslator("test_translate_excel_to_excel"))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
