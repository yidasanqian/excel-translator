import unittest

from translator.integrated_translator import IntegratedTranslator


class TestContextAwareTranslation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        model = "azure-openai/gpt-4o"
        self.translator = IntegratedTranslator(
            model=model, use_context_aware=True, preserve_format=True
        )
        self.translator_with_batch = IntegratedTranslator(
            model=model,
            use_context_aware=True,
            preserve_format=True,
            batch_translation_enabled=True,
        )
        self.translator_without_batch = IntegratedTranslator(
            model=model,
            use_context_aware=True,
            preserve_format=True,
            batch_translation_enabled=False,
        )
        self.test_file = "docs/案例4.xlsx"
        self.output_path = "output"

    async def test_translate_excel_with_context(self):
        # Test translation with context
        result = await self.translator.translate_excel_file(
            self.test_file,
            output_path=self.output_path,
            source_language="chinese",
            target_language="english",
        )

        print("Translation result:", result)

    async def test_translate_excel_with_batch_translation(self):
        # Test translation with batch translation enabled
        result = await self.translator_with_batch.translate_excel_file(
            self.test_file,
            output_path=self.output_path,
            source_language="chinese",
            target_language="english",
        )

        print("Batch translation result:", result)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTest(TestContextAwareTranslation("test_translate_excel_with_context"))
    suite.addTest(
        TestContextAwareTranslation("test_translate_excel_with_batch_translation")
    )

    runner = unittest.TextTestRunner()
    runner.run(suite)
