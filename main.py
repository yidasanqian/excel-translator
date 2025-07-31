#!/usr/bin/env python3
"""Excel Translator - 命令行工具."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径，以便可以导入 translator 模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from translator.integrated_translator import IntegratedTranslator
from config.settings import settings


def create_parser():
    """创建命令行参数解析器."""
    parser = argparse.ArgumentParser(
        description="Excel Translator - 将Excel文件翻译成指定语言"
    )

    parser.add_argument("-i", "--input", required=True, help="输入Excel文件路径")

    parser.add_argument("-o", "--output", help="输出目录路径（默认为输入文件所在目录）")

    parser.add_argument(
        "-l",
        "--language",
        default=settings.target_language,
        help=f"目标语言（默认: {settings.target_language}）",
    )

    parser.add_argument(
        "-c",
        "--context-aware",
        action="store_true",
        default=True,
        help="使用上下文感知翻译（默认启用）",
    )

    parser.add_argument(
        "--no-context-aware",
        action="store_false",
        dest="context_aware",
        help="不使用上下文感知翻译",
    )

    parser.add_argument(
        "-p",
        "--preserve-format",
        action="store_true",
        default=settings.preserve_format,
        help=f"保留Excel格式（默认: {settings.preserve_format}）",
    )

    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API密钥（也可以通过环境变量OPENAI_API_KEY设置）",
    )

    parser.add_argument(
        "--openai-model",
        default=settings.openai_model,
        help=f"OpenAI模型（默认: {settings.openai_model}）",
    )

    parser.add_argument("--openai-base-url", help="OpenAI API基础URL")

    return parser


async def translate_file(args):
    """根据命令行参数翻译文件."""
    # 如果提供了OpenAI API密钥，更新配置
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    if args.openai_model:
        os.environ["OPENAI_MODEL"] = args.openai_model

    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url

    # 创建翻译器实例
    translator = IntegratedTranslator(
        use_context_aware=args.context_aware, preserve_format=args.preserve_format
    )

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = str(Path(args.input).parent)

    # 执行翻译
    print(f"开始翻译文件: {args.input}")
    print(f"目标语言: {args.language}")
    print(f"使用上下文感知翻译: {args.context_aware}")
    print(f"保留格式: {args.preserve_format}")
    print(f"输出路径: {output_path}")

    try:
        result_path = await translator.translate_excel_file(
            args.input, output_path, args.language
        )
        print(f"翻译完成，结果保存在: {result_path}")
    except Exception as e:
        print(f"翻译失败: {e}")
        sys.exit(1)


def main():
    """主函数."""
    parser = create_parser()
    args = parser.parse_args()

    # 运行异步翻译任务
    asyncio.run(translate_file(args))


if __name__ == "__main__":
    main()
