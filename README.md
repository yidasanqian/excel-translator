# Excel Translator

![License](https://img.shields.io/badge/license-MIT-blue)
![OpenAI](https://img.shields.io/badge/openai-gpt--4o-green)

一个基于OpenAI的智能Excel翻译器，具有上下文感知功能和批量翻译能力，能够准确翻译Excel文件中的内容，同时保持原有的格式和结构。

## 简介

Excel Translator是一个强大的工具，专门用于翻译Excel电子表格中的内容。与传统的逐单元格翻译工具不同，Excel Translator利用先进的AI技术和上下文感知算法，能够理解表格的结构、列的含义以及数据之间的关系，从而提供更准确、更一致的翻译结果。

该工具特别适用于需要翻译技术文档、数据报告、产品规格表等复杂Excel文件的场景，能够确保专业术语的一致性，并保持原始文件的格式和布局。

## 特性

- **上下文感知翻译**：利用表格结构、列类型和专业领域信息进行智能翻译，确保翻译结果符合上下文语境
- **批量翻译**：支持将多行数据合并为单个翻译请求，显著提高翻译效率并减少API调用次数。具有以下高级特性：
- **智能分批**：根据模型token限制智能分批处理大量数据，自动优化批次大小
- **多Sheet支持**：支持翻译包含多个工作表的Excel文件，保持工作表间的引用关系
- **格式保留**：可选择保留原始Excel文件的格式，包括合并单元格、字体样式、边框等
- **术语管理**：内置专业领域术语库（机械、电气等），确保专业术语翻译的一致性
- **智能缓存**：内置缓存机制，避免重复翻译相同内容，提高处理效率
- **领域检测**：自动识别内容所属的专业领域（机械、电气、软件、医疗等），应用相应的翻译规则
- **OpenAI集成**：使用先进的AI模型（如GPT-4o）进行高质量翻译，支持多种语言
- **异步处理**：采用异步编程模型，提高处理效率和响应速度
- **错误处理**：完善的错误处理机制，确保在翻译失败时能够优雅地处理并提供有用的错误信息

## 目录

- [安装](#安装)
- [配置](#配置)
- [使用方法](#使用方法)
- [贡献](#贡献)
- [许可证](#许可证)
## 安装

### 环境要求

- Python 3.11 或更高版本
- OpenAI API密钥

### 克隆项目

```bash
git clone https://github.com/your-username/excel-translator.git
cd excel-translator
```

### 安装依赖

推荐使用 [uv](https://github.com/astral-sh/uv) 来管理依赖：

```bash
# 安装 uv (如果尚未安装)
pip install uv

# 安装项目依赖
uv sync
```

或者使用 pip：

```bash
pip install -e .
```
## 配置

1. 复制 `.env.example` 文件并重命名为 `.env`：

   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，填写必要的配置信息：

   ```bash
   # OpenAI配置
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o
   OPENAI_BASE_URL=https://api.openai.com/v1
   
   # 翻译设置
   TARGET_LANGUAGE=english
   MAX_BATCH_SIZE=50
   REQUEST_TIMEOUT=30
   PRESERVE_FORMAT=true
   # 批量翻译设置
   BATCH_TRANSLATION_ENABLED=true
   MAX_TOKENS=8192
   TOKEN_BUFFER=500
   
   # 文件设置
   UPLOAD_DIR=uploads
   OUTPUT_DIR=output
   MAX_FILE_SIZE=10485760
   
   # 应用设置
   APP_NAME=Excel Translator
   APP_VERSION=1.0.0
   
   # 日志设置
   LOG_LEVEL=INFO
   ```

### 配置参数说明

- `OPENAI_API_KEY`: 你的OpenAI API密钥（必需）
- `OPENAI_MODEL`: 使用的OpenAI模型，默认为 `gpt-4o`
- `OPENAI_BASE_URL`: OpenAI API的基础URL，可选，用于使用代理或自定义端点
- `TARGET_LANGUAGE`: 目标语言，默认为 `english`
- `MAX_BATCH_SIZE`: 批量翻译的最大单元数，默认为 `50`
- `REQUEST_TIMEOUT`: API请求超时时间（秒），默认为 `30`
- `PRESERVE_FORMAT`: 是否保留原始Excel格式，默认为 `true`
- `BATCH_TRANSLATION_ENABLED`: 是否启用批量翻译，默认为 `true`。启用后将使用上下文感知的批量翻译功能，显著提高翻译效率并减少API调用次数
- `MAX_TOKENS`: 最大输出token数量，默认为 `8192`。用于控制批量翻译中每个批次的最大token数量，避免超出模型限制
- `TOKEN_BUFFER`: token缓冲区大小，默认为 `1000`。为批量翻译中的格式化内容预留的token空间，确保不会因格式化内容超出token限制
- `UPLOAD_DIR`: 上传文件目录，默认为 `uploads`
- `OUTPUT_DIR`: 输出文件目录，默认为 `output`
- `MAX_FILE_SIZE`: 最大文件大小（字节），默认为 `10485760`（10MB）
- `APP_NAME`: 应用名称，默认为 `Excel Translator`
- `APP_VERSION`: 应用版本，默认为 `1.0.0`
- `LOG_LEVEL`: 日志级别，默认为 `INFO`
## 使用方法

### 编程接口使用

#### 基本用法

```python
import asyncio
from translator.integrated_translator import IntegratedTranslator

async def translate_excel():
    # 创建翻译器实例
    translator = IntegratedTranslator(
        use_context_aware=True,  # 使用上下文感知翻译
        preserve_format=True     # 保留原始格式
    )
    
    # 翻译Excel文件
    result_path = await translator.translate_excel_file(
        file_path="input.xlsx",
        output_path="output",
        target_language="english"
    )
    
    print(f"翻译完成，结果保存在: {result_path}")

# 运行异步函数
asyncio.run(translate_excel())
```

#### 批量翻译

Excel Translator默认启用批量翻译功能，可以显著提高翻译效率并减少API调用次数。批量翻译会智能地将多个文本单元组合成批次进行翻译，同时保持上下文信息。

```python
import asyncio
from translator.integrated_translator import IntegratedTranslator

async def translate_excel_with_batch():
    # 创建翻译器实例（默认启用批量翻译）
    translator = IntegratedTranslator(
        use_context_aware=True,      # 使用上下文感知翻译
        preserve_format=True,       # 保留原始格式
        batch_translation_enabled=True  # 启用批量翻译（默认值）
    )
    
    # 翻译Excel文件（将自动使用批量翻译）
    result_path = await translator.translate_excel_file(
        file_path="input.xlsx",
        output_path="output",
        target_language="english"
    )
    
    print(f"翻译完成，结果保存在: {result_path}")
    
    # 获取翻译统计信息
    stats = translator.get_translation_stats()
    print(f"翻译统计: {stats}")

# 运行异步函数
asyncio.run(translate_excel_with_batch())
```

### 命令行使用

Excel Translator 提供了一个命令行接口，方便用户直接从终端翻译Excel文件。

#### 基本用法

```bash
python main.py -i input.xlsx -o output_dir -l english
```

#### 命令行参数说明

- `-i`, `--input`: 输入Excel文件路径（必需）
- `-o`, `--output`: 输出目录路径（可选，默认为输入文件所在目录）
- `-l`, `--language`: 目标语言（可选，默认为 "english"）
- `-c`, `--context-aware`: 使用上下文感知翻译（可选，默认启用）
- `--no-context-aware`: 不使用上下文感知翻译
- `-p`, `--preserve-format`: 保留Excel格式（可选，默认为 True）
- `--openai-api-key`: OpenAI API密钥（也可以通过环境变量OPENAI_API_KEY设置）
- `--openai-model`: OpenAI模型（可选，默认为 "gpt-4o"）
- `--openai-base-url`: OpenAI API基础URL（可选）

注意：Excel Translator默认启用批量翻译功能，可以显著提高翻译效率并减少API调用次数。批量翻译会智能地将多个文本单元组合成批次进行翻译，同时保持上下文信息。目前命令行接口不提供直接控制批量翻译的参数，但可以通过配置文件中的`BATCH_TRANSLATION_ENABLED`参数来控制（请参见配置部分）。

#### 示例

1. 基本翻译：
   ```bash
   python main.py -i docs/案例5.xlsx -o output -l english
   ```

2. 使用上下文感知翻译并保留格式：
   ```bash
   python main.py -i docs/案例5.xlsx -o output -l english -c -p
   ```

3. 不使用上下文感知翻译：
   ```bash
   python main.py -i docs/案例5.xlsx -o output -l english --no-context-aware
   ```

4. 指定OpenAI API密钥和模型：
   ```bash
   python main.py -i docs/案例5.xlsx -o output -l english --openai-api-key your_api_key_here --openai-model gpt-4o
   ```

5. 使用配置文件控制批量翻译（默认启用）：
   ```bash
   # 在.env文件中设置 BATCH_TRANSLATION_ENABLED=false 可以禁用批量翻译
   python main.py -i docs/案例5.xlsx -o output -l english
   ```

## 贡献

欢迎任何形式的贡献！如果你想为这个项目做贡献，请遵循以下步骤：

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 开发环境设置

1. 克隆 Fork 的仓库
2. 安装依赖：`uv sync`

### 提交规范

请确保你的代码遵循项目的编码规范，并包含适当的测试。

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 支持

如果你觉得这个项目对你有帮助，请考虑给它一个⭐️！

如果你有任何问题或建议，请提交 issue 或联系项目维护者。