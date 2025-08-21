# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Excel Translator is an intelligent Excel translation tool based on OpenAI, featuring context-aware capabilities and batch translation functionality. It can accurately translate content in Excel files while maintaining the original format and structure.

Key features:
- Context-aware translation using table structure and domain knowledge
- Batch translation for improved efficiency and reduced API calls
- Format preservation including merged cells, fonts, and borders
- Terminology management for consistent professional terms
- Smart caching to avoid re-translating identical content
- Domain detection (mechanical, electrical, software, medical, etc.)
- Asynchronous processing for better performance
- Error handling for graceful failure recovery

## Code Architecture

The project follows a modular architecture with the following key components:

1. **Main Translation Interface**:
   - `IntegratedTranslator` (`src/translator/integrated_translator.py`): Unified translation interface supporting both context-aware and traditional translation methods.

2. **Core Translation Engines**:
   - `ContextAwareTranslator` (`src/translator/context_aware_translator.py`): Context-aware translation engine with smart batching capabilities.
   - `BatchTranslator` (`src/translator/batch_translator.py`): Handles batch translation of multiple text units while preserving context.
   - `ExcelCellTranslator` (`src/translator/cell_translator.py`): Traditional cell-by-cell translation method.

3. **Excel Processing**:
   - `ExcelHandler` (`src/translator/excel_handler.py`): Basic Excel file reading/writing.
   - `EnhancedExcelHandler` (`src/translator/enhanced_excel_handler.py`): Advanced Excel processing with format preservation.

4. **Supporting Components**:
   - `TableStructureAnalyzer`: Analyzes table structure and detects domains.
   - `TerminologyManager`: Manages domain-specific terminology.
   - `SmartBatcher`: Creates intelligent translation batches.
   - `TokenManager`: Manages token counting for batch translation.
   - `TranslationFilter` (`src/translator/translation_filter.py`): Determines if text needs translation.

5. **Configuration**:
   - `Settings` (`src/config/settings.py`): Application configuration using Pydantic.
   - Environment variables for API keys, model settings, and translation options.

## Common Development Tasks

### Building and Running

1. Install dependencies:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

2. Configure environment:
   ```bash
   # Copy example config
   cp .env.example .env
   
   # Edit .env with your settings
   # Set OPENAI_API_KEY at minimum
   ```

3. Translate Excel files:
   ```bash
   # Basic usage
   python main.py -i input.xlsx -o output_dir -l english
   
   # With context-aware translation and format preservation
   python main.py -i input.xlsx -o output_dir -l english -c -p
   ```

### Testing

Run tests with:
```bash
python -m pytest tests/ -v
```

### Key Classes and Methods

1. **IntegratedTranslator**:
   - `translate_excel_file()`: Main method for translating Excel files
   - `translate_excel_data()`: Translates DataFrame data
   - `get_translation_stats()`: Returns translation statistics

2. **ContextAwareTranslator**:
   - `translate_dataframe()`: Translates entire DataFrames with context
   - `get_cache_stats()`: Returns cache statistics

3. **BatchTranslator**:
   - `translate_dataframe_batch()`: Batch translates DataFrames
   - `create_translation_batches()`: Creates token-aware translation batches

## Configuration Parameters

Key environment variables in `.env`:
- `OPENAI_API_KEY`: OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-4o)
- `TARGET_LANGUAGE`: Target language (default: english)
- `PRESERVE_FORMAT`: Preserve Excel formatting (default: true)
- `BATCH_TRANSLATION_ENABLED`: Enable batch translation (default: true)
- `MAX_TOKENS`: Maximum tokens per batch (default: 4096)
- `TOKEN_BUFFER`: Token buffer for formatting (default: 500)