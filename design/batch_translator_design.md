# 批量翻译器设计文档

## 概述

本文档描述了新的批量翻译器模块的设计，该模块旨在提高Excel翻译器的效率，通过将多行数据合并为单个翻译请求，而不是逐行翻译。

## 设计目标

1. 提高翻译效率，减少API调用次数
2. 保持上下文感知翻译的质量
3. 考虑模型的上下文长度限制（如8192 tokens）
4. 保持与现有代码的兼容性

## 架构设计

### 类结构

#### BatchTranslator
主要的批量翻译器类，负责处理整个DataFrame的批量翻译。

#### TokenManager
负责管理token计数和分批逻辑，确保不超过模型的上下文长度限制。

#### BatchContextBuilder
负责构建批量翻译的上下文信息。

### 核心方法

1. `translate_dataframe_batch`: 批量翻译整个DataFrame
2. `create_translation_batches`: 创建适合模型上下文限制的翻译批次
3. `build_batch_context`: 为每个批次构建上下文信息
4. `translate_batch_with_context`: 使用上下文进行批量翻译

## 实现细节

### Token管理

使用tiktoken库来计算token数量，确保不超过模型的上下文长度限制。

### 批次创建逻辑

1. 分析DataFrame结构
2. 计算每行数据的token数量
3. 根据token限制创建批次
4. 为每个批次构建适当的上下文

### 上下文构建

为每个批次构建包含以下信息的上下文：
- 表格结构信息
- 专业领域信息
- 数据模式信息
- 批次内数据的位置信息

## 与现有代码的集成

新的批量翻译器将作为现有`ContextAwareTranslator`类的一个可选功能，通过配置参数启用。

## 配置选项

在settings.py中添加新的配置选项：
- `batch_translation_enabled`: 是否启用批量翻译
- `max_tokens`: 最大输出token数量