# 批量翻译方法实现设计

## 概述

本文档详细描述了批量翻译方法的实现，包括上下文感知的批量翻译、token长度检查和分批逻辑。

## 核心方法设计

### 1. translate_dataframe_batch 方法

```python
async def translate_dataframe_batch(
    self, df: pd.DataFrame, target_lang: str = None
) -> pd.DataFrame:
    """
    批量翻译整个DataFrame
    
    Args:
        df: 要翻译的DataFrame
        target_lang: 目标语言
        
    Returns:
        翻译后的DataFrame
    """
    pass
```

### 2. create_translation_batches 方法

```python
def create_translation_batches(
    self, df: pd.DataFrame, max_tokens: int = 4096
) -> List[TranslationBatch]:
    """
    创建适合模型上下文限制的翻译批次
    
    Args:
        df: 要翻译的DataFrame
        max_tokens: 最大token数量
        
    Returns:
        翻译批次列表
    """
    pass
```

### 3. build_batch_context 方法

```python
def build_batch_context(
    self, batch: TranslationBatch, structure: Dict[str, Any]
) -> str:
    """
    为批次构建上下文信息
    
    Args:
        batch: 翻译批次
        structure: 表格结构信息
        
    Returns:
        上下文字符串
    """
    pass
```

### 4. translate_batch_with_context 方法

```python
async def translate_batch_with_context(
    self, batch: TranslationBatch, context: str, target_lang: str
) -> List[str]:
    """
    使用上下文进行批量翻译
    
    Args:
        batch: 翻译批次
        context: 上下文信息
        target_lang: 目标语言
        
    Returns:
        翻译结果列表
    """
    pass
```

## 数据结构设计

### TranslationBatch

```python
@dataclass
class TranslationBatch:
    """翻译批次数据结构"""
    
    # 批次中的文本单元
    texts: List[str]
    
    # 批次的位置信息 (行索引, 列名)
    positions: List[Tuple[int, str]]
    
    # 批次的token数量
    token_count: int
    
    # 批次的上下文信息
    context_info: Dict[str, Any]
```

## 上下文感知批量翻译实现

### 上下文构建策略

1. **表格级上下文**：
   - 表格结构信息（列名、数据类型等）
   - 专业领域信息
   - 数据模式信息

2. **批次级上下文**：
   - 批次内数据的位置信息
   - 批次内数据的关联关系

3. **文本级上下文**：
   - 每个文本单元的行上下文
   - 每个文本单元的列上下文

### 上下文格式

```json
{
  "table_context": {
    "domain": "mechanical",
    "columns": ["零件号", "名称", "规格"],
    "data_types": {"零件号": "text", "名称": "text", "规格": "text"},
    "patterns": {"零件号": ["A123", "B456"], "名称": ["轴承", "齿轮"]}
  },
  "batch_context": {
    "batch_id": 1,
    "total_batches": 5,
    "position_info": [
      {"row": 0, "column": "名称", "original_text": "轴承"},
      {"row": 1, "column": "名称", "original_text": "齿轮"}
    ]
  }
}
```

### 翻译提示格式

```
Translate the following Chinese texts to English. 
Table context:
- Domain: mechanical
- Columns: [零件号, 名称, 规格]
- Data types: {零件号: text, 名称: text, 规格: text}

Batch context:
- Batch 1 of 5
- Position information:
  - Row 0, Column "名称": 轴承
  - Row 1, Column "名称": 齿轮

Translation requirements:
1. Translate all texts to English, do not leave any Chinese characters
2. Maintain consistency with the context provided
3. Return translations in the same order as the original texts
4. Return only the translated texts without any explanations
5. Ensure complete translation, no partial translation allowed

Chinese texts to translate:
1. 轴承
2. 齿轮
```

## Token管理实现

### Token计算方法

使用tiktoken库计算token数量：

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """计算文本的token数量"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

### 批次Token管理

1. 计算上下文信息的token数量
2. 计算每个文本单元的token数量
3. 确保总token数量不超过限制

## 错误处理和重试机制

### 重试策略

1. 网络错误重试（最多3次）
2. API限流错误重试（指数退避）
3. 上下文长度超限错误处理（自动分割批次）

### 错误恢复

1. 缓存已成功的翻译结果
2. 失败时回退到单元格级翻译
3. 记录详细的错误日志

## 性能优化

### 并行处理

1. 批次间并行翻译
2. 多个工作表并行处理
3. 异步API调用

### 缓存机制

1. 批次级翻译结果缓存
2. 上下文信息缓存
3. 领域术语缓存