"""Excel Translator API 路由."""

import os
import uuid
import tempfile
import asyncio
from typing import Dict, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from .sse_manager import SSEManager
from translator.integrated_translator import IntegratedTranslator
from config.logging_config import get_logger

logger = get_logger(__name__)

# 创建路由实例
router = APIRouter(prefix="/api/v1/excel-translator")

# 创建SSE管理器实例
sse_manager = SSEManager()


@router.post("/translate")
async def translate_excel(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    model: str = Form(...),
    domain_terms: Optional[str] = Form(None),
):
    """
    翻译Excel文件并返回SSE流式响应

    Args:
        file: 上传的Excel文件
        source_language: 源语言
        target_language: 目标语言
        model: 使用的模型
        domain_terms: 领域术语字典（可选，JSON格式字符串，格式为 {domain: {term: translation}}

    Returns:
    SSE消息格式：
       - 进度消息：data: {"type": "progress", "progress": 50, "message": "已完成5/10个批次"}
       - 完成消息：data: {"type": "complete", "message": "翻译完成"}
       - 文件内容消息：data: {"type": "file", "filename": "xxx.xlsx", "content": "base64_encoded_content"}
       - 错误消息：data: {"type": "error", "message": "错误详情"}
    """
    if source_language == target_language:
        raise ValueError("源语言和目标语言不能相同")

    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 解析领域术语
    parsed_domain_terms = None
    if domain_terms:
        try:
            import json

            parsed_domain_terms = json.loads(domain_terms)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid domain_terms format: {str(e)}"
            )

    # 读取文件内容，避免文件句柄关闭问题
    file_content = await file.read()

    # 返回SSE流式响应
    return StreamingResponse(
        translate_excel_stream(
            task_id,
            file_content,
            source_language,
            target_language,
            model,
            parsed_domain_terms,
        ),
        media_type="text/event-stream",
    )


async def translate_excel_stream(
    task_id: str,
    file_content: bytes,
    source_language: str,
    target_language: str,
    model: str,
    domain_terms: Optional[Dict] = None,
):
    """翻译Excel文件并流式返回结果

    FastAPI StreamingResponse的工作机制：
    - StreamingResponse需要一个生成器函数作为参数
    - 客户端连接实际上是在响应开始流式传输时建立的
    - 在函数开始时注册客户端确保了所有消息都能被正确发送
    """
    # 注册客户端连接，确保所有消息都能被发送
    queue = await sse_manager.register_client(task_id)

    # 定义翻译任务
    async def translation_task():
        try:
            # 发送初始化消息
            await sse_manager.send_progress(task_id, 0, "开始处理文件")

            # 保存上传的文件到临时位置
            # 使用传入的文件内容创建临时文件
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
            tmp_file.close()

            # 创建临时输出目录
            output_dir = tempfile.mkdtemp()

            # 创建翻译器实例
            translator = IntegratedTranslator(
                model=model,
                use_context_aware=True,
                preserve_format=True,
                batch_translation_enabled=True,
            )

            # 定义进度回调函数
            async def progress_callback(progress: float, message: str):
                await sse_manager.send_progress(task_id, progress, message)

            # 执行翻译
            result_path = await translator.translate_excel_file(
                tmp_file_path,
                output_dir,
                source_language,
                target_language,
                domain_terms,
                progress_callback,
            )

            logger.info(f"翻译完成，结果文件路径: {result_path}")
            # 读取翻译后的文件内容
            with open(result_path, "rb") as f:
                result_file_content = f.read()

            # 发送文件内容
            await sse_manager.send_file(
                task_id, os.path.basename(result_path), result_file_content
            )

            # 发送完成消息
            await sse_manager.send_complete(task_id, "翻译完成")

        except Exception as e:
            # 发送错误消息
            logger.exception(f"翻译失败: {str(e)}")
            await sse_manager.send_error(task_id, f"翻译失败: {str(e)}")

        finally:
            try:
                if "tmp_file_path" in locals():
                    os.unlink(tmp_file_path)
            except Exception:
                pass

    # 启动翻译任务作为后台任务
    asyncio.create_task(translation_task())

    # 流式传输消息
    try:
        while True:
            message = await queue.get()
            yield message
            queue.task_done()

            # 如果是完成或错误消息，结束流式传输
            if '"type": "complete"' in message or '"type": "error"' in message:
                break

    except asyncio.CancelledError:
        await sse_manager.unregister_client(task_id)
        raise
    except Exception:
        await sse_manager.unregister_client(task_id)
    finally:
        await sse_manager.unregister_client(task_id)
