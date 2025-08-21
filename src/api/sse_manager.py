"""SSE管理器，用于处理实时进度更新和文件传输."""

import json
import base64
import asyncio
from typing import AsyncGenerator, Dict, Any
from models.models import SSEMessageType


class SSEManager:
    """SSE管理器类."""

    def __init__(self):
        self.clients = {}  # 存储客户端连接

    async def send_progress(self, task_id: str, progress: float, message: str) -> None:
        """发送进度更新."""
        if task_id in self.clients:
            message_data = {
                "type": SSEMessageType.PROGRESS,
                "progress": progress,
                "message": message,
            }
            await self._send_sse_message(task_id, message_data)

    async def send_complete(self, task_id: str, message: str) -> None:
        """发送完成消息."""
        if task_id in self.clients:
            message_data = {
                "type": SSEMessageType.COMPLETE,
                "progress": 100,
                "message": message,
            }
            await self._send_sse_message(task_id, message_data)

    async def send_error(self, task_id: str, message: str) -> None:
        """发送错误消息."""
        if task_id in self.clients:
            message_data = {"type": SSEMessageType.ERROR, "message": message}
            await self._send_sse_message(task_id, message_data)

    async def send_file(self, task_id: str, filename: str, file_content: bytes) -> None:
        """发送文件内容."""
        if task_id in self.clients:
            # 将文件内容编码为base64
            encoded_content = base64.b64encode(file_content).decode("utf-8")
            message_data = {
                "type": SSEMessageType.FILE,
                "filename": filename,
                "content": encoded_content,
            }
            await self._send_sse_message(task_id, message_data)

    async def _send_sse_message(
        self, task_id: str, message_data: Dict[str, Any]
    ) -> None:
        """发送SSE消息."""
        if task_id in self.clients:
            queue = self.clients[task_id]
            message = f"data: {json.dumps(message_data, ensure_ascii=False)}\n\n"
            await queue.put(message)

    async def register_client(self, task_id: str) -> asyncio.Queue:
        """注册客户端连接."""
        queue = asyncio.Queue()
        self.clients[task_id] = queue
        return queue

    async def unregister_client(self, task_id: str) -> None:
        """注销客户端连接."""
        if task_id in self.clients:
            del self.clients[task_id]

    async def stream_messages(self, task_id: str) -> AsyncGenerator[str, None]:
        """流式传输消息."""
        queue = await self.register_client(task_id)
        try:
            while True:
                message = await queue.get()
                yield message
                queue.task_done()
        except asyncio.CancelledError:
            await self.unregister_client(task_id)
            raise
        except Exception:
            await self.unregister_client(task_id)
