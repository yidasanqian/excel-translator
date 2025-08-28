FROM python:3.11-slim

ARG PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

# 设置工作目录
WORKDIR /app

# 安装uv
RUN pip config set global.index-url $PIP_INDEX ; \
    pip install uv

# 复制项目文件
COPY main.py pyproject.toml uv.lock README.md ./
COPY src/ ./

# 安装依赖
RUN uv sync --frozen

# 设置环境PATH
ENV TZ=Asia/Shanghai \
    PATH=/app/.venv/bin:$PATH \
    TIKTOKEN_CACHE_DIR=/app/tiktoken_cache
# Cache the tiktoken encoding file
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o')"


# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "18000"]