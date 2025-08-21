"""Excel Translator API 主入口."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# 创建FastAPI应用实例
app = FastAPI(
    title="Excel Translator API", description="Excel文件翻译服务API", version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(router)


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Excel Translator API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=18000, reload=True)
