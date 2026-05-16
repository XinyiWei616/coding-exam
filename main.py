"""On-Call 助手：FastAPI 入口，挂载 v1 / v2 / v3 API。"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.env_bootstrap import PROJECT_ROOT, gemini_api_key_from_env

# 项目根 .env：override=True 覆盖 shell 中已导出但为空的变量
_APP_ROOT = Path(__file__).resolve().parent
load_dotenv(_APP_ROOT / ".env", override=True)
if (_APP_ROOT / ".env.local").is_file():
    load_dotenv(_APP_ROOT / ".env.local", override=True)
load_dotenv(override=True)

_env_file = _APP_ROOT / ".env"
if not _env_file.is_file():
    print(f"[startup] 警告: 未找到 {_env_file}", flush=True)
elif _env_file.stat().st_size == 0:
    print(
        f"[startup] 警告: {_env_file} 为空文件(0 字节)，请写入 GEMINI_API_KEY=你的密钥",
        flush=True,
    )
_gemini_len = len(os.getenv("GEMINI_API_KEY") or "")
print(f"[startup] GEMINI_API_KEY length={_gemini_len}", flush=True)
if _gemini_len == 0 and not gemini_api_key_from_env():
    print(
        "[startup] 提示: 若仅配置了 GOOGLE_API_KEY，/health 的 ready 仍可能为 true；"
        "但本行 length 只统计 GEMINI_API_KEY 变量名",
        flush=True,
    )

# 与 env_bootstrap 内 PROJECT_ROOT 一致；此处不再二次加载，避免顺序混淆

from routers import v1, v2, v3
from services.data_bootstrap import bootstrap_from_data_dir
from services.document_store import clear_store
from services.vector_index import clear_all_vectors, init_vector_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_vector_index()
    clear_all_vectors()
    clear_store()
    bootstrap_from_data_dir()
    yield


app = FastAPI(title="On-Call Assistant", version="0.1.0", lifespan=lifespan)

app.include_router(v1.router, prefix="/v1")
app.include_router(v2.router, prefix="/v2")
app.include_router(v3.router, prefix="/v3")

app.mount(
    "/data-files",
    StaticFiles(directory=str(PROJECT_ROOT / "data")),
    name="data_files",
)


@app.get("/v3")
def v3_chat_ui() -> FileResponse:
    """极简对话页 v3.html（Gemini / 本地降级双轨展示）。"""
    path = PROJECT_ROOT / "v3.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"未找到 {path}")
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict[str, str | bool | int]:
    """
    ready：任一有效 Gemini/Google API Key 已加载即为 true。
    gemini_api_key_length：仅统计环境变量 GEMINI_API_KEY 字符串长度（不含 GOOGLE_API_KEY）。
    """
    env_path = PROJECT_ROOT / ".env"
    gemini_raw = os.getenv("GEMINI_API_KEY") or ""
    ready = bool(gemini_api_key_from_env())
    return {
        "status": "ok",
        "ready": ready,
        "gemini_api_key_loaded": ready,
        "gemini_api_key_length": len(gemini_raw),
        "env_file_path": str(env_path),
        "env_file_present": env_path.is_file(),
    }
