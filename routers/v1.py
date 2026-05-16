"""Phase 1：文档入库与关键词检索。"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from services.document_store import DOCUMENT_ID_RE, put_document
from services.keyword_search import keyword_search
from services.vector_index import upsert_document_embedding
from utils.html_processor import extract_title_and_body

router = APIRouter(tags=["v1"])


class DocumentCreateResponse(BaseModel):
    id: str
    title: str
    body_preview: str = Field(description="正文前 200 字预览")


class SearchHit(BaseModel):
    id: str
    title: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


def _normalize_client_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    if not DOCUMENT_ID_RE.fullmatch(s):
        raise HTTPException(
            status_code=400,
            detail="id 仅允许字母数字开头，可含 . _ -，长度 1–256",
        )
    return s


@router.post("/documents", response_model=DocumentCreateResponse)
async def create_document(
    html: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    client_id: str | None = Form(default=None, alias="id"),
) -> Any:
    """
    上传 HTML：支持 multipart 字段 `html` 文本，或上传 `file`（text/html 或纯文本 html）。
    可选字段 `id`：指定文档 id（同名则覆盖）；省略时由服务端生成 UUID。
    """
    raw: str | None = html
    if file is not None:
        raw_bytes = await file.read()
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw_bytes.decode("utf-8", errors="replace")

    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="请提供非空的 html 表单字段或上传文件")

    title, body = extract_title_and_body(raw)
    custom = _normalize_client_id(client_id)
    doc_id = custom if custom is not None else str(uuid.uuid4())
    put_document(doc_id, title, body)
    await asyncio.to_thread(upsert_document_embedding, doc_id, title, body)
    preview = body[:200] + ("…" if len(body) > 200 else "")
    return DocumentCreateResponse(id=doc_id, title=title, body_preview=preview)


@router.get("/search", response_model=SearchResponse)
def search(q: str) -> Any:
    """
    简单关键词匹配：在标题与正文中不区分大小写查找子串。
    使用正则 re.escape 处理查询中的特殊字符（如 &、<、>、正则元字符等），避免被当作模式解释。
    """
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询参数 q 不能为空")

    try:
        hits = keyword_search(query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    results = [SearchHit(id=h.id, title=h.title, snippet=h.snippet, score=h.score) for h in hits]
    return SearchResponse(query=query, results=results)
