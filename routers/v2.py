"""Phase 2：混合召回 + Gemini 精排（JSON）。"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.v2_search_flow import run_v2_search

router = APIRouter(tags=["v2"])


class SemanticHit(BaseModel):
    id: str
    title: str
    snippet: str = Field(description="检索阶段摘要")
    score: float = Field(
        description="相关度：优先为 Gemini confidence，回退时为规则分或混合分（通常 ≥0.55）"
    )
    reason: str | None = Field(default=None, description="Gemini 给出的匹配理由")


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[SemanticHit]
    message: str | None = Field(
        default=None,
        description="可选提示：如未配置 API Key 或 Gemini 异常后已本地回退；有结果时通常为 null",
    )


@router.get("/search", response_model=SemanticSearchResponse)
async def v2_semantic_search(q: str) -> Any:
    """
    混合召回：关键词 Top5 与向量 Top5 去重合并，并补足至少 5 个不同 SOP。
    精排：使用 `google.generativeai` + `GEMINI_API_KEY`（见 `.env`），模型默认 `gemini-1.5-flash`（可用 `GEMINI_V2_RERANK_MODEL` 覆盖），
    返回 1–2 条结果；Gemini 失败时自动用规则路由 + 混合检索 Top2 兜底，不再返回「未找到精准匹配」。
    """
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询参数 q 不能为空")

    rows, msg = await asyncio.to_thread(run_v2_search, query)
    results = [
        SemanticHit(
            id=h.id,
            title=h.title,
            snippet=h.snippet,
            score=h.score,
            reason=h.reason,
        )
        for h in rows
    ]
    return SemanticSearchResponse(query=query, results=results, message=msg)
