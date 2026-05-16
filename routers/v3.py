"""Phase 3：SRE On-Call 对话（readFile 工具 + 先验路由）。"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.agent_service import run_agent
from services.sre_agent import run_sre_agent

router = APIRouter(tags=["v3"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="用户故障描述或提问")


class ToolCallItem(BaseModel):
    tool: str
    fname: str
    ok: bool
    error: str | None = None
    title: str | None = None
    body_preview: str | None = None


class AskResponse(BaseModel):
    answer: str
    thoughts: list[str] = Field(description="Agent 思考与工具调用过程")
    tool_calls: list[ToolCallItem]
    files_read: list[str]


class AgentHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, description="当前用户提问")
    history: list[AgentHistoryItem] = Field(
        default_factory=list,
        description="已完成的多轮对话（不含当前 question）",
    )


@router.post("/agent")
async def agent_ask(req: AgentRequest) -> Any:
    """
    Gemini（默认 2.5 Flash）+ readFile；低 Token、固定 [关键命令]/[离线状态] 模板；失败则本地 Fallback。
    返回字段：answer, mode, steps, sop_links, error, quota_exceeded。
    """
    try:
        hist = [{"role": h.role, "content": h.content} for h in req.history]
        return await asyncio.to_thread(run_agent, req.question, hist)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"run_agent 执行失败: {e}") from e


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> Any:
    """
    SRE On-Call 专家对话：先验预测 `data/sop-xxx.html`，**单领域问题时只 readFile 1 个**以省配额；
    回答为结构化四模块（故障诊断 / 立即行动 / 排查建议 / 风险预警），**禁止全文打印 SOP**。
    响应含 **thoughts**、**tool_calls**、**files_read**。
    """
    try:
        result = await asyncio.to_thread(run_sre_agent, req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent 执行失败: {e}") from e

    return AskResponse(
        answer=result.answer,
        thoughts=result.thoughts,
        tool_calls=[
            ToolCallItem(
                tool=t.tool,
                fname=t.fname,
                ok=t.ok,
                error=t.error,
                title=t.title,
                body_preview=t.body_preview,
            )
            for t in result.tool_calls
        ],
        files_read=result.files_read,
    )
