"""Gemini（默认 2.5 Flash）+ readFile；工具侧截正文、答复侧可调 max_token；Fallback 离线。"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import google.generativeai as genai
from bs4 import BeautifulSoup
from google.api_core import exceptions as google_api_exceptions

from services.env_bootstrap import gemini_api_key_from_env
from services.gemini_rerank import _rerank_model_candidates
from services.sop_tools import DATA_DIR, normalize_fname, predict_sop_files, read_file

logger = logging.getLogger(__name__)

# Agent 路由：默认 2.5 Flash；环境变量 GEMINI_AGENT_MODEL 优先；其后与精排共用回退列表
_DEFAULT_AGENT_MODEL = "gemini-2.5-flash"
# readFile 返回正文注入模型的最大字符数（控制输入 Token，超长截断保留前几节）
_AGENT_TOOL_BODY_CHARS = max(2048, int(os.environ.get("GEMINI_AGENT_TOOL_BODY_CHARS", "8000")))
_AGENT_MAX_OUTPUT = max(512, int(os.environ.get("GEMINI_AGENT_MAX_OUTPUT_TOKENS", "4096")))


def _agent_model_sequence() -> list[str]:
    out: list[str] = []
    configured = (os.environ.get("GEMINI_AGENT_MODEL") or "").strip()
    if configured:
        out.append(configured)
    for m in (_DEFAULT_AGENT_MODEL, *_rerank_model_candidates()):
        if m not in out:
            out.append(m)
    return out


_SYSTEM_INSTRUCTION = """你是 SRE On-Call Agent，模型：Gemini 2.5 Flash。你只负责两件事：选对 SOP → 读出工具正文后**合成最终结论**。
工具 readFile(fileName)：仅可读 data 目录下 sop-001.html～sop-010.html。

【必须遵守】
1) 先看用户问题，再调用 readFile；一般只读 **1** 份最相关文件（跨域至多 2 份）。例如「数据库主从延迟」「慢查询」「主从延迟超 30 秒」应对应 **sop-002.html**，不要泛泛读 sop-001。
2) **工具返回≠结束**。readFile 成功返回正文后，你**还必须**在同一轮回复里写完下面四段；禁止只出现函数调用而没有最终文字回答。
3) 正文若在工具结果被截断，只围绕与用户问题相关的片段推理；没有的写「文档未提及」，严禁编造联系人/外链。
4) 禁止英文套话、禁止「希望对你有帮助」、禁止要求用户再问一轮。
5) 输出一次完成，少用装饰性 Markdown（不需要大段标题）。

【输出模板——四段须按顺序出现，标题原样】
[思考纪要]:
• 至多 4 条极短子弹：读的哪个 sop-xxx.html、用户现象对应文档哪一类场景。（中文，无空话）

[处理步骤]:
面向用户问题的**可执行**处置清单；用 numbered 简短中文列出（至少 **3** 步；文档无解则逐项写「文档未提及」）。

[关键命令]:
仅从 SOP 抽取的 Shell / SQL / 工具单行，每行一条；没有则一行「-」

[离线状态]:
仅写一行：OK （已能从 SOP 作答）或 LACK_DATA （正文全无相关信息）"""


_USER_COMPLETION_GUARD = """
【本条必须执行】调用 readFile 若已成功返回正文：请立即写完 [思考纪要]、[处理步骤]、[关键命令]、[离线状态] 四段再结束；不得在工具结果后留白。"""


def _truncate_tool_body(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    if not payload.get("ok") or not isinstance(payload.get("body"), str):
        return payload
    body = payload["body"]
    if len(body) <= limit:
        return payload
    clipped = body[:limit].rstrip() + "\n\n…（以上为工具返回节选，后续已截断以省 Token；请据节选结合用户问题作答，不足处标「文档未提及」）"
    out = dict(payload)
    out["body"] = clipped
    out["body_chars"] = len(clipped)
    return out


def _cmd_snippets_from_text(text: str, max_cmds: int = 12) -> list[str]:
    """从摘录正文中抓取疑似 Shell/SQL 片段（正则粗抽，Fallback 专用）。"""
    if not (text or "").strip():
        return []
    patterns = [
        r"SHOW\s+[A-Za-z_][A-Za-z0-9_\s]{0,100}",
        r"\bkubectl\b[^\n。.;；]{2,140}",
        r"\bmysql\b[^\n。.;；]{2,120}",
        r"\bjmap\b[^\n。.;；]{2,100}",
        r"\bhelm\b[^\n。.;；]{2,100}",
        r"\bgit\b[^\n。.;；]{2,80}",
        r"\bcurl\b[^\n。.;；]{2,120}",
        r"`([^`]{2,140})`",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s = (m.group(1) if m.lastindex else m.group(0)).strip()
            s = " ".join(s.split())
            if len(s) < 2 or s in seen:
                continue
            seen.add(s)
            found.append(s)
            if len(found) >= max_cmds:
                return found
    return found


def _schema_answer(cmd_block: str, status_line: str) -> str:
    return f"[关键命令]:\n{cmd_block}\n\n[离线状态]:\n{status_line}"


def _detect_quota_error(exc: BaseException) -> bool:
    if isinstance(exc, (google_api_exceptions.ResourceExhausted, google_api_exceptions.TooManyRequests)):
        return True
    msg = str(exc).lower()
    return "429" in msg or "resource exhausted" in msg or "quota" in msg or "rate limit" in msg


def _safe_response_text(response: Any) -> str:
    try:
        return (response.text or "").strip()
    except Exception:
        return ""


def _pick_relevant_files(question: str, max_n: int = 2) -> list[str]:
    predicted = predict_sop_files(question, max_files=max_n)
    out: list[str] = []
    seen: set[str] = set()
    for p in predicted:
        fn = normalize_fname(p)
        if fn in seen:
            continue
        if (DATA_DIR / fn).is_file():
            seen.add(fn)
            out.append(fn)
        if len(out) >= max_n:
            return out
    for path in sorted(DATA_DIR.glob("sop-*.html")):
        name = path.name
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= max_n:
            break
    return out


def _fallback_payload(
    question: str,
    steps: list[str],
    *,
    prior_error: str | None,
    quota_exceeded: bool,
) -> dict[str, Any]:
    files = _pick_relevant_files(question, max_n=2)
    blob_for_cmds: list[str] = []
    for fn in files:
        path = DATA_DIR / fn
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        blob_for_cmds.append(soup.get_text(" ", strip=True))

    cmds = _cmd_snippets_from_text("\n".join(blob_for_cmds))
    if cmds:
        cmd_block = "\n".join(cmds)
        status = "OK"
    else:
        cmd_block = "-"
        status = "LACK_DATA"

    fb = " ".join(f"/data-files/{fn}" for fn in files) if files else ""
    answer = _schema_answer(cmd_block, status)
    if fb:
        answer += f"\nFallback:{fb}"

    sop_links = [{"label": fn, "href": f"/data-files/{fn}"} for fn in files]

    return {
        "answer": answer,
        "mode": "fallback",
        "steps": steps,
        "sop_links": sop_links,
        "error": prior_error,
        "quota_exceeded": quota_exceeded,
    }


def run_agent(question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Gemini 2.5 Flash + readFile；工具正文可截断以省 Token；reply 配额默认 4096；失败则 Fallback。
    """
    history = history or []
    q = (question or "").strip()
    if not q:
        return {
            "answer": _schema_answer("-", "LACK_DATA"),
            "mode": "fallback",
            "steps": [],
            "sop_links": [],
            "error": "empty_question",
            "quota_exceeded": False,
        }

    steps_out: list[str] = []

    def readFile(fileName: str) -> dict:
        raw = (fileName or "").strip()
        steps_out.append(f"Step: 🔍 正在检索 {raw}")
        return _truncate_tool_body(read_file(raw), _AGENT_TOOL_BODY_CHARS)

    api_key = gemini_api_key_from_env()
    if not api_key:
        return _fallback_payload(
            q,
            steps_out,
            prior_error="未配置 GEMINI_API_KEY / GOOGLE_API_KEY",
            quota_exceeded=False,
        )

    genai.configure(api_key=api_key)

    model_names = _agent_model_sequence()

    gen_cfg = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=_AGENT_MAX_OUTPUT,
    )
    for raw_name in model_names:
        model_name = raw_name if "/" in raw_name else f"models/{raw_name}"
        try:
            model = genai.GenerativeModel(
                model_name,
                tools=[readFile],
                system_instruction=_SYSTEM_INSTRUCTION,
            )

            hist: list[dict[str, Any]] = []
            for item in history:
                role = item.get("role") or "user"
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                r = "user" if role == "user" else "model"
                hist.append({"role": r, "parts": [content]})

            chat = model.start_chat(history=hist, enable_automatic_function_calling=True)
            user_turn = (q.strip() + _USER_COMPLETION_GUARD).strip()
            response = chat.send_message(user_turn, generation_config=gen_cfg)
            text = _safe_response_text(response)

            if not text:
                return _fallback_payload(
                    q,
                    steps_out,
                    prior_error="模型未返回有效文本（可能被安全策略拦截或候选为空）",
                    quota_exceeded=False,
                )

            return {
                "answer": text,
                "mode": "gemini",
                "steps": list(steps_out),
                "sop_links": [],
                "error": None,
                "quota_exceeded": False,
            }

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "not supported" in msg:
                logger.info("run_agent 模型不可用，尝试下一个: %s -> %s", model_name, e)
                continue
            logger.warning("run_agent Gemini 路径失败，进入 fallback: %s", e, exc_info=True)
            quota = _detect_quota_error(e)
            return _fallback_payload(q, steps_out, prior_error=str(e), quota_exceeded=quota)

    if last_err is not None:
        quota = _detect_quota_error(last_err)
        return _fallback_payload(q, steps_out, prior_error=str(last_err), quota_exceeded=quota)

    return _fallback_payload(q, steps_out, prior_error="无可用 Gemini 模型", quota_exceeded=False)
