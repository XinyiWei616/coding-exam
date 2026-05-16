"""SRE On-Call 对话 Agent：预测 SOP → readFile → Gemini 综合回答。"""

from __future__ import annotations

from dataclasses import dataclass, field

import google.generativeai as genai

from services.env_bootstrap import gemini_api_key_from_env
from services.gemini_rerank import _generate_rerank_content
from services.sop_tools import predict_sop_files, read_file, resolve_read_targets

_MAX_BODY_IN_PROMPT = 8000

_RESPONSE_GUIDE = """## 响应指南 (CRITICAL)
1. **禁止全文打印**：除非用户明确要求「输出全文/原文」，否则严禁复述或粘贴 SOP 的原始长文本。
2. **结构化回复**：回答必须严格包含以下 Markdown 小节（无内容可写「文档未提及」）：
   - **故障诊断**：一句话说明当前问题对应 SOP 中的哪类场景/章节。
   - **立即行动**：列出前 3 条必须执行的命令或操作（如回滚、扩容、保存 Dump、止血开关等），用有序列表。
   - **排查建议**：应联系或协同的团队/角色，以及需要重点查看的核心监控指标或平台。
   - **风险预警**：根据 SOP 列出绝对禁止或高风险操作（如未灰度全量、线上 DDL 等）。"""


@dataclass
class ToolCallRecord:
    tool: str
    fname: str
    ok: bool
    error: str | None = None
    title: str | None = None
    body_preview: str | None = None


@dataclass
class SreAgentResult:
    answer: str
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)


def _try_read_with_fallback(primary_files: list[str], thoughts: list[str]) -> tuple[list[dict], list[ToolCallRecord]]:
    """按序 readFile；全部失败时再尝试少量邻近 SOP。"""
    loaded: list[dict] = []
    records: list[ToolCallRecord] = []

    def attempt(fname: str) -> None:
        result = read_file(fname)
        preview = ""
        if result.get("ok"):
            body = result.get("body") or ""
            preview = body[:200] + ("…" if len(body) > 200 else "")
        records.append(
            ToolCallRecord(
                tool="readFile",
                fname=result.get("fname", fname),
                ok=bool(result.get("ok")),
                error=result.get("error"),
                title=result.get("title"),
                body_preview=preview or None,
            )
        )
        thoughts.append(
            f"工具 readFile(\"{fname}\") -> "
            + ("成功" if result.get("ok") else f"失败: {result.get('error')}")
        )
        if result.get("ok"):
            loaded.append(result)

    for f in primary_files:
        attempt(f)

    if not loaded:
        thoughts.append("首次读取均失败，尝试备用 SOP …")
        for f in ("sop-001.html", "sop-002.html"):
            if f not in primary_files:
                attempt(f)
            if loaded:
                break

    return loaded, records


def _build_sop_context(docs: list[dict]) -> str:
    parts: list[str] = []
    for d in docs:
        body = d.get("body") or ""
        if len(body) > _MAX_BODY_IN_PROMPT:
            body = body[:_MAX_BODY_IN_PROMPT] + "\n…（正文已截断）"
        parts.append(
            f"### 文件: {d['fname']}\n标题: {d.get('title', '')}\n\n{body}\n"
        )
    return "\n".join(parts)


def _gemini_answer(user_message: str, context: str, thoughts: list[str]) -> str:
    key = gemini_api_key_from_env()
    if not key:
        thoughts.append("未配置 GEMINI_API_KEY，使用本地结构化摘要")
        return _local_fallback_answer(user_message, context)

    genai.configure(api_key=key)
    prompt = f"""## Role
你是资深 SRE On-Call 专家，通过对话协助用户排查系统故障。

{_RESPONSE_GUIDE}

## 内部材料（仅供你提炼，禁止对用户输出原文大段）
{context}

## 用户问题
{user_message}

## 其它要求
- 仅依据内部材料中的 SOP 提炼结论；材料未写明的请写「文档未提及」。
- 若同时涉及分级与基础设施响应，请保证步骤与两份 SOP 一致、无矛盾。
- 不要编造材料中不存在的命令、链接或联系人。"""

    generation_config = genai.types.GenerationConfig(
        temperature=0.2,
        max_output_tokens=4096,
    )
    try:
        raw = _generate_rerank_content(genai, prompt, generation_config)
        if raw.strip():
            thoughts.append("已调用 Gemini 生成结构化回答")
            return raw.strip()
    except Exception as e:
        thoughts.append(f"Gemini 生成失败: {e}，改用本地结构化摘要")

    return _local_fallback_answer(user_message, context)


def _local_fallback_answer(user_message: str, context: str) -> str:
    if not context.strip():
        return (
            "未能读取任何 SOP 文件。请检查 `data/` 下是否存在 `sop-001.html` 等文件，"
            "或换一种方式描述故障。"
        )
    titles = []
    for block in context.split("### 文件:")[1:]:
        sub = block.split("标题:", 1)
        if len(sub) > 1:
            titles.append(sub[1].split("\n", 1)[0].strip())
    title_hint = " / ".join(titles[:2]) if titles else "已加载 SOP"

    return f"""## 故障诊断
已根据本地 SOP（{title_hint}）做离线摘要；未接大模型，细节可能不完整。用户问题与「{user_message[:120]}」相关（请对照 SOP 目录自行核对小节）。

## 立即行动
1. 打开对应 SOP 文档，定位「常见故障处理」与当前现象最贴近的一节。  
2. 按文档中的告警分级先止血（限流/降级/扩容/回滚择一，以 SOP 为准）。  
3. 保存现场证据（日志片段、监控截图、heap dump 路径等），便于复盘。

## 排查建议
- 按 SOP「值班职责 / 升级路径」联系对口值班 Owner 或二级负责人。  
- 对照 SOP「监控指标」章节核对核心大盘（延迟、错误率、资源、复制延迟等）。

## 风险预警
- 未经 SOP 明确授权的操作（如生产直接全量 DDL、绕过变更窗口等）一律禁止；以文档「禁止操作」为准。

---
*以下为模型不可用时的内部摘录（非完整原文，仅供人工跳转查阅）：*

{context[:1200]}{"…" if len(context) > 1200 else ""}
"""


def run_sre_agent(user_message: str) -> SreAgentResult:
    """分析意图 → 锁定 readFile 列表（单文件优先）→ 综合回答。"""
    msg = (user_message or "").strip()
    thoughts: list[str] = []
    if not msg:
        return SreAgentResult(
            answer="请描述当前故障现象、影响范围与已观察到的告警信息。",
            thoughts=["用户输入为空"],
        )

    thoughts.append("1. 分析意图：识别故障所属领域（后端/DB/前端/SRE/安全等）。")
    predicted_preview = predict_sop_files(msg, max_files=2)
    to_read, quota_note = resolve_read_targets(msg)
    thoughts.append(f"2. 先验预测（参考）：{', '.join(predicted_preview)}")
    thoughts.append(f"3. 读取策略：{quota_note} → 实际 readFile：{', '.join(to_read)}")

    docs, tool_records = _try_read_with_fallback(to_read, thoughts)
    thoughts.append(f"4. 已成功读取 {len(docs)} 个 SOP 文件。")

    context = _build_sop_context(docs)
    thoughts.append("5. 按响应指南生成回答（禁止全文打印）。")
    answer = _gemini_answer(msg, context, thoughts)

    return SreAgentResult(
        answer=answer,
        thoughts=thoughts,
        tool_calls=tool_records,
        files_read=[d["fname"] for d in docs],
    )
