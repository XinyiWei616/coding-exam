"""Gemini 精排：google.generativeai + JSON 输出（v2/search）。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import google.generativeai as genai

from services.env_bootstrap import gemini_api_key_from_env

# 与 data/ 下 SOP 对齐的领域提示（供模型推理）
# 硬编码关键词 → SOP 映射（精排时必须遵守；候选 ID 一般为 sop-001 形式，与 sop-001.html 等价）
_FORCED_KEYWORD_ROUTE_RULES = """
【强制路由规则】（用户故障描述含下列关键词时，必须按映射优先选择对应 SOP；JSON 中的 id 必须与下方「候选列表」里出现的 ID 完全一致，通常为 sop-001、sop-002…无 .html 后缀）
1) 若描述含「内存」或「OOM」或「崩溃」→ 必须优先匹配 sop-001.html（后端服务，OOM/进程类）。
2) 若描述含「慢查询」或「数据库」或「主从」→ 必须优先匹配 sop-002.html（数据库 DBA）。
3) 若描述含「服务器挂了」或「K8s」→ 必须同时考虑 sop-001.html（后端服务）与 sop-004.html（SRE）；若候选中两者均存在，JSON 中可返回 1～2 条（置信度与理由分别说明）。
"""

_SOP_DOMAIN_TABLE = """参考表格（文件名 → 部门 → 关键内容）：
sop-001.html | 后端服务 | OOM 排查、服务超时、降级策略、故障分级
sop-002.html | 数据库 DBA | 主从延迟、慢查询、连接池满、数据恢复
sop-003.html | 前端 | 页面白屏、CDN 资源加载失败、兼容性、性能劣化
sop-004.html | SRE | K8s 集群问题、监控告警、容量规划、故障响应
sop-005.html | 安全团队 | 安全事件分级、入侵检测、漏洞响应
sop-006.html | 数据平台 | 数据管道故障、ETL 失败、Spark 集群
sop-007.html | 移动端 | App 崩溃率、热修复、推送服务
sop-008.html | AI & 算法 | 模型推理延迟、推荐质量下降、GPU 集群
sop-009.html | QA | 测试环境故障、自动化测试、发版卡点
sop-010.html | 网络 & CDN | CDN 节点故障、DNS 异常、DDoS 防护"""


class MissingGeminiKeyError(RuntimeError):
    """未配置 GEMINI_API_KEY。"""


# 默认使用 Gemini 1.5 Flash；若当前 Key/区域不可用则按列表回退到 2.x
_DEFAULT_RERANK_MODEL = "gemini-1.5-flash"
_FALLBACK_RERANK_MODELS = (
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash-001",
)


def _rerank_model_candidates() -> list[str]:
    """环境变量指定模型优先，其后按回退列表去重。"""
    configured = (os.environ.get("GEMINI_V2_RERANK_MODEL") or "").strip()
    out: list[str] = []
    if configured:
        out.append(configured)
    for m in (_DEFAULT_RERANK_MODEL, *_FALLBACK_RERANK_MODELS):
        if m not in out:
            out.append(m)
    return out


def _generate_rerank_content(genai: Any, prompt: str, generation_config: Any) -> str:
    """按候选模型依次调用 generateContent，404 时自动换下一个模型。"""
    last_err: Exception | None = None
    for name in _rerank_model_candidates():
        try:
            model = genai.GenerativeModel(name)
            response = model.generate_content(prompt, generation_config=generation_config)
            return (response.text or "").strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "not supported" in msg:
                print(f"[gemini_rerank] 模型不可用，尝试下一个: {name} -> {e}", flush=True)
                continue
            raise
    if last_err is not None:
        raise last_err
    return ""


def _normalize_sop_id(raw: str) -> str:
    s = (raw or "").strip()
    if s.lower().endswith(".html"):
        s = s[:-5]
    return s.strip()


def _extract_json_array(text: str) -> list[Any] | None:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def gemini_rerank_sops_json(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    candidates_data: str,
) -> list[dict[str, Any]]:
    """
    调用 Gemini 精排，返回 0–2 条 {id, confidence, reason}（id 须在候选中；低置信度也保留，由上层兜底）。
    解析失败返回空列表，由 v2_search_flow 做混合/规则回退。
    """
    key = gemini_api_key_from_env()
    if not key:
        raise MissingGeminiKeyError("需要配置 GEMINI_API_KEY（或 GOOGLE_API_KEY）")

    genai.configure(api_key=key)

    prompt = f"""你是一个专业的 SRE On-Call 专家。现有用户故障描述："{query}"。
以下是系统检索到的候选 SOP 列表（按 ID、标题、内容摘要排列）：
{candidates_data}

{_FORCED_KEYWORD_ROUTE_RULES}

{_SOP_DOMAIN_TABLE}

并基于你的专业知识进行推理；若用户描述同时命中多条强制规则，以强制规则优先级覆盖一般向量相似度。另：'空间/Binlog/同步' 类问题仍应对应数据库 DBA 相关 SOP。

任务：
1. 严格遵守上方【强制路由规则】，并参考表格与候选摘要，从候选列表中选出最匹配的 1-2 个 SOP（id 必须与候选列表中的 ID 完全一致）。
2. 必须按以下 JSON 格式返回，不要有任何多余文字：
[{{"id": "sop-xxx", "confidence": 0.9, "reason": "理由"}}, ...]"""

    generation_config = genai.types.GenerationConfig(
        temperature=0.0,
        max_output_tokens=2048,
    )
    try:
        raw = _generate_rerank_content(genai, prompt, generation_config)
    except Exception as e:
        print(f"[gemini_rerank] generate_content 失败: {e}", flush=True)
        raw = ""

    if not raw:
        print("[gemini_rerank] 模型返回为空，将走本地回退", flush=True)
    elif not _extract_json_array(raw):
        print(f"[gemini_rerank] 无法解析 JSON，原始片段: {raw[:400]!r}", flush=True)

    parsed = _extract_json_array(raw)
    if not parsed:
        return []

    valid_ids = {_normalize_sop_id(str(c["id"])) for c in candidates if c.get("id")}
    canon_by_raw: dict[str, str] = {}
    for c in candidates:
        rid = str(c["id"])
        canon_by_raw[_normalize_sop_id(rid)] = rid

    picked: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        sid_raw = str(item.get("id", "")).strip()
        sid = _normalize_sop_id(sid_raw)
        if sid not in valid_ids:
            continue
        try:
            conf = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        # 不再丢弃低分项；展示分下限 0.55，避免接口侧 score 过低
        conf = max(conf, 0.55) if conf > 0 else 0.6
        reason = str(item.get("reason", "") or "").strip()
        display_id = canon_by_raw.get(sid, sid_raw or sid)
        picked.append({"id": display_id, "confidence": conf, "reason": reason})

    picked.sort(key=lambda x: -x["confidence"])
    return picked[:2]
