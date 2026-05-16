"""v2 搜索：混合召回（各 Top5）+ Gemini 精排；精排失败时用规则/混合分兜底，保证 1–2 条结果。"""

from __future__ import annotations

from dataclasses import dataclass

from services.document_store import get_document
from services.gemini_rerank import MissingGeminiKeyError, gemini_rerank_sops_json
from services.keyword_search import keyword_search
from services.vector_index import semantic_search

_KEYWORD_TOP = 5
_VECTOR_TOP = 5
_MIN_POOL = 5
_MAX_BACKFILL_KW = 40
_MAX_BACKFILL_VEC = 30
_KEYWORD_WEIGHT = 0.4
_VECTOR_WEIGHT = 0.6
_MAX_RESULTS = 2


@dataclass
class V2SearchHit:
    id: str
    title: str
    snippet: str
    score: float
    reason: str | None = None


def _hybrid_display_score(keyword_matched: bool, vector_score: float) -> float:
    s = (_KEYWORD_WEIGHT if keyword_matched else 0.0) + float(vector_score) * _VECTOR_WEIGHT
    return max(s, 0.55)


def _enrich_candidate(doc_id: str, title: str, snippet: str, vector_score: float, kw_hit: bool) -> dict:
    row = get_document(doc_id)
    body = (row or {}).get("body", "") or ""
    excerpt = " ".join(body.split())
    if len(excerpt) > 800:
        excerpt = excerpt[:800] + "…"
    disp_title = (row or {}).get("title") or title
    disp_snip = snippet or excerpt[:280]
    return {
        "id": doc_id,
        "title": disp_title,
        "snippet": disp_snip,
        "vector_score": float(vector_score),
        "keyword_matched": kw_hit,
        "excerpt_for_llm": excerpt or disp_snip,
    }


def collect_hybrid_candidates_at_least_5(query: str) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []

    kw_all = keyword_search(q)
    vec_all = semantic_search(q, k=max(_VECTOR_TOP, _MAX_BACKFILL_VEC))

    seen: set[str] = set()
    pool: list[dict] = []

    for h in kw_all[:_KEYWORD_TOP]:
        if h.id in seen:
            continue
        seen.add(h.id)
        vs = next((v[3] for v in vec_all if v[0] == h.id), 0.0)
        pool.append(_enrich_candidate(h.id, h.title, h.snippet, vs, True))

    for doc_id, title, snip, vscore in vec_all[:_VECTOR_TOP]:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        pool.append(_enrich_candidate(doc_id, title, snip, vscore, False))

    if len(pool) < _MIN_POOL:
        for h in kw_all[_KEYWORD_TOP:_MAX_BACKFILL_KW]:
            if h.id in seen:
                continue
            seen.add(h.id)
            vs = next((v[3] for v in vec_all if v[0] == h.id), 0.0)
            pool.append(_enrich_candidate(h.id, h.title, h.snippet, vs, True))
            if len(pool) >= _MIN_POOL:
                break

    if len(pool) < _MIN_POOL:
        for doc_id, title, snip, vscore in vec_all[_VECTOR_TOP:_MAX_BACKFILL_VEC]:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            pool.append(_enrich_candidate(doc_id, title, snip, vscore, False))
            if len(pool) >= _MIN_POOL:
                break

    return pool


def _build_candidates_data_for_prompt(candidates: list[dict]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(candidates, start=1):
        blocks.append(
            f"--- 候选 [{i}] ---\n"
            f"ID: {c['id']}\n"
            f"标题: {c['title']}\n"
            f"内容摘要/摘录:\n{c.get('excerpt_for_llm', c.get('snippet', ''))}\n"
        )
    return "\n".join(blocks)


def _hits_from_gemini_ranked(
    ranked: list[dict],
    cand_by_id: dict[str, dict],
) -> list[V2SearchHit]:
    hits: list[V2SearchHit] = []
    for item in ranked:
        sid = item["id"]
        c = cand_by_id.get(sid)
        if not c:
            continue
        hits.append(
            V2SearchHit(
                id=sid,
                title=c["title"],
                snippet=c["snippet"],
                score=float(item["confidence"]),
                reason=item.get("reason") or None,
            )
        )
    return hits[:_MAX_RESULTS]


def _rule_based_hits(query: str, candidates: list[dict], cand_by_id: dict[str, dict]) -> list[V2SearchHit]:
    """与 Prompt 中强制路由一致的本地理赔，Gemini 不可用或解析失败时使用。"""
    q = query
    ql = q.lower()
    ordered_ids: list[tuple[str, str, float]] = []

    def want(*keywords: str) -> bool:
        return any(k in q or k.lower() in ql for k in keywords)

    if want("内存", "OOM", "oom", "崩溃", "进程崩溃"):
        ordered_ids.append(("sop-001", "规则匹配：内存/OOM/崩溃 → 后端服务 SOP", 0.85))
    if want("慢查询", "数据库", "主从", "Binlog", "binlog", "同步延迟"):
        ordered_ids.append(("sop-002", "规则匹配：数据库/主从/慢查询 → DBA SOP", 0.85))
    if want("服务器挂了", "K8s", "k8s", "kubernetes", "集群"):
        ordered_ids.append(("sop-001", "规则匹配：服务器/K8s → 后端服务 SOP", 0.8))
        ordered_ids.append(("sop-004", "规则匹配：服务器/K8s → SRE SOP", 0.7))

    hits: list[V2SearchHit] = []
    seen: set[str] = set()
    for sop_id, reason, score in ordered_ids:
        if sop_id in seen or sop_id not in cand_by_id:
            continue
        seen.add(sop_id)
        c = cand_by_id[sop_id]
        hits.append(
            V2SearchHit(
                id=sop_id,
                title=c["title"],
                snippet=c["snippet"],
                score=score,
                reason=reason,
            )
        )
        if len(hits) >= _MAX_RESULTS:
            return hits
    return hits


def _hybrid_fallback_hits(candidates: list[dict]) -> list[V2SearchHit]:
    """按混合分取 Top 1–2，保证总有可见结果。"""
    ranked = sorted(
        candidates,
        key=lambda c: (
            -_hybrid_display_score(c["keyword_matched"], c["vector_score"]),
            -c["vector_score"],
            c["id"],
        ),
    )
    hits: list[V2SearchHit] = []
    for c in ranked[:_MAX_RESULTS]:
        score = _hybrid_display_score(c["keyword_matched"], c["vector_score"])
        hits.append(
            V2SearchHit(
                id=c["id"],
                title=c["title"],
                snippet=c["snippet"],
                score=score,
                reason="混合检索回退（关键词 + 向量相似度）",
            )
        )
    return hits


def run_v2_search(query: str) -> tuple[list[V2SearchHit], str | None]:
    """
    返回 (结果列表, message)。有候选时尽量返回 1–2 条；仅无候选或缺少 API Key 时 results 为空。
    """
    q = (query or "").strip()
    if not q:
        return [], None

    candidates = collect_hybrid_candidates_at_least_5(q)
    if not candidates:
        return [], None

    cand_by_id = {c["id"]: c for c in candidates}
    data_for_prompt = _build_candidates_data_for_prompt(candidates)
    print("=== v2/search 发送给 Gemini 的完整候选名单 ===", flush=True)
    print(data_for_prompt, flush=True)
    print("=== 候选结束 ===", flush=True)

    hits: list[V2SearchHit] = []
    gemini_error: str | None = None

    try:
        ranked = gemini_rerank_sops_json(q, candidates, candidates_data=data_for_prompt)
        hits = _hits_from_gemini_ranked(ranked, cand_by_id)
    except MissingGeminiKeyError as e:
        gemini_error = str(e)
    except Exception as e:
        print(f"[v2/search] Gemini 调用异常: {e}", flush=True)
        gemini_error = f"精排异常，已使用本地回退: {e}"

    if not hits:
        hits = _rule_based_hits(q, candidates, cand_by_id)

    if not hits:
        hits = _hybrid_fallback_hits(candidates)
        print("[v2/search] 使用混合检索 Top2 回退", flush=True)

    return hits[:_MAX_RESULTS], gemini_error
