"""Phase 1 关键词检索（供 v1 与 v2 流水线复用）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.document_store import iter_documents


@dataclass
class KeywordHit:
    id: str
    title: str
    snippet: str
    score: float


def _snippet_around(text: str, start: int, end: int, context: int = 30) -> str:
    lo = max(0, start - context)
    hi = min(len(text), end + context)
    return text[lo:hi]


def _count_overlaps(pattern: re.Pattern[str], text: str) -> int:
    n = 0
    pos = 0
    for m in pattern.finditer(text):
        if m.start() >= pos:
            n += 1
            pos = m.end()
    return n


def _search_in_field(pattern: re.Pattern[str], field_text: str) -> tuple[int, re.Match[str] | None]:
    count = _count_overlaps(pattern, field_text)
    first = pattern.search(field_text)
    return count, first


def keyword_search(query: str) -> list[KeywordHit]:
    q = (query or "").strip()
    if not q:
        return []
    try:
        pattern = re.compile(re.escape(q), re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"无法构建查询模式: {e}") from e

    results: list[KeywordHit] = []
    for doc_id, doc in iter_documents():
        title = doc["title"]
        body = doc["body"]
        title_count, title_match = _search_in_field(pattern, title)
        body_count, body_match = _search_in_field(pattern, body)
        total = title_count + body_count
        if total == 0:
            continue
        if body_match:
            sn = _snippet_around(body, body_match.start(), body_match.end())
        elif title_match:
            sn = _snippet_around(title, title_match.start(), title_match.end())
        else:
            sn = ""
        results.append(KeywordHit(id=doc_id, title=title, snippet=sn, score=float(total)))

    results.sort(key=lambda h: (-h.score, h.title))
    return results
