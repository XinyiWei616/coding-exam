"""Chroma 向量索引：可切换中文本地嵌入 / OpenAI，分段入库与标题加权检索。

环境变量：
- EMBEDDING_BACKEND：auto（默认，有 OPENAI_API_KEY 则用 OpenAI，否则本地下载中文模型）、
  huggingface（shibing624/text2vec-base-chinese）、openai（text-embedding-3-small）。
- OPENAI_API_KEY：选择 OpenAI 嵌入时必填。
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from services.document_store import get_document
from utils.semantic_expansion import build_semantic_expansion_hints, embed_text_with_title
from utils.text_chunking import chunk_plain_text

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "chroma_data"
COLLECTION_NAME = "oncall_docs"

_embeddings: Embeddings | None = None
_vectorstore: Chroma | None = None

_SNIPPET_MAX_LEN = 280
# 标题专用向量块在相似度上的乘性加权（用于排序与最终 score）
_TITLE_SEGMENT_BOOST = 1.32
# 多取候选块再在文档维度聚合，避免细粒度切块后排不到文档级 TopK
_SEMANTIC_CANDIDATE_CHUNKS = 32


def _embedding_backend() -> str:
    raw = os.environ.get("EMBEDDING_BACKEND", "auto").strip().lower()
    if raw in ("auto", ""):
        return "openai" if os.environ.get("OPENAI_API_KEY") else "huggingface"
    if raw in ("huggingface", "hf", "local"):
        return "huggingface"
    if raw in ("openai", "openai_api"):
        return "openai"
    raise ValueError(
        "EMBEDDING_BACKEND 须为 auto、huggingface 或 openai，当前为: " + repr(raw)
    )


def _make_embeddings() -> Embeddings:
    backend = _embedding_backend()
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "使用 OpenAI 嵌入（EMBEDDING_BACKEND=openai 或 auto 且已配置密钥）"
                "需要环境变量 OPENAI_API_KEY"
            )
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")


def _recreate_chroma_store() -> None:
    global _vectorstore
    if _embeddings is None:
        raise RuntimeError("嵌入模型尚未初始化")
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def init_vector_index() -> None:
    """加载嵌入模型并创建 Chroma 集合句柄。"""
    global _embeddings, _vectorstore
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _embeddings = _make_embeddings()
    _recreate_chroma_store()


def _require_vectorstore() -> Chroma:
    if _vectorstore is None:
        raise RuntimeError("向量库尚未初始化，请确认应用已完成 startup / lifespan")
    return _vectorstore


def clear_all_vectors() -> None:
    """清空集合中全部向量（保留集合，避免 delete_collection 后句柄失效）。"""
    vs = _require_vectorstore()
    try:
        batch = vs.get(include=[])
        ids = batch.get("ids") or []
        if ids:
            vs.delete(ids=list(ids))
    except Exception:
        pass


def distance_to_cosine_similarity(distance: float) -> float:
    """
    Chroma 在 hnsw:space=cosine 下返回的余弦距离 d 与余弦相似度 s 满足 s ≈ 1 - d。
    将分数限制在 [0, 1] 便于对外展示。
    """
    s = 1.0 - float(distance)
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def _snippet_for_result(doc_id: str, page_content: str) -> str:
    """优先用内存中的正文做摘要；否则退回向量块文本。"""
    row = get_document(doc_id)
    body = (row or {}).get("body", "") or ""
    text = " ".join(body.split())
    if not text:
        text = " ".join((page_content or "").split())
    if len(text) > _SNIPPET_MAX_LEN:
        return text[:_SNIPPET_MAX_LEN] + "…"
    return text


def _truncate_snippet(text: str) -> str:
    t = " ".join(text.split())
    if len(t) > _SNIPPET_MAX_LEN:
        return t[:_SNIPPET_MAX_LEN] + "…"
    return t


def _delete_doc_chunks(vs: Chroma, doc_id: str) -> None:
    try:
        prev = vs.get(where={"doc_id": doc_id}, include=[])
        pids = prev.get("ids") or []
        if pids:
            vs.delete(ids=list(pids))
        return
    except Exception:
        pass
    batch = vs.get(include=["metadatas"])
    ids_all = batch.get("ids") or []
    metas = batch.get("metadatas") or []
    to_del = [
        i
        for i, m in zip(ids_all, metas)
        if (m or {}).get("doc_id") == doc_id or (m or {}).get("id") == doc_id
    ]
    if to_del:
        vs.delete(ids=to_del)


def upsert_document_embedding(doc_id: str, title: str, body: str) -> None:
    """按段落/300 字切块写入；含标题加权专用块。元数据含 doc_id、title、segment、chunk_index。"""
    vs = _require_vectorstore()
    _delete_doc_chunks(vs, doc_id)

    body_chunks = chunk_plain_text(body or "", max_chars=300)
    docs: list[Document] = []
    ids: list[str] = []

    title_line = (title or "").strip() or doc_id
    hints = build_semantic_expansion_hints(title_line, body or "")

    title_block = embed_text_with_title(
        title_line,
        hints,
        f"文档编号: {doc_id}",
    )
    docs.append(
        Document(
            page_content=title_block,
            metadata={
                "doc_id": doc_id,
                "id": doc_id,
                "title": title_line,
                "segment": "title_header",
                "chunk_index": -1,
            },
        )
    )
    ids.append(f"{doc_id}::title")

    for i, chunk in enumerate(body_chunks):
        body_text = embed_text_with_title(title_line, hints, chunk)
        docs.append(
            Document(
                page_content=body_text,
                metadata={
                    "doc_id": doc_id,
                    "id": doc_id,
                    "title": title_line,
                    "segment": "body",
                    "chunk_index": i,
                },
            )
        )
        ids.append(f"{doc_id}::b::{i}")

    if docs:
        vs.add_documents(docs, ids=ids)


def semantic_search(query: str, k: int = 3) -> list[tuple[str, str, str, float]]:
    """
    语义检索 Top-k（按文档聚合）。
    返回 (id, title, snippet, score)；score 为标题加权后的综合分，上限 1。
    """
    q = query.strip()
    if not q:
        return []
    vs = _require_vectorstore()
    pairs = vs.similarity_search_with_score(q, k=_SEMANTIC_CANDIDATE_CHUNKS)

    # doc_id -> list of (boosted_score, raw_sim, page_content, segment)
    grouped: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    title_by_doc: dict[str, str] = {}

    for doc, distance in pairs:
        meta: dict[str, Any] = doc.metadata or {}
        doc_id = str(meta.get("doc_id") or meta.get("id", ""))
        if not doc_id:
            continue
        title_m = str(meta.get("title", "")).strip()
        if title_m:
            title_by_doc[doc_id] = title_m
        segment = str(meta.get("segment", "body"))
        raw_sim = distance_to_cosine_similarity(distance)
        boost = _TITLE_SEGMENT_BOOST if segment == "title_header" else 1.0
        boosted = min(1.0, raw_sim * boost)
        grouped[doc_id].append((boosted, raw_sim, doc.page_content or "", segment))

    ranked: list[tuple[str, str, str, float]] = []
    for doc_id, entries in grouped.items():
        best_boosted = max(e[0] for e in entries)
        best_entry = max(entries, key=lambda e: e[0])
        _bb, _rs, page_content, segment = best_entry
        row = get_document(doc_id)
        title = (row or {}).get("title") or title_by_doc.get(doc_id, "")
        if segment == "title_header" and row and (row.get("body") or "").strip():
            snippet = _snippet_for_result(doc_id, page_content)
        else:
            snippet = (
                _truncate_snippet(page_content)
                if page_content.strip()
                else _snippet_for_result(doc_id, page_content)
            )
        ranked.append((doc_id, title, snippet, best_boosted))

    ranked.sort(key=lambda x: (-x[3], x[0]))
    return ranked[:k]
