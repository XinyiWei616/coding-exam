"""启动时从 data/ 目录加载 HTML，写入内存并建立向量索引。"""

from __future__ import annotations

from pathlib import Path

from services.document_store import DOCUMENT_ID_RE, put_document
from services.vector_index import upsert_document_embedding
from utils.html_processor import extract_title_and_body

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def bootstrap_from_data_dir() -> int:
    """
    扫描 data/ 下所有 .html，将文件名 stem 作为 id 入库。
    跳过不符合 DOCUMENT_ID_RE 的文件名。
    返回成功索引的文件数量。
    """
    if not DATA_DIR.is_dir():
        return 0

    n = 0
    for path in sorted(DATA_DIR.rglob("*.html")):
        stem = path.stem
        if not DOCUMENT_ID_RE.fullmatch(stem):
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not raw.strip():
            continue
        title, body = extract_title_and_body(raw)
        put_document(stem, title, body)
        upsert_document_embedding(stem, title, body)
        n += 1
    return n
