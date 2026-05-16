"""SOP 文件读取工具 readFile（仅 data/ 下 sop-xxx.html）。"""

from __future__ import annotations

import re
from pathlib import Path

from services.env_bootstrap import PROJECT_ROOT
from utils.html_processor import extract_title_and_body

DATA_DIR = PROJECT_ROOT / "data"
_ALLOWED_FNAME = re.compile(r"^sop-\d{3}\.html$", re.IGNORECASE)

# 先验：用户描述 -> 优先尝试的文件（顺序即优先级）
_PRIOR_RULES: list[tuple[tuple[str, ...], list[str]]] = [
    (("p0", "P0", "重大故障", "全线", "事故响应"), ["sop-001.html", "sop-004.html"]),
    (("内存", "OOM", "oom", "溢出", "进程崩溃", "超时", "后端", "服务挂了", "服务器挂了"), ["sop-001.html"]),
    (("慢查询", "数据库", "DBA", "主从", "binlog", "Binlog", "连接池", "mysql", "redis"), ["sop-002.html"]),
    (("白屏", "前端", "页面加载", "CDN", "静态资源", "H5", "WebView"), ["sop-003.html"]),
    (("K8s", "k8s", "kubernetes", "集群", "SRE", "监控告警", "容量", "节点"), ["sop-004.html"]),
    (("安全", "入侵", "漏洞", "攻击", "勒索", "泄露"), ["sop-005.html"]),
    (("数据管道", "ETL", "Spark", "Flink", "数据平台", "hive"), ["sop-006.html"]),
    (("App崩溃", "崩溃率", "热修复", "移动端", "iOS", "Android", "发版"), ["sop-007.html"]),
    (("推荐", "模型推理", "GPU", "算法", "AB实验", "推理延迟"), ["sop-008.html"]),
    (("测试环境", "自动化测试", "QA", "发版卡点"), ["sop-009.html"]),
    (("CDN", "DNS", "DDoS", "网络故障", "带宽"), ["sop-010.html"]),
]


def normalize_fname(fname: str) -> str:
    """只保留 basename，统一为小写编号形式 sop-xxx.html。"""
    base = Path(fname.strip()).name
    m = re.match(r"^(sop-\d{3})\.html$", base, re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}.html"
    return base


def predict_sop_files(user_text: str, max_files: int = 2) -> list[str]:
    """根据用户描述预测 1–2 个最可能的 SOP 文件名（与 resolve_read_targets 使用同一先验表）。"""
    text = user_text or ""
    picked: list[str] = []
    seen: set[str] = set()

    for keywords, files in _PRIOR_RULES:
        if any(k in text for k in keywords):
            for f in files:
                fn = normalize_fname(f)
                if fn not in seen and _ALLOWED_FNAME.match(fn):
                    seen.add(fn)
                    picked.append(fn)
                    if len(picked) >= max_files:
                        return picked

    # 默认：后端 + SRE 通用入口
    for f in ("sop-001.html", "sop-004.html"):
        if len(picked) >= max_files:
            break
        if f not in seen:
            seen.add(f)
            picked.append(f)
    return picked[:max_files]


def resolve_read_targets(user_text: str) -> tuple[list[str], str]:
    """
    决定实际 readFile 的文件列表：单一领域命中时只读 1 个以省配额；P0/多领域最多 2 个。

    返回 (文件列表, 人类可读说明)。
    """
    text = user_text or ""

    p0_kw, p0_files = _PRIOR_RULES[0]
    if any(k in text for k in p0_kw):
        return list(p0_files)[:2], "命中 P0/重大故障先验：读取分级与 SRE 两份 SOP"

    primaries: list[str] = []
    for keywords, files in _PRIOR_RULES[1:]:
        if any(k in text for k in keywords):
            fn = normalize_fname(files[0])
            if _ALLOWED_FNAME.match(fn):
                primaries.append(fn)

    seen: set[str] = set()
    ordered: list[str] = []
    for p in primaries:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    if len(ordered) == 1:
        return ordered, "问题聚焦单一领域：仅读取最相关 1 个 SOP 以节省 API 配额"
    if len(ordered) >= 2:
        return ordered[:2], "命中多个领域：最多读取 2 个 SOP"
    return ["sop-001.html"], "未命中具体先验关键词：默认仅读取后端通用 SOP（1 个）"


def read_file(fname: str) -> dict:
    """
    readFile 工具：读取 data/ 下单个 SOP HTML，返回标题与正文纯文本。
    """
    base = normalize_fname(fname)
    if not _ALLOWED_FNAME.match(base):
        return {
            "ok": False,
            "fname": base,
            "error": f"非法文件名，仅允许 sop-001.html … sop-010.html，收到: {fname!r}",
        }

    path = (DATA_DIR / base).resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError:
        return {"ok": False, "fname": base, "error": "路径越界，拒绝读取"}

    if not path.is_file():
        return {"ok": False, "fname": base, "error": f"文件不存在: data/{base}"}

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        return {"ok": False, "fname": base, "error": str(e)}

    title, body = extract_title_and_body(raw)
    return {
        "ok": True,
        "fname": base,
        "title": title,
        "body": body,
        "body_chars": len(body),
    }
