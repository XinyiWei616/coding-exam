"""根据标题与正文做轻量规则识别，生成写入向量库的「检索增强」提示词。"""

from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def build_semantic_expansion_hints(title: str, body: str) -> str:
    """
    基于全文规则匹配，返回可拼入向量文本的一行「相关检索词」内容；无匹配时返回空串。
    """
    raw = f"{title}\n{body}"
    text = raw
    compact = _norm(raw)

    phrases: list[str] = []

    # 系统级 / 基础设施（用户示例：服务器、后端、基础设施）
    system_kw = (
        "系统级",
        "基础设施",
        "全线",
        "大面积故障",
        "核心链路",
        "机房",
        "网络故障",
        "服务不可用",
        "容灾",
        "降级预案",
        "可用性",
        "监控大盘",
        "告警风暴",
    )
    if any(k in text for k in system_kw) or any(
        x in compact for x in ("p0", "p1", "sre", "基础设施", "系统故障")
    ):
        phrases.append("服务器、后端、基础设施、系统级故障、线上服务、监控告警")

    if any(
        k in text
        for k in (
            "数据库",
            "DBA",
            "MySQL",
            "Redis",
            "主从",
            "慢查询",
            "连接池",
            "备份恢复",
        )
    ):
        phrases.append("数据库、存储、主从同步、备份、连接与性能")

    if any(
        k in text
        for k in ("前端", "H5", "CDN", "白屏", "浏览器", "WebView", "静态资源")
    ):
        phrases.append("前端、Web、CDN、页面加载、JavaScript、用户体验")

    if any(
        k in text
        for k in ("客户端", "iOS", "Android", "小程序", "App", "发版", "应用商店")
    ):
        phrases.append("移动端、客户端、应用发布、崩溃监控")

    if any(
        k in text
        for k in ("模型", "推理", "推荐系统", "特征", "AB实验", "算法", "向量化")
    ):
        phrases.append("机器学习、模型服务、推理延迟、特征与效果")

    if not phrases:
        return ""

    seen: set[str] = set()
    ordered: list[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return " ".join(ordered)


def embed_text_with_title(title_line: str, hints: str, rest: str) -> str:
    """向量入库统一格式：首行 Title，其次可选检索增强词，再接正文片段。"""
    t = (title_line or "").strip()
    lines = [f"Title: {t}"]
    h = (hints or "").strip()
    if h:
        lines.append(f"相关检索词: {h}")
    r = (rest or "").strip()
    if r:
        lines.append(r)
    return "\n".join(lines)
