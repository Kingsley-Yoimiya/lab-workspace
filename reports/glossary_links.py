#!/usr/bin/env python3
"""在报告正文关键缩写后附注含义（括号内），不用跳转链接。

每一逻辑行、每个词条只附注一次；若缩写后已有中文括号附注则跳过。
"""
from __future__ import annotations

import re
from pathlib import Path

# 附注正文尽量避免再写可匹配缩写（MTE/Cube/…），防止套娃
NOTES: dict[str, str] = {
    "AI Core": "昇腾主计算核，内含矩阵/向量/标量计算与片上搬运等部件",
    "AICore": "即 AI Core（昇腾主计算核）；npu-smi 的 Aicore Usage Rate 对应该核占用",
    "AICPU": "器件侧 AI CPU，与主计算核上的矩阵/向量单元不是同一执行体；npu-smi 报 Aicpu Usage Rate",
    "AI CPU": "器件侧 AI CPU，与主计算核上的矩阵/向量单元不是同一执行体",
    "CtrlCPU": "器件侧控制 CPU；npu-smi 报 Ctrlcpu Usage Rate，不是宿主机 top 里的 CPU%",
    "Ctrlcpu": "器件侧控制 CPU；npu-smi 报 Ctrlcpu Usage Rate，不是宿主机 top 里的 CPU%",
    "Control CPU": "器件侧控制 CPU",
    "health_power_w": "流程早期轻载时刻的 npu-smi Real-time Power；health 是采样阶段标签，不是健康分",
    "health_power": "流程早期轻载时刻的 npu-smi Real-time Power；health 是采样阶段标签，不是健康分",
    "health_temp_c": "流程早期轻载/开测温度快照，与负载中 board_temp 不同时刻",
    "health_temp": "流程早期轻载/开测温度快照，与负载中 board_temp 不同时刻",
    "npu-smi": "昇腾 NPU 系统管理命令行，可查功耗/温度/usages 等",
    "MTE": "Memory Transfer Engine，片上 Buffer 与 Global Memory 之间的数据搬运引擎；本字段多用 Tensor.copy_ 作纯搬运带宽代理，并非直接读该引擎指令计数器",
    "SFU": "特殊函数类吞吐代理；本探针默认 torch.exp，按 1 op/元素计，公开叙述常归在向量计算能力面",
    "HBM": "High Bandwidth Memory，器件高带宽外存",
    "Cube": "矩阵计算单元：主计算核内专做大规模矩阵乘加、提供主算力的部件",
    "Vector": "向量计算单元：做逐元素/向量运算与部分数学函数，灵活度高于矩阵单元、峰值算力通常更低",
    "Scalar": "标量与控制单元：负责循环/分支，并为矩阵/向量/搬运指令计算地址与参数",
    "DaVinci Core": "昇腾主计算核的常用别称",
    "DaVinci": "昇腾 AI 处理器架构族常用名；计算核心即 AI Core",
}

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AI\s*Cores?"), "AI Core"),
    (re.compile(r"AICores?"), "AICore"),
    (re.compile(r"AICPUs?"), "AICPU"),
    (re.compile(r"AI\s*CPUs?"), "AI CPU"),
    (re.compile(r"CtrlCPUs?"), "CtrlCPU"),
    (re.compile(r"Ctrlcpus?"), "Ctrlcpu"),
    (re.compile(r"Control\s*CPUs?"), "Control CPU"),
    (re.compile(r"health_power_w"), "health_power_w"),
    (re.compile(r"health_power"), "health_power"),
    (re.compile(r"health_temp_c"), "health_temp_c"),
    (re.compile(r"health_temp"), "health_temp"),
    (re.compile(r"npu-smi"), "npu-smi"),
    (re.compile(r"MTEs?"), "MTE"),
    (re.compile(r"SFUs?"), "SFU"),
    (re.compile(r"HBMs?"), "HBM"),
    (re.compile(r"Cubes?"), "Cube"),
    (re.compile(r"Vectors?"), "Vector"),
    (re.compile(r"Scalars?"), "Scalar"),
    (re.compile(r"DaVinci\s*Cores?"), "DaVinci Core"),
    (re.compile(r"DaVinci"), "DaVinci"),
]

_GLOSSARY_LINK = re.compile(
    r"\[([^\]]+)\]\(ASCEND_HARDWARE_GLOSSARY_20260711\.md#[^)]+\)"
)

# 识别我们插入过的附注（跟在缩写后）
_NOTE_MARKERS = (
    "矩阵计算单元",
    "向量计算单元",
    "标量与控制",
    "Memory Transfer Engine",
    "High Bandwidth Memory",
    "器件侧 AI CPU",
    "器件侧控制 CPU",
    "昇腾主计算核",
    "Real-time Power",
    "系统管理命令行",
    "特殊函数类吞吐",
    "采样阶段标签",
    "轻载/开测温度",
    "AI 处理器架构族",
)


def strip_glossary_links(text: str) -> str:
    return _GLOSSARY_LINK.sub(r"\1", text)


def strip_existing_notes(text: str) -> str:
    """去掉「缩写（我们的附注）」中的括号段，便于干净重跑。"""

    def _strip_one(m: re.Match[str]) -> str:
        inner = m.group(2)
        if any(k in inner for k in _NOTE_MARKERS):
            return m.group(1)
        return m.group(0)

    prev = None
    while prev != text:
        prev = text
        for cre, _key in _PATTERNS:
            # 显式两捕获组：整词 + 括号内
            text = re.sub(
                rf"({cre.pattern})（([^）]{{8,400}})）",
                _strip_one,
                text,
            )
    return text


def _stash_protect(text: str) -> tuple[str, list[str]]:
    slots: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        slots.append(m.group(0))
        return f"@@SLOT{len(slots) - 1}@@"

    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", _stash, text)
    text = re.sub(r"\[[^\]]*\]\([^)]+\)", _stash, text)
    text = re.sub(r"`[^`\n]+`", _stash, text)
    text = re.sub(r"<[^>]+>", _stash, text)
    return text, slots


def _unstash(text: str, slots: list[str]) -> str:
    for i in range(len(slots) - 1, -1, -1):
        text = text.replace(f"@@SLOT{i}@@", slots[i])
    return text


def annotate_line(line: str) -> str:
    text, slots = _stash_protect(line)
    seen: set[str] = set()
    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        best: tuple[int, int, str, str] | None = None
        for cre, key in _PATTERNS:
            m = cre.search(text, pos)
            if not m:
                continue
            start, end = m.start(), m.end()
            if best is None or start < best[0] or (start == best[0] and end > best[1]):
                best = (start, end, m.group(0), key)
        if best is None:
            out.append(text[pos:])
            break
        start, end, raw, key = best
        out.append(text[pos:start])
        note = NOTES.get(key, "")
        rest = text[end : end + 10]
        already = rest.startswith("（") or rest.startswith("@@SLOT")
        if key in seen or not note or already:
            out.append(raw)
        else:
            out.append(f"{raw}（{note}）")
            seen.add(key)
        pos = end
    return _unstash("".join(out), slots)


def annotate_abbr(text: str) -> str:
    text = strip_glossary_links(text)
    text = strip_existing_notes(text)
    text = re.sub(r"图注中的缩写均可点击跳转。?", "", text)
    text = re.sub(r"图注缩写已链到昇腾词条锚点。?", "", text)
    lines = text.splitlines(keepends=True)
    return "".join(annotate_line(L) if L.strip() else L for L in lines)


def linkify_abbr(text: str, *, glossary: str | None = None) -> str:  # noqa: ARG001
    return annotate_abbr(text)


def annotate_file(path: Path) -> bool:
    # Ascend 专用 annotate：勿跑到沐曦交付稿上
    name = path.name
    if name.startswith("ASCEND_HARDWARE_GLOSSARY"):
        return False
    if "muxi" in name.lower():
        return False
    old = path.read_text(encoding="utf-8")
    new = annotate_abbr(old)
    if new == old:
        return False
    path.write_text(new, encoding="utf-8")
    return True


if __name__ == "__main__":
    rounds = Path(__file__).resolve().parent / "rounds"
    targets = [
        rounds / "card_constitution_20260711.md",
        rounds / "constitution_extra_fillgap_20260711.md",
        rounds / "METRIC_SEMANTICS_20260711.md",
        rounds / "CAMPAIGN_FINAL_20260711.md",
        rounds / "FIGURE_PROVENANCE_AUDIT_20260711.md",
    ]
    n = 0
    for p in targets:
        if p.exists() and annotate_file(p):
            n += 1
            print(" ", p.name)
    print(f"updated {n} files")
