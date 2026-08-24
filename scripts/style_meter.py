#!/usr/bin/env python3
"""Measure sentence/paragraph rhythm for Arknights Sensory Writer drafts.

Usage:
  python style_meter.py draft.txt
  python style_meter.py --strict draft.txt
  cat draft.txt | python style_meter.py --strict -
  python style_meter.py --json draft.txt

Normal mode reports diagnostics. --strict exits non-zero when v7 global
paragraph/sentence hard anti-patterns are detected.
"""
from __future__ import annotations
import argparse, json, math, re, statistics, sys
from pathlib import Path

END = set("。！？!?")
TRANSITION_PREFIXES = (
    "不过", "但是", "当然", "那么", "可", "问题", "于是", "所以", "毕竟",
    "只是", "然而", "正因如此", "不得不说", "换句话说", "反过来",
)
DIALOGUE_PREFIXES = ("“", '"', "「", "『", "—")

# Configuration vocabulary is allowed in setup/chat, but should not leak into finished prose.
FORBIDDEN_META_TOKENS = (
    "writing_mode", "sensory_intensity", "erotic_fetish",
    "成人恋物", "文艺轻口", "默认鉴赏", "专业分析", "专业重口",
)
META_LABEL_PATTERNS = (
    re.compile(r"[“\"「『]极重[”\"」』]"),
    re.compile(r"(?:写作)?模式\s*[:：=]"),
    re.compile(r"(?:感官)?强度\s*[:：=]"),
    re.compile(r"(?:极重|重口|专业|文艺|均衡)(?:模式|档位|强度)"),
    re.compile(r"(?:推到|达到|进入|切到|调到|升到|写到)[^。！？!?\n]{0,10}(?:极重|重口|专业模式|文艺模式)"),
)


def effective_len(text: str) -> int:
    return sum(
        1 for ch in text
        if ("\u3400" <= ch <= "\u9fff") or ("\u3040" <= ch <= "\u30ff") or ch.isalnum()
    )


def split_sentences(paragraph: str) -> list[str]:
    out, buf = [], []
    for ch in paragraph:
        buf.append(ch)
        if ch in END:
            s = "".join(buf).strip()
            if effective_len(s):
                out.append(s)
            buf = []
    tail = "".join(buf).strip()
    if effective_len(tail):
        out.append(tail)
    return out


def percentile(xs: list[int], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def paragraphs_from_text(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [re.sub(r"\s*\n\s*", "", b).strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1 and "\n" in text:
        blocks = [x.strip() for x in text.splitlines() if x.strip()]
    return blocks


def is_dialogue_paragraph(p: str) -> bool:
    s = p.lstrip()
    return s.startswith(DIALOGUE_PREFIXES)


def analyze(text: str) -> dict:
    pars = paragraphs_from_text(text)
    groups = [split_sentences(p) for p in pars]
    lens = [effective_len(s) for g in groups for s in g]
    pchars = [effective_len(p) for p in pars]
    psc = [len(g) for g in groups]
    if not lens:
        raise ValueError("No Chinese-style sentences found")

    labels = []
    narrative_short_single = 0
    transition_single = 0
    for p, c, n in zip(pars, pchars, psc):
        dialogue = is_dialogue_paragraph(p)
        if dialogue:
            labels.append("D")
            continue
        if n == 1 and c <= 30:
            labels.append("S")
            narrative_short_single += 1
            stripped = p.lstrip(" \t　")
            if stripped.startswith(TRANSITION_PREFIXES):
                transition_single += 1
        elif c >= 60 and n >= 2:
            labels.append("L")
        else:
            labels.append("M")

    isolated_lsl = sum(
        1 for i in range(1, len(labels) - 1)
        if labels[i] == "S" and labels[i - 1] == "L" and labels[i + 1] == "L"
    )
    abab4 = sum(
        1 for i in range(len(labels) - 3)
        if labels[i:i + 4] in (["L", "S", "L", "S"], ["S", "L", "S", "L"])
    )
    alternating_short_between_long = sum(
        1 for i in range(1, len(labels) - 1)
        if labels[i] == "S" and labels[i - 1] in ("L", "M") and labels[i + 1] in ("L", "M")
    )

    mean = statistics.mean(lens)
    sd = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    meta_token_hits = sorted({token for token in FORBIDDEN_META_TOKENS if token in text})
    meta_pattern_hits = sorted({m.group(0) for pat in META_LABEL_PATTERNS for m in pat.finditer(text)})

    result = {
        "paragraphs": len(pars),
        "sentences": len(lens),
        "effective_chars": sum(lens),
        "sentence_mean": mean,
        "sentence_median": statistics.median(lens),
        "sentence_sd": sd,
        "sentence_cv": sd / mean if mean else 0.0,
        "sentence_q25": percentile(lens, .25),
        "sentence_q75": percentile(lens, .75),
        "short_le_15": sum(x <= 15 for x in lens) / len(lens),
        "short_le_20": sum(x <= 20 for x in lens) / len(lens),
        "long_ge_45": sum(x >= 45 for x in lens) / len(lens),
        "long_ge_60": sum(x >= 60 for x in lens) / len(lens),
        "paragraph_mean_chars": statistics.mean(pchars) if pchars else 0.0,
        "single_sentence_paragraph_ratio": sum(n == 1 for n in psc) / len(psc) if psc else 0.0,
        "short_single_paragraph_ratio": narrative_short_single / len(labels) if labels else 0.0,
        "isolated_long_short_long": isolated_lsl,
        "abab4_windows": abab4,
        "short_between_nonshort": alternating_short_between_long,
        "transition_short_single": transition_single,
        "paragraph_labels": "".join(labels),
        "meta_token_hits": meta_token_hits,
        "meta_label_hits": meta_pattern_hits,
    }

    warnings: list[str] = []
    hard: list[str] = []

    if mean < 30:
        warnings.append("平均句长偏低：草稿可能过碎。")
    elif mean > 60:
        warnings.append("平均句长偏高：检查是否复句过度堆叠。")
    if result["short_le_15"] > .20:
        warnings.append("<=15字短句超过20%：明显高于作者常态，检查碎句。")
    if result["short_le_20"] > .30:
        warnings.append("<=20字句超过30%：整体节奏可能过于短促。")
    if sd < 15:
        warnings.append("句长标准差低于15：句子可能过于整齐。")
    elif sd > 35:
        warnings.append("句长标准差高于35：检查极短句与超长句是否跳动过强。")

    if result["short_single_paragraph_ratio"] > .20:
        warnings.append("非对话短单句自然段超过20%：检查是否把句法节奏误当成换段节奏。")
    if result["short_single_paragraph_ratio"] > .25:
        hard.append("非对话短单句自然段超过25%。")

    if abab4 >= 1:
        warnings.append("检测到长/短/长/短ABAB窗口：逐个检查是否为人为节拍。")
    if len(pars) <= 40:
        if isolated_lsl >= 2:
            hard.append("短中篇出现至少2个长段-短单句-长段(L-S-L)孤立短段。")
        if abab4 >= 2:
            hard.append("短中篇出现至少2个长/短/长/短ABAB窗口。")
    else:
        if (isolated_lsl / len(pars) * 100) > 5:
            hard.append("L-S-L孤立短段超过5次/100段。")
        if (abab4 / len(pars) * 100) > 4:
            hard.append("ABAB长短段窗口超过4次/100段。")

    if transition_single >= 3:
        warnings.append("出现至少3个短转折单句段：检查是否把连接词当成换段标记。")
    if alternating_short_between_long >= 4 and len(pars) <= 40:
        warnings.append("多个短单句段规律插在非短段之间：即使未严格形成ABAB，也可能过于刻意。")

    if result["short_le_15"] > .25 and mean < 32:
        hard.append("短句密度与平均句长同时明显偏离历史文风。")

    if meta_token_hits:
        hard.append("正文泄漏内部配置词/预设名：" + "、".join(meta_token_hits))
    if meta_pattern_hits:
        hard.append("正文出现把模式/强度写成文章概念的元叙述：" + "；".join(meta_pattern_hits[:6]))

    result["warnings"] = warnings
    result["hard_failures"] = hard
    result["strict_pass"] = not hard
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="UTF-8 text/markdown file, or - for stdin")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on v7 hard anti-patterns")
    args = ap.parse_args()

    if args.path == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.path).read_text(encoding="utf-8")
    result = analyze(text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        pct = lambda x: f"{x*100:.1f}%"
        print(f"paragraphs={result['paragraphs']} sentences={result['sentences']} effective_chars={result['effective_chars']}")
        print(
            "sentence: "
            f"mean={result['sentence_mean']:.1f} median={result['sentence_median']:.1f} "
            f"sd={result['sentence_sd']:.1f} cv={result['sentence_cv']:.2f} "
            f"q25={result['sentence_q25']:.1f} q75={result['sentence_q75']:.1f}"
        )
        print(
            "density: "
            f"<=15={pct(result['short_le_15'])} <=20={pct(result['short_le_20'])} "
            f">=45={pct(result['long_ge_45'])} >=60={pct(result['long_ge_60'])}"
        )
        print(
            "paragraph: "
            f"mean_chars={result['paragraph_mean_chars']:.1f} "
            f"single_sentence={pct(result['single_sentence_paragraph_ratio'])} "
            f"short_single={pct(result['short_single_paragraph_ratio'])} "
            f"isolated_LSL={result['isolated_long_short_long']} "
            f"abab4={result['abab4_windows']} "
            f"short_between_nonshort={result['short_between_nonshort']} "
            f"transition_short_single={result['transition_short_single']}"
        )
        print(f"paragraph_labels={result['paragraph_labels']}")
        if result["meta_token_hits"] or result["meta_label_hits"]:
            print("meta_leak:")
            if result["meta_token_hits"]:
                print("- tokens=" + "、".join(result["meta_token_hits"]))
            if result["meta_label_hits"]:
                print("- labels=" + "；".join(result["meta_label_hits"]))
        if result["warnings"]:
            print("warnings:")
            for w in result["warnings"]:
                print(f"- {w}")
        else:
            print("warnings: none")
        if result["hard_failures"]:
            print("hard_failures:")
            for h in result["hard_failures"]:
                print(f"- {h}")
        else:
            print("hard_failures: none")

    if args.strict and result["hard_failures"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
