"""Evaluate a knowledge-base QA system with small, explicit metrics.

The tool is intentionally narrow: it reads JSONL cases, compares each answer
with expected sources and keywords, and writes a Markdown report. It does not
call an LLM, so the evaluation itself is deterministic and easy to audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REFUSAL_MARKERS = (
    "无法回答",
    "无法确定",
    "无法确认",
    "无法提供",
    "不能回答",
    "不知道",
    "不清楚",
    "未提供",
    "没有找到",
    "没有相关",
    "知识库中没有",
    "知识库未包含",
    "未提及",
)


class DatasetError(ValueError):
    """The JSONL dataset is missing, malformed, or semantically invalid."""


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    answer: str
    cited_sources: Tuple[str, ...]
    expected_sources: Tuple[str, ...]
    expected_keywords: Tuple[str, ...]
    answerable: bool = True
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    case: Case
    source_recall: float
    citation_precision: float
    keyword_coverage: float
    refusal_ok: bool
    passed: bool
    missing_sources: Tuple[str, ...]
    missing_keywords: Tuple[str, ...]
    unexpected_sources: Tuple[str, ...]


@dataclass(frozen=True)
class Summary:
    total: int
    passed: int
    pass_rate: float
    answerable: int
    unanswerable: int
    avg_source_recall: float
    avg_citation_precision: float
    avg_keyword_coverage: float
    refusal_accuracy: float


def _norm(value: str) -> str:
    return value.strip().casefold()


def _norm_set(values: Sequence[str]) -> set:
    return {_norm(value) for value in values}


def _require_str(
    obj: Dict[str, Any],
    key: str,
    where: str,
) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{where}: field '{key}' must be a non-empty string")
    return value.strip()


def _optional_str_list(
    obj: Dict[str, Any],
    key: str,
    where: str,
) -> Tuple[str, ...]:
    value = obj.get(key, [])
    if not isinstance(value, list):
        raise DatasetError(f"{where}: field '{key}' must be a list of strings")
    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetError(
                f"{where}: field '{key}' must contain non-empty strings"
            )
        result.append(item.strip())
    return tuple(result)


def _parse_case(
    obj: Dict[str, Any],
    where: str,
) -> Case:
    answerable = obj.get("answerable", True)
    if not isinstance(answerable, bool):
        raise DatasetError(f"{where}: field 'answerable' must be true or false")

    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        raise DatasetError(f"{where}: field 'notes' must be a string")

    expected_sources = _optional_str_list(obj, "expected_sources", where)
    if answerable and not expected_sources:
        raise DatasetError(
            f"{where}: answerable cases need at least one expected source"
        )

    return Case(
        id=_require_str(obj, "id", where),
        question=_require_str(obj, "question", where),
        answer=_require_str(obj, "answer", where),
        cited_sources=_optional_str_list(obj, "cited_sources", where),
        expected_sources=expected_sources,
        expected_keywords=_optional_str_list(obj, "expected_keywords", where),
        answerable=answerable,
        notes=notes.strip(),
    )


def load_cases(path: str) -> List[Case]:
    """Load and validate a JSONL file, with line numbers in error messages."""
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"dataset not found: {source}")

    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"could not read {source}: {exc}") from exc

    cases: List[Case] = []
    seen_ids = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        where = f"{source}:{line_number}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"{where}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(obj, dict):
            raise DatasetError(f"{where}: each line must be a JSON object")
        case = _parse_case(obj, where)
        if case.id in seen_ids:
            raise DatasetError(f"{where}: duplicate case id '{case.id}'")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetError(f"{source}: dataset is empty")
    return cases


def is_refusal(answer: str) -> bool:
    """Detect a clear refusal/insufficient-knowledge answer."""
    text = answer.strip().casefold()
    if not text:
        return True
    return any(marker in text for marker in REFUSAL_MARKERS)


def evaluate_case(case: Case, keyword_threshold: float = 0.6) -> CaseResult:
    expected_sources = _norm_set(case.expected_sources)
    cited_sources = _norm_set(case.cited_sources)
    matched_sources = expected_sources & cited_sources

    source_recall = (
        len(matched_sources) / len(expected_sources)
        if expected_sources
        else 1.0
    )
    citation_precision = (
        len(matched_sources) / len(cited_sources)
        if cited_sources
        else (1.0 if not expected_sources else 0.0)
    )

    missing_sources = tuple(
        source
        for source in case.expected_sources
        if _norm(source) not in cited_sources
    )
    unexpected_sources = tuple(
        source
        for source in case.cited_sources
        if _norm(source) not in expected_sources
    )
    missing_keywords = tuple(
        keyword
        for keyword in case.expected_keywords
        if _norm(keyword) not in case.answer.casefold()
    )
    keyword_coverage = (
        len(case.expected_keywords) - len(missing_keywords)
    ) / len(case.expected_keywords) if case.expected_keywords else 1.0

    detected_refusal = is_refusal(case.answer)
    refusal_ok = detected_refusal == (not case.answerable)

    if case.answerable:
        passed = (
            refusal_ok
            and source_recall >= 1.0
            and keyword_coverage >= keyword_threshold
        )
    else:
        passed = refusal_ok

    return CaseResult(
        case=case,
        source_recall=source_recall,
        citation_precision=citation_precision,
        keyword_coverage=keyword_coverage,
        refusal_ok=refusal_ok,
        passed=passed,
        missing_sources=missing_sources,
        missing_keywords=missing_keywords,
        unexpected_sources=unexpected_sources,
    )


def evaluate_cases(
    cases: Sequence[Case],
    keyword_threshold: float = 0.6,
) -> List[CaseResult]:
    return [evaluate_case(case, keyword_threshold) for case in cases]


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: Sequence[CaseResult]) -> Summary:
    answerable = [result for result in results if result.case.answerable]
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    return Summary(
        total=total,
        passed=passed,
        pass_rate=passed / total if total else 0.0,
        answerable=len(answerable),
        unanswerable=total - len(answerable),
        avg_source_recall=_average(
            [result.source_recall for result in answerable]
        ),
        avg_citation_precision=_average(
            [result.citation_precision for result in answerable]
        ),
        avg_keyword_coverage=_average(
            [result.keyword_coverage for result in answerable]
        ),
        refusal_accuracy=_average(
            [1.0 if result.refusal_ok else 0.0 for result in results]
        ),
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _cell(value: str, limit: int = 60) -> str:
    text = " ".join(value.split()).replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def render_markdown(
    results: Sequence[CaseResult],
    summary: Summary,
    keyword_threshold: float = 0.6,
) -> str:
    lines: List[str] = [
        "# KB QA Evaluation Report",
        "",
        (
            f"- Cases: {summary.total} "
            f"({summary.answerable} answerable, {summary.unanswerable} unanswerable)"
        ),
        f"- Passed: {summary.passed}/{summary.total} ({_percent(summary.pass_rate)})",
        f"- Keyword threshold: {_percent(keyword_threshold)}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Pass rate | {_percent(summary.pass_rate)} |",
        f"| Source recall (answerable) | {_percent(summary.avg_source_recall)} |",
        (
            "| Citation precision (answerable) | "
            f"{_percent(summary.avg_citation_precision)} |"
        ),
        f"| Keyword coverage (answerable) | {_percent(summary.avg_keyword_coverage)} |",
        f"| Refusal accuracy (all) | {_percent(summary.refusal_accuracy)} |",
        "",
        "## Cases",
        "",
        "`Refusal OK` means the answer correctly refused an unanswerable question "
        "or correctly did not refuse an answerable one.",
        "",
        "| ID | Question | Pass | Recall | Cite P | Keywords | Refusal OK |",
        "| --- | --- | :---: | ---: | ---: | ---: | :---: |",
    ]

    for result in results:
        if result.case.answerable:
            recall = _percent(result.source_recall)
            precision = _percent(result.citation_precision)
            keywords = _percent(result.keyword_coverage)
        else:
            recall = precision = keywords = "n/a"
        lines.append(
            "| {id} | {question} | {passed} | {recall} | {precision} | "
            "{keywords} | {refusal} |".format(
                id=_cell(result.case.id, 24),
                question=_cell(result.case.question),
                passed="PASS" if result.passed else "FAIL",
                recall=recall,
                precision=precision,
                keywords=keywords,
                refusal="ok" if result.refusal_ok else "bad",
            )
        )

    lines.extend(["", "## Failures", ""])
    failed = [result for result in results if not result.passed]
    if not failed:
        lines.append("None.")
    else:
        for result in failed:
            case = result.case
            lines.extend(
                [
                    f"### {case.id}: {_cell(case.question, 120)}",
                    "",
                    f"- Answer: {_cell(case.answer, 400)}",
                    f"- Expected sources: {', '.join(case.expected_sources) or '(none)'}",
                    f"- Cited sources: {', '.join(case.cited_sources) or '(none)'}",
                    (
                        "- Missing sources: "
                        f"{', '.join(result.missing_sources) or '(none)'}"
                    ),
                    (
                        "- Unexpected sources: "
                        f"{', '.join(result.unexpected_sources) or '(none)'}"
                    ),
                    (
                        "- Missing keywords: "
                        f"{', '.join(result.missing_keywords) or '(none)'}"
                    ),
                    (
                        "- Refusal expected/detected: "
                        f"{'no' if case.answerable else 'yes'}/"
                        f"{'yes' if is_refusal(case.answer) else 'no'}"
                    ),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate KB QA answers from a JSONL file.",
    )
    parser.add_argument(
        "--cases",
        default="cases.jsonl",
        help="input JSONL file (default: cases.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="report.md",
        help="output Markdown report (default: report.md)",
    )
    parser.add_argument(
        "--keyword-threshold",
        type=float,
        default=0.6,
        help="minimum keyword coverage for an answerable case to pass (0..1)",
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.keyword_threshold <= 1.0:
        parser.error("--keyword-threshold must be between 0 and 1")

    try:
        cases = load_cases(args.cases)
        results = evaluate_cases(cases, args.keyword_threshold)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = summarize(results)
    report = render_markdown(results, summary, args.keyword_threshold)

    try:
        Path(args.out).write_text(report, encoding="utf-8")
    except OSError as exc:
        print(f"error: could not write {args.out}: {exc}", file=sys.stderr)
        return 2

    print(
        f"wrote {args.out}: {summary.passed}/{summary.total} passed "
        f"({_percent(summary.pass_rate)})"
    )
    return 0 if summary.passed == summary.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
