"""Tests for the deterministic KB QA evaluation metrics."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kb_eval import (
    Case,
    DatasetError,
    evaluate_case,
    evaluate_cases,
    is_refusal,
    load_cases,
    main,
    render_markdown,
    summarize,
)


def make_case(**overrides):
    values = {
        "id": "case-1",
        "question": "年假有几天？",
        "answer": "入职满 1 年有 5 天年假。",
        "cited_sources": ("HR-LEAVE-02",),
        "expected_sources": ("HR-LEAVE-02",),
        "expected_keywords": ("5 天",),
        "answerable": True,
    }
    values.update(overrides)
    return Case(**values)


class MetricTests(unittest.TestCase):
    def test_perfect_answer_passes(self) -> None:
        result = evaluate_case(make_case())
        self.assertTrue(result.passed)
        self.assertEqual(result.source_recall, 1.0)
        self.assertEqual(result.citation_precision, 1.0)
        self.assertEqual(result.keyword_coverage, 1.0)
        self.assertTrue(result.refusal_ok)

    def test_missing_keyword_fails(self) -> None:
        result = evaluate_case(
            make_case(
                answer="入职满 1 年有 5 天年假。",
                expected_keywords=("5 天", "10 天"),
            )
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.keyword_coverage, 0.5)
        self.assertEqual(result.missing_keywords, ("10 天",))

    def test_missing_source_fails(self) -> None:
        result = evaluate_case(
            make_case(
                cited_sources=("HR-OTHER",),
                expected_sources=("HR-LEAVE-02",),
            )
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.source_recall, 0.0)
        self.assertEqual(result.citation_precision, 0.0)
        self.assertEqual(result.missing_sources, ("HR-LEAVE-02",))
        self.assertEqual(result.unexpected_sources, ("HR-OTHER",))

    def test_unanswerable_case_passes_when_refusing(self) -> None:
        result = evaluate_case(
            make_case(
                answer="知识库中没有这项信息，无法回答。",
                cited_sources=(),
                expected_sources=(),
                expected_keywords=(),
                answerable=False,
            )
        )
        self.assertTrue(result.refusal_ok)
        self.assertTrue(result.passed)

    def test_unanswerable_case_fails_when_answering(self) -> None:
        result = evaluate_case(
            make_case(
                answer="员工持股计划每年开放一次。",
                cited_sources=(),
                expected_sources=(),
                expected_keywords=(),
                answerable=False,
            )
        )
        self.assertFalse(result.refusal_ok)
        self.assertFalse(result.passed)

    def test_answerable_case_fails_when_refusing(self) -> None:
        result = evaluate_case(
            make_case(answer="无法确认，知识库中没有相关信息。")
        )
        self.assertFalse(result.refusal_ok)
        self.assertFalse(result.passed)

    def test_is_refusal_is_not_fooled_by_normal_negation(self) -> None:
        self.assertTrue(is_refusal("知识库中没有相关信息，无法回答。"))
        self.assertTrue(is_refusal("我没有找到食堂菜单。"))
        self.assertFalse(is_refusal("病假没有硬性规定必须提前几天。"))
        self.assertFalse(is_refusal("发票抬头必须填写公司全称。"))

    def test_summary_uses_answerable_cases_for_retrieval(self) -> None:
        perfect = evaluate_case(make_case(id="a"))
        wrong_source = evaluate_case(
            make_case(
                id="b",
                cited_sources=("OTHER",),
                expected_sources=("HR-LEAVE-02",),
            )
        )
        unanswerable = evaluate_case(
            make_case(
                id="c",
                answer="知识库中没有相关信息，无法回答。",
                cited_sources=(),
                expected_sources=(),
                expected_keywords=(),
                answerable=False,
            )
        )
        summary = summarize([perfect, wrong_source, unanswerable])
        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.answerable, 2)
        self.assertEqual(summary.unanswerable, 1)
        self.assertEqual(summary.passed, 2)
        self.assertAlmostEqual(summary.avg_source_recall, 0.5)
        self.assertAlmostEqual(summary.avg_keyword_coverage, 1.0)
        self.assertAlmostEqual(summary.refusal_accuracy, 1.0)


class DatasetTests(unittest.TestCase):
    def test_load_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "one",
                                "question": "q1",
                                "answer": "a1",
                                "cited_sources": ["S1"],
                                "expected_sources": ["S1"],
                                "expected_keywords": ["a1"],
                            },
                            ensure_ascii=False,
                        ),
                        "",
                        json.dumps(
                            {
                                "id": "two",
                                "question": "q2",
                                "answer": "知识库中没有相关信息。",
                                "answerable": False,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            cases = load_cases(str(path))
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].id, "one")
        self.assertFalse(cases[1].answerable)

    def test_invalid_json_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "ok",
                        "question": "q",
                        "answer": "a",
                        "cited_sources": ["S1"],
                        "expected_sources": ["S1"],
                        "expected_keywords": [],
                    }
                )
                + "\nnot-json\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError) as ctx:
                load_cases(str(path))
        self.assertIn(":2:", str(ctx.exception))
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_duplicate_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            record = {
                "id": "same",
                "question": "q",
                "answer": "a",
                "cited_sources": ["S1"],
                "expected_sources": ["S1"],
                "expected_keywords": [],
            }
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError) as ctx:
                load_cases(str(path))
        self.assertIn("duplicate case id", str(ctx.exception))

    def test_answerable_case_requires_expected_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "question": "q",
                        "answer": "a",
                        "expected_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError) as ctx:
                load_cases(str(path))
        self.assertIn("expected source", str(ctx.exception))


class ReportTests(unittest.TestCase):
    def test_report_contains_summary_and_failure_detail(self) -> None:
        result = evaluate_case(
            make_case(
                id="bad",
                answer="只有 5 天。",
                expected_keywords=("5 天", "10 天"),
            )
        )
        summary = summarize([result])
        report = render_markdown([result], summary)
        self.assertIn("# KB QA Evaluation Report", report)
        self.assertIn("| Pass rate | 0.0% |", report)
        self.assertIn("### bad:", report)
        self.assertIn("Missing keywords: 10 天", report)

    def test_main_writes_report_and_returns_failure_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.jsonl"
            out_path = Path(tmp) / "report.md"
            cases_path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "question": "年假几天？",
                        "answer": "只有 5 天。",
                        "cited_sources": ["HR-LEAVE-02"],
                        "expected_sources": ["HR-LEAVE-02"],
                        "expected_keywords": ["5 天", "10 天"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = main(
                    [
                        "--cases",
                        str(cases_path),
                        "--out",
                        str(out_path),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertTrue(out_path.exists())
            self.assertIn("0/1 passed", stdout.getvalue())
            self.assertIn("Missing keywords: 10 天", out_path.read_text("utf-8"))

    def test_main_returns_two_for_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(
                    [
                        "--cases",
                        str(Path(tmp) / "missing.jsonl"),
                        "--out",
                        str(Path(tmp) / "report.md"),
                    ]
                )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
