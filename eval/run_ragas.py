"""Evaluate the assistant on a golden Q&A set.

Always computes offline retrieval/grounding metrics (no API key needed):
  hit@k, MRR, citation hit rate, grounded-answer rate, abstention accuracy, fallback rate, latency.

In live mode (OPENAI_API_KEY set) it also runs RAGAS with an LLM judge:
  faithfulness, answer relevancy, context precision, context recall.

Usage:
    python -m eval.run_ragas                  # auto: RAGAS if a key is configured
    python -m eval.run_ragas --no-ragas       # offline metrics only
    python -m eval.run_ragas --judge-model gpt-4o --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.pipeline import AskResult, RAGPipeline  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
NOT_IN_CORPUS = "NOT_IN_CORPUS"


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def abstained(result: AskResult) -> bool:
    return not result.citations and result.confidence == "low"


def offline_metrics(rows: list[dict], results: list[AskResult]) -> tuple[dict, list[dict]]:
    per_question, rr, hits, cite_hits, grounded, false_abstain = [], [], [], [], [], []
    oos_correct = oos_total = 0

    for row, res in zip(rows, results, strict=True):
        expected = set(row["expected_ids"])
        cited = [c["id"] for c in res.citations]
        record = {
            "id": row["id"], "question": row["question"], "answer": res.answer, "cited": cited,
            "expected": sorted(expected), "confidence": res.confidence,
            "fallback": res.used_fallback, "latency_ms": res.latency_ms,
        }
        if row["ground_truth"] == NOT_IN_CORPUS:
            oos_total += 1
            oos_correct += abstained(res)
            record["correct_abstention"] = abstained(res)
        else:
            rank = next((i for i, cid in enumerate(res.context_ids, 1) if cid in expected), None)
            rr.append(1 / rank if rank else 0.0)
            hits.append(rank is not None)
            cite_hits.append(bool(expected & set(cited)))
            grounded.append(res.grounded and bool(cited))
            false_abstain.append(abstained(res))
            record.update({"hit": rank is not None, "rank": rank, "cited_expected": bool(expected & set(cited))})
        per_question.append(record)

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    latencies = [r.latency_ms for r in results]
    summary = {
        "questions": len(rows),
        "answerable": len(hits),
        "out_of_scope": oos_total,
        "hit_rate@k": mean(hits),
        "mrr": mean(rr),
        "citation_hit_rate": mean(cite_hits),
        "grounded_answer_rate": mean(grounded),
        "false_abstention_rate": mean(false_abstain),
        "abstention_accuracy": round(oos_correct / oos_total, 3) if oos_total else None,
        "fallback_rate": mean([r.used_fallback for r in results]),
        "latency_ms_p50": round(statistics.median(latencies), 1),
        "latency_ms_max": round(max(latencies), 1),
    }
    return summary, per_question


def ragas_metrics(rows: list[dict], results: list[AskResult], judge_model: str, embedding_model: str) -> dict:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    samples = [
        {
            "user_input": row["question"],
            "response": res.answer,
            "retrieved_contexts": res.contexts or [""],
            "reference": row["ground_truth"],
        }
        for row, res in zip(rows, results, strict=True)
        if row["ground_truth"] != NOT_IN_CORPUS
    ]
    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if df[c].dtype.kind == "f"]
    return {col: round(float(df[col].mean()), 3) for col in metric_cols}


def write_report(out_dir: Path, meta: dict, summary: dict, ragas: dict | None, per_question: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "offline": summary, "ragas": ragas, "questions": per_question}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Evaluation Report",
        "",
        f"- **Run:** {meta['timestamp']}",
        f"- **Mode:** {meta['mode']} · LLM `{meta['llm']}` · reranker `{meta['reranker']}` · embedder `{meta['embedder']}`",
        "",
        "## Retrieval & grounding (offline)",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
    ]
    if ragas:
        lines += ["## RAGAS (LLM judge)", "", "| Metric | Score |", "|---|---|",
                  *[f"| {k} | {v} |" for k, v in ragas.items()], ""]
    else:
        lines += ["_RAGAS metrics skipped (mock mode or `--no-ragas`)._", ""]
    lines += ["## Per question", "", "| id | hit | cited expected | confidence | fallback | cited |", "|---|---|---|---|---|---|"]
    for q in per_question:
        hit = q.get("hit", "abstain ✓" if q.get("correct_abstention") else "abstain ✗")
        lines.append(f"| {q['id']} | {hit} | {q.get('cited_expected', '-')} | {q['confidence']} | "
                     f"{q['fallback']} | {', '.join(q['cited']) or '-'} |")
    report = out_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "golden_qa.jsonl")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    parser.add_argument("--no-ragas", action="store_true", help="skip LLM-judged RAGAS metrics")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="LLM used by RAGAS as the judge")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    rows = load_dataset(args.dataset, args.limit)

    print(f"Evaluating {len(rows)} questions in {pipeline.mode.upper()} mode...")
    results = [pipeline.ask(row["question"]) for row in rows]
    summary, per_question = offline_metrics(rows, results)

    ragas = None
    if args.no_ragas or settings.is_mock:
        if settings.is_mock and not args.no_ragas:
            print("Mock mode: skipping RAGAS (needs OPENAI_API_KEY). Offline metrics only.")
    else:
        ragas = ragas_metrics(rows, results, args.judge_model, settings.openai_embedding_model)

    meta = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), **pipeline.describe(),
            "judge_model": args.judge_model if ragas else None}
    report = write_report(args.out, meta, summary, ragas, per_question)

    print("\nOffline metrics:")
    for k, v in summary.items():
        print(f"  {k:<24} {v}")
    if ragas:
        print("\nRAGAS:")
        for k, v in ragas.items():
            print(f"  {k:<24} {v}")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
