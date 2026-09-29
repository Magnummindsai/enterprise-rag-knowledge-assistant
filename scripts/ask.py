"""Ask a question from the command line.

    python scripts/ask.py "What is the hotel rate cap in London?"
    python scripts/ask.py --trace "How much PTO can I roll over?"
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--trace", action="store_true", help="show the LangGraph path taken")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    pipeline = RAGPipeline(get_settings())
    result = pipeline.ask(" ".join(args.question))

    print(f"\n[{pipeline.mode} | confidence={result.confidence} | fallback={result.used_fallback} "
          f"| {result.latency_ms:.0f} ms]\n")
    print(result.answer, "\n")
    for c in result.citations or result.related:
        print(f"  [{c['id']}] {c['doc_title']} - {c['section']} {c['heading']}")
    if args.trace:
        print("\nTrace:")
        for step in result.trace:
            print(f"  {step['node']:<20} {step['ms']:>7.1f} ms  {step['detail']}")
