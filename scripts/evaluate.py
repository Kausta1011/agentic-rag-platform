"""CLI: run the offline evaluation harness.

Usage::

    python -m scripts.evaluate --dataset tests/eval/sample_dataset.json \
                               --out reports/eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentic_rag.api.dependencies import get_service
from agentic_rag.core.logging import configure_logging, get_logger
from agentic_rag.evaluation.dataset import EvalDataset
from agentic_rag.evaluation.runner import EvalRunner

log = get_logger(__name__)


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    dataset = EvalDataset.from_json(args.dataset)
    service = get_service()
    runner = EvalRunner(service.graph, concurrency=args.concurrency)
    report = await runner.run(dataset)

    summary = report.summary()
    print("\n=== Eval Summary ===")
    for k, v in summary.items():
        v_fmt = f"{v:.3f}" if isinstance(v, float) else v
        print(f"  {k:<20} {v_fmt}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "rows": report.to_rows()}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic-RAG evaluation harness")
    parser.add_argument("--dataset", required=True, help="Path to EvalDataset JSON")
    parser.add_argument("--out", help="Optional JSON output path")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
