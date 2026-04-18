"""Evaluation dataset schema + (de)serialisation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    """One evaluation example.

    * ``question``       — user query.
    * ``ground_truth``   — gold answer used by ``answer_relevance`` and
                            ``context_recall`` (may be a single sentence).
    * ``relevant_chunk_ids`` — ids of chunks that SHOULD be retrieved;
                            optional but enables retrieval-precision /
                            recall metrics.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    ground_truth: str = ""
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    """A collection of :class:`EvalCase`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cases: list[EvalCase] = Field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> EvalDataset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


__all__ = ["EvalCase", "EvalDataset"]
