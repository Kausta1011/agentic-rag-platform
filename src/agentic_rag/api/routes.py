"""HTTP + SSE routes."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from agentic_rag.api.dependencies import Service, get_service
from agentic_rag.core.exceptions import AgenticRAGError, GuardrailViolationError
from agentic_rag.core.logging import get_logger
from agentic_rag.models.queries import AnswerResponse, QueryRequest
from agentic_rag.models.state import build_initial_state
from agentic_rag.observability.metrics import get_metrics

log = get_logger(__name__)
router = APIRouter()


@router.get("/healthz", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics", tags=["system"])
async def metrics() -> dict[str, Any]:
    return get_metrics().snapshot()


@router.post("/query", response_model=AnswerResponse, tags=["rag"])
async def query(
    request: QueryRequest,
    service: Annotated[Service, Depends(get_service)],
) -> AnswerResponse:
    """Non-streaming RAG query."""
    started = time.perf_counter()

    # ---- input guardrail ------------------------------------------------
    try:
        verdict = service.input_guard.check(request.question)
    except GuardrailViolationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.to_dict()) from exc

    initial = build_initial_state(verdict.clean_text, session_id=request.session_id)
    try:
        final: dict[str, Any] = await service.graph.ainvoke(initial)  # type: ignore[attr-defined]
    except AgenticRAGError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=exc.to_dict()
        ) from exc

    # ---- output guardrail ----------------------------------------------
    out = service.output_guard
    output_verdict = await out.check(
        question=request.question,
        answer=final.get("answer", ""),
        context=final.get("reranked") or None,
    )

    latency = (time.perf_counter() - started) * 1_000
    return AnswerResponse(
        answer=final.get("answer", ""),
        citations=final.get("citations", []),
        route=final.get("route"),  # type: ignore[arg-type]
        reflection_steps=int(final.get("reflection_step", 0)),
        faithfulness_score=output_verdict.faithfulness,
        latency_ms=round(latency, 2),
        tokens=final.get("tokens"),
        metadata={
            "guardrail_notes": output_verdict.notes,
            "output_ok": output_verdict.ok,
        },
    )


@router.post("/query/stream", tags=["rag"])
async def query_stream(
    request: QueryRequest,
    service: Annotated[Service, Depends(get_service)],
) -> EventSourceResponse:
    """SSE-streamed RAG query.

    Emits events:
      * ``node``   — whenever a graph node finishes (includes its name)
      * ``token``  — token delta from the generator (when available)
      * ``done``   — final :class:`AnswerResponse` payload
      * ``error``  — any failure
    """
    try:
        verdict = service.input_guard.check(request.question)
    except GuardrailViolationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.to_dict()) from exc

    initial = build_initial_state(verdict.clean_text, session_id=request.session_id)

    async def _stream():  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        try:
            async for event in service.graph.astream(initial, stream_mode="updates"):  # type: ignore[attr-defined]
                # ``event`` is {node_name: partial_update}
                for node_name, update in event.items():
                    payload = {
                        "node": node_name,
                        "keys": list(update.keys()) if isinstance(update, dict) else [],
                    }
                    yield {"event": "node", "data": json.dumps(payload)}
                    await asyncio.sleep(0)  # allow flush

            # One final pull to get the terminal state — the last event above
            # already produced the answer, but we re-run for structured output.
            final = await service.graph.ainvoke(initial)  # type: ignore[attr-defined]
            out_verdict = await service.output_guard.check(
                question=request.question,
                answer=final.get("answer", ""),
                context=final.get("reranked") or None,
            )
            latency = (time.perf_counter() - started) * 1_000
            response = AnswerResponse(
                answer=final.get("answer", ""),
                citations=final.get("citations", []),
                route=final.get("route"),  # type: ignore[arg-type]
                reflection_steps=int(final.get("reflection_step", 0)),
                faithfulness_score=out_verdict.faithfulness,
                latency_ms=round(latency, 2),
                tokens=final.get("tokens"),
                metadata={"output_ok": out_verdict.ok, "guardrail_notes": out_verdict.notes},
            )
            yield {"event": "done", "data": response.model_dump_json()}
        except AgenticRAGError as exc:
            yield {"event": "error", "data": json.dumps(exc.to_dict())}
        except Exception as exc:  # noqa: BLE001
            log.exception("stream failed")
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(_stream())


__all__ = ["router"]
