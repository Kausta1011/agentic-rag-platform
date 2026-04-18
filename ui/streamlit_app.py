"""Streamlit demo UI for the Agentic RAG API.

Talks to the FastAPI backend over HTTP / SSE. Run it via::

    streamlit run ui/streamlit_app.py

The UI is deliberately thin — backend owns *all* business logic; the UI
only formats requests and responses. This keeps the project's unit of
review small (the backend) and makes the demo honest.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get("AGENTIC_RAG_API", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Agentic RAG", layout="wide", page_icon="🔎")


# ---------------------------------------------------------------------------
# sidebar: config + status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Agentic RAG Platform")
    st.caption("LangGraph · Hybrid retrieval · Cross-encoder · Reflection · Guardrails")
    api_base = st.text_input("API base URL", value=API_BASE)
    use_stream = st.toggle("Stream (SSE)", value=True, help="Use server-sent-event streaming")
    session_id = st.text_input("Session id", value="demo-session")

    try:
        r = httpx.get(f"{api_base}/healthz", timeout=3)
        if r.status_code == 200:
            st.success("API healthy")
        else:
            st.warning(f"API status: {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API unreachable: {exc}")

    if st.button("Refresh metrics"):
        try:
            m = httpx.get(f"{api_base}/metrics", timeout=3).json()
            st.json(m)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


# ---------------------------------------------------------------------------
# main chat
# ---------------------------------------------------------------------------
st.title("Agentic RAG — ask your corpus")
st.write(
    "Queries are routed (RAG / web / direct), rewritten, retrieved with hybrid "
    "BM25+dense, reranked by a cross-encoder, grounded-generated, and "
    "self-reflected. Citations are inline."
)

if "history" not in st.session_state:
    st.session_state["history"] = []

# Re-render history
for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("meta"):
            with st.expander("Details"):
                st.json(turn["meta"])


def _render_answer(resp: dict[str, Any]) -> None:
    answer = resp.get("answer", "")
    st.markdown(answer)

    citations = resp.get("citations", [])
    if citations:
        with st.expander(f"Citations ({len(citations)})"):
            for i, c in enumerate(citations, start=1):
                st.markdown(f"**[{i}]** *{c.get('source', '?')}*")
                st.caption(c.get("snippet", ""))
    meta = {
        "route": resp.get("route"),
        "reflection_steps": resp.get("reflection_steps"),
        "faithfulness_score": resp.get("faithfulness_score"),
        "latency_ms": resp.get("latency_ms"),
        "tokens": resp.get("tokens"),
        "metadata": resp.get("metadata"),
    }
    with st.expander("Details"):
        st.json(meta)
    st.session_state["history"].append(
        {"role": "assistant", "content": answer, "meta": meta}
    )


def _post_json(url: str, body: dict) -> dict[str, Any]:
    with httpx.Client(timeout=120) as client:
        r = client.post(url, json=body)
        r.raise_for_status()
        return r.json()


def _post_stream(url: str, body: dict) -> dict[str, Any] | None:
    """Read an SSE stream, showing node events, return the final payload."""
    final_payload: dict[str, Any] | None = None
    placeholder = st.empty()
    log_lines: list[str] = []

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", url, json=body) as r:
            r.raise_for_status()
            event_name = "message"
            for raw in r.iter_lines():
                if not raw:
                    continue
                if raw.startswith("event:"):
                    event_name = raw.split(":", 1)[1].strip()
                elif raw.startswith("data:"):
                    data_str = raw.split(":", 1)[1].strip()
                    if event_name == "node":
                        try:
                            data = json.loads(data_str)
                            log_lines.append(f"· **{data.get('node')}** → {data.get('keys')}")
                            placeholder.markdown("\n".join(log_lines[-10:]))
                        except json.JSONDecodeError:
                            pass
                    elif event_name == "done":
                        try:
                            final_payload = json.loads(data_str)
                        except json.JSONDecodeError:
                            pass
                    elif event_name == "error":
                        st.error(data_str)
                        return None
    placeholder.empty()
    return final_payload


prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    body = {"question": prompt, "session_id": session_id, "stream": use_stream}

    with st.chat_message("assistant"):
        try:
            if use_stream:
                payload = _post_stream(f"{api_base}/query/stream", body)
            else:
                payload = _post_json(f"{api_base}/query", body)

            if payload:
                _render_answer(payload)
            else:
                st.error("No response payload received.")
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
