# LangGraph

LangGraph is a framework for building stateful, multi-actor applications with large language models. It models each agent as a finite-state machine where each **node** represents a reasoning or tool-use step and **edges** define control flow. Edges can be conditional — they consult the graph state to decide which node to run next, which is what enables cyclic workflows (e.g. self-reflection loops) without the fragility of hand-rolled conditionals.

State in LangGraph is a typed dictionary. Every node receives the full state and returns a *partial* update that the framework merges in. Reducers can be attached to specific keys to define *how* updates compose — for example, `add_messages` appends to a conversation history rather than overwriting it.

LangGraph is designed to complement LangChain rather than replace it. Where LangChain provides chains, memory, and tool integrations, LangGraph provides the orchestration layer that lets you wire those building blocks into complex, observable agent loops.

Why choose LangGraph over a DIY state machine?

- **Explicit state**: state is a schema; every transition is visible.
- **Checkpointing**: graphs can be persisted and resumed, which is essential for long-running agents and human-in-the-loop workflows.
- **Streaming**: nodes can stream token deltas to the caller as they run.
- **Observability**: each node invocation is trivially traceable with OpenTelemetry.
