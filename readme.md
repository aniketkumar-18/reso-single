Given the complexity, I'll build this as four precise, clean diagrams — one per architectural zone — so every node is readable and no arrows cross boxes. Click any node to explore it deeper.
Diagram 1 of 4 — Ingress & orchestration spine (how a message enters and gets routed)
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.25s;
    animation-timing-function: cubic-bezier(0.19, 1, 0.22, 1);
  }
VvisualizeVvisualize show_widgetDiagram 2 of 4 — Post-execution pipeline (what happens after tool results return, before the response reaches the user)
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.25s;
    animation-timing-function: cubic-bezier(0.19, 1, 0.22, 1);
  }
VvisualizeVvisualize show_widgetDiagram 3 of 4 — Memory & user profile layer (the three-tier memory model and how it feeds back into every agent call)
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.25s;
    animation-timing-function: cubic-bezier(0.19, 1, 0.22, 1);
  }
VvisualizeVvisualize show_widgetDiagram 4 of 4 — Infrastructure & observability (Kubernetes topology, Kafka event bus, and the full 4-signal observability stack)
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.25s;
    animation-timing-function: cubic-bezier(0.19, 1, 0.22, 1);
  }
VvisualizeVvisualize show_widgetThat's the complete enhanced architecture across all four zones. Here's how to read them together:
Diagram 1 is your request path — a message enters, gets hydrated with context, classified by mode, routed to the right agents (1, 2, or all 3 in parallel via LangGraph's Send API), and results are aggregated.
Diagram 2 is your safety and persistence path — after tool results return, the summarizer extracts new facts, the synthesizer merges domains, and critically the constraint validator checks the final response against the user's medical profile before it ever reaches the client. The memory write node then persists everything and publishes fact events to Kafka.
Diagram 3 is your memory model — three independent tiers (short-term sliding window, long-term pgvector embeddings, structured profile with auto-derived constraint rules) all feeding back into the context hydration node in Diagram 1. This is the loop that makes the system smarter over time.
Diagram 4 is your infrastructure — every domain agent is its own independently-scalable FastAPI service, Kafka decouples event streaming with DLQ safety, Supabase holds all persistent data with RLS, Redis handles ephemeral state, and the full 4-signal observability stack (LangSmith + Braintrust + Prometheus + Loki + Grafana) gives you visibility from LLM call level down to pod CPU. Every node in every diagram is clickable if you want to drill into any specific area.Start your own conversation