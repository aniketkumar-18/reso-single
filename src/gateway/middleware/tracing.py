"""OpenTelemetry tracing setup.

Initialises the OTEL SDK on startup and provides a helper to inject
trace context into each request's state.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.infra.config import get_settings
from src.infra.logger import setup_logger

logger = setup_logger(__name__)
_tracer: trace.Tracer | None = None


def setup_tracing(app) -> None:  # type: ignore[type-arg]
    """Call once at app startup to configure OTEL SDK."""
    settings = get_settings()
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter_otlp_endpoint:
        try:
            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTEL exporter configured → %s", settings.otel_exporter_otlp_endpoint)
        except Exception:
            logger.warning("OTEL exporter unavailable — traces will not be exported")
    else:
        logger.info("OTEL exporter disabled (no endpoint configured)")

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    global _tracer
    _tracer = trace.get_tracer(settings.otel_service_name)
    logger.info("Tracing initialised for service '%s'", settings.otel_service_name)


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("reso")
    return _tracer


def get_current_trace_id() -> str:
    """Return the current OTEL trace_id as a hex string, or empty string."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        return format(ctx.trace_id, "032x")
    return ""
