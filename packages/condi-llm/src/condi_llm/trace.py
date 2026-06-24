"""OpenTelemetry-native tracing for every forward pass, KV hit, and step."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional


# Lazily initialised; falls back to a no-op if opentelemetry isn't installed.
_tracer: Optional[Any] = None
_exporter: Optional[Any] = None
_enabled: bool = False


def configure(endpoint: str = "http://localhost:4317", service_name: str = "condi-llm") -> None:
    """Configure the OTLP exporter. Safe to call multiple times."""
    global _tracer, _exporter, _enabled
    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider = TracerProvider(resource=_make_resource(service_name))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        ot_trace.set_tracer_provider(provider)
        _tracer = ot_trace.get_tracer("condi-llm")
        _enabled = True
    except Exception:
        # No opentelemetry installed — tracing becomes a silent no-op.
        _enabled = False


def _make_resource(service_name: str):
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    return Resource.create({SERVICE_NAME: service_name})


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """A traced span. Becomes a contextmanager no-op when tracing is disabled."""
    if _enabled and _tracer is not None:
        with _tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, v)
            yield s
    else:
        yield None


def set_attr(key: str, value: Any) -> None:
    if _enabled:
        from opentelemetry import trace as ot_trace
        s = ot_trace.get_current_span()
        if s:
            s.set_attribute(key, value)


def log(payload: Dict[str, Any]) -> None:
    """Emit a structured log event on the current span (or print when disabled)."""
    if _enabled:
        from opentelemetry import trace as ot_trace
        s = ot_trace.get_current_span()
        if s:
            s.add_event("log", attributes={k: str(v) for k, v in payload.items()})
    else:
        print(f"[condi-llm] {payload}")
