"""Prometheus metrics and optional OpenTelemetry tracing for the gateway.

Uses an isolated `CollectorRegistry` per gateway instance so multiple app
instances in tests or multi-worker setups never collide on Prometheus globals.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class GatewayTelemetry:
    """Per-gateway Prometheus metrics and optional OTel span helper."""

    def __init__(self, *, service_name: str = "henri-v6-gateway", enable_otel: bool = False) -> None:
        self.service_name = str(service_name)
        self.enable_otel = bool(enable_otel)
        self.registry = CollectorRegistry()

        self.requests_total = Counter(
            "henri_gateway_requests_total",
            "Total HTTP/MCP requests handled by the HENRI-v6 gateway",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "henri_gateway_request_duration_seconds",
            "End-to-end request latency in seconds",
            ["endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0),
            registry=self.registry,
        )
        self.cascade_total = Counter(
            "henri_gateway_cascade_total",
            "Arm0->Arm4 cascade outcomes",
            ["endpoint", "outcome"],
            registry=self.registry,
        )
        self.veto_total = Counter(
            "henri_gateway_veto_total",
            "Sagnac/ontology/AST veto decisions",
            ["endpoint", "vetoed"],
            registry=self.registry,
        )
        self.spend_calls_total = Counter(
            "henri_gateway_spend_calls_total",
            "Arm4 external LLM calls reserved via SpendGuard",
            ["tenant_id", "outcome"],
            registry=self.registry,
        )
        self.sandbox_exec_total = Counter(
            "henri_gateway_sandbox_exec_total",
            "Sandboxed execution invocations",
            ["mode", "outcome"],
            registry=self.registry,
        )
        self.active_compute_jobs = Gauge(
            "henri_gateway_active_compute_jobs",
            "Current jobs executing in the bounded compute worker pool",
            registry=self.registry,
        )

        self._tracer = None
        if self.enable_otel:
            try:
                from opentelemetry import trace

                self._tracer = trace.get_tracer(self.service_name)
            except Exception:
                self._tracer = None

    def render_prometheus(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
        if self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name) as current:
            if attributes:
                for key, val in attributes.items():
                    try:
                        current.set_attribute(str(key), val)
                    except Exception:
                        pass
            yield current
