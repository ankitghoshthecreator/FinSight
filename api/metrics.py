"""Prometheus-compatible and JSON observability metrics for FinSight RAG microservice."""

import time
from typing import Dict, Any


class FinSightMetricsCollector:
    """Collects real-time operational and model guardrail metrics."""

    def __init__(self):
        self.request_count: int = 0
        self.endpoint_counts: Dict[str, int] = {}
        self.status_code_counts: Dict[int, int] = {}
        self.total_queries: int = 0
        self.escalations_triggered: int = 0
        self.pii_redactions_total: int = 0
        self.chunks_indexed_total: int = 0
        self.query_latencies: list[float] = []

    def record_request(self, endpoint: str, status_code: int, latency_ms: float):
        self.request_count += 1
        self.endpoint_counts[endpoint] = self.endpoint_counts.get(endpoint, 0) + 1
        self.status_code_counts[status_code] = self.status_code_counts.get(status_code, 0) + 1

    def record_query(self, latency_ms: float, escalated: bool):
        self.total_queries += 1
        if escalated:
            self.escalations_triggered += 1
        self.query_latencies.append(latency_ms)
        # Retain last 1,000 latency measurements to prevent memory leak
        if len(self.query_latencies) > 1000:
            self.query_latencies.pop(0)

    def record_pii_redacted(self, count: int):
        self.pii_redactions_total += count

    def record_chunks_indexed(self, count: int):
        self.chunks_indexed_total += count

    def get_summary(self) -> Dict[str, Any]:
        """Returns JSON operational summary."""
        escalation_rate = (
            round((self.escalations_triggered / self.total_queries) * 100, 2)
            if self.total_queries > 0 else 0.0
        )
        avg_latency = (
            round(sum(self.query_latencies) / len(self.query_latencies), 2)
            if self.query_latencies else 0.0
        )
        p95_latency = (
            round(sorted(self.query_latencies)[int(len(self.query_latencies) * 0.95)], 2)
            if len(self.query_latencies) >= 20 else avg_latency
        )

        return {
            "total_requests": self.request_count,
            "total_queries": self.total_queries,
            "escalations_count": self.escalations_triggered,
            "escalation_rate_pct": escalation_rate,
            "avg_query_latency_ms": avg_latency,
            "p95_query_latency_ms": p95_latency,
            "pii_entities_redacted_total": self.pii_redactions_total,
            "chunks_indexed_total": self.chunks_indexed_total,
            "endpoints": self.endpoint_counts,
            "status_codes": {str(k): v for k, v in self.status_code_counts.items()},
        }

    def to_prometheus_format(self) -> str:
        """Exports Prometheus text exposition format."""
        summary = self.get_summary()
        lines = [
            "# HELP finsight_http_requests_total Total HTTP requests served.",
            "# TYPE finsight_http_requests_total counter",
            f"finsight_http_requests_total {summary['total_requests']}",
            "",
            "# HELP finsight_rag_queries_total Total RAG QA queries processed.",
            "# TYPE finsight_rag_queries_total counter",
            f"finsight_rag_queries_total {summary['total_queries']}",
            "",
            "# HELP finsight_rag_escalations_total Total queries triggering hallucination/insufficient context guardrail.",
            "# TYPE finsight_rag_escalations_total counter",
            f"finsight_rag_escalations_total {summary['escalations_count']}",
            "",
            "# HELP finsight_pii_redactions_total Total sensitive PII entities redacted.",
            "# TYPE finsight_pii_redactions_total counter",
            f"finsight_pii_redactions_total {summary['pii_entities_redacted_total']}",
            "",
            "# HELP finsight_query_latency_p95_ms 95th percentile query latency in milliseconds.",
            "# TYPE finsight_query_latency_p95_ms gauge",
            f"finsight_query_latency_p95_ms {summary['p95_query_latency_ms']}",
        ]
        return "\n".join(lines) + "\n"


metrics_collector = FinSightMetricsCollector()
