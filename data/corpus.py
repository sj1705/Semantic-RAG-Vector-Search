"""Technical corpus used by the RAG benchmark.

Sixteen short passages covering distinct operational topics. Queries in the
benchmark target specific topics so the ranked results are interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """A single corpus document."""

    id: str
    title: str
    text: str


CORPUS: tuple[Document, ...] = (
    Document(
        id="doc-01",
        title="Autoscaling and Peak Traffic",
        text=(
            "The platform handles peak load by horizontally scaling stateless services "
            "through a Kubernetes Horizontal Pod Autoscaler driven by CPU utilization and "
            "a custom requests-per-second metric. During traffic spikes the autoscaler "
            "provisions additional replicas, while a global rate limiter sheds abusive "
            "traffic and a queue-based load leveler absorbs short bursts so downstream "
            "services never see more than their provisioned throughput."
        ),
    ),
    Document(
        id="doc-02",
        title="Edge Caching and CDN Strategy",
        text=(
            "Static and semi-static responses are served from a global content delivery "
            "network with a multi-tier cache hierarchy. Hot objects are pinned in the "
            "edge tier, while origin shield nodes collapse cache misses to protect the "
            "origin. Cache-Control and surrogate-control headers give product teams "
            "fine-grained control over TTLs and stale-while-revalidate behavior."
        ),
    ),
    Document(
        id="doc-03",
        title="Database Replication and Failover",
        text=(
            "The primary Postgres cluster uses synchronous streaming replication to a "
            "hot standby in a second availability zone. If the primary becomes "
            "unreachable, Patroni promotes the standby within thirty seconds, updates "
            "the service mesh, and new connections are routed automatically. A separate "
            "asynchronous replica in a remote region serves as the disaster-recovery "
            "target and is used for point-in-time restores."
        ),
    ),
    Document(
        id="doc-04",
        title="Data Encryption in Transit and at Rest",
        text=(
            "All customer data is protected by TLS 1.3 in transit using mutual "
            "authentication between services, with certificates rotated every ninety "
            "days through an internal ACME pipeline. At rest, disks are encrypted with "
            "AES-256 via customer-managed KMS keys, and application-layer envelope "
            "encryption is applied to personally identifiable fields before they reach "
            "the database."
        ),
    ),
    Document(
        id="doc-05",
        title="Observability: Logs, Metrics, and Traces",
        text=(
            "The observability stack combines structured JSON logs shipped through "
            "Fluent Bit, Prometheus metrics scraped from every pod, and OpenTelemetry "
            "distributed traces correlated by a shared request ID. Dashboards in "
            "Grafana and alerts in Alertmanager give on-call engineers a unified view "
            "when diagnosing incidents, and service-level objectives drive paging "
            "policy."
        ),
    ),
    Document(
        id="doc-06",
        title="Message Queues and Backpressure",
        text=(
            "Asynchronous workloads flow through Kafka topics partitioned by tenant. "
            "Consumers use cooperative rebalancing and commit offsets only after "
            "successful processing. When a downstream dependency slows down, consumers "
            "apply backpressure by reducing poll frequency, and a dead-letter queue "
            "captures poison messages after a bounded number of retries so a single "
            "bad payload never stalls the pipeline."
        ),
    ),
    Document(
        id="doc-07",
        title="Continuous Delivery and Progressive Rollouts",
        text=(
            "Every merge to main triggers a CI pipeline that runs unit, integration, "
            "and contract tests before building a container image. Deployments use "
            "Argo Rollouts to progressively shift traffic through canary stages gated "
            "by service-level indicators, and any regression in error rate or latency "
            "triggers an automatic rollback to the last known-good revision."
        ),
    ),
    Document(
        id="doc-08",
        title="Billing and Invoicing Pipeline",
        text=(
            "Usage events from every service are aggregated nightly into the billing "
            "data warehouse. Rating rules convert raw usage into billable line items, "
            "and a separate invoicing service generates PDF invoices, pushes them to "
            "the customer portal, and reconciles payments against the accounts "
            "receivable ledger. This pipeline is intentionally decoupled from the "
            "online transactional path."
        ),
    ),
)


__all__ = ["Document", "CORPUS"]
