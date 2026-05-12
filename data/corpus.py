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
    Document(
        id="doc-09",
        title="Authentication and Authorization",
        text=(
            "User authentication uses OpenID Connect with short-lived JWT access "
            "tokens and rotating refresh tokens stored as HTTP-only cookies. "
            "Authorization is enforced by a central policy engine evaluating "
            "role-based and attribute-based rules, and every service validates "
            "tokens locally with cached JWKS keys. Multi-factor authentication "
            "is required for privileged roles, and session revocation propagates "
            "through a pub/sub invalidation channel within seconds."
        ),
    ),
    Document(
        id="doc-10",
        title="Secrets Management and Key Rotation",
        text=(
            "All service credentials, database passwords, and API tokens live in "
            "HashiCorp Vault and are fetched at process start via workload identity. "
            "Short-lived dynamic secrets are preferred over static keys, and every "
            "long-lived secret is rotated automatically on a schedule through a "
            "controller that updates dependent services and restarts pods "
            "gracefully. Emergency rotation is a single command and completes in "
            "under five minutes."
        ),
    ),
    Document(
        id="doc-11",
        title="Feature Flags and Experimentation",
        text=(
            "Feature flags are evaluated at the edge and inside services via a "
            "low-latency SDK backed by a consistent-hash store. Rollouts target "
            "user cohorts, geographies, or percentage-based buckets, and every "
            "flag emits an assignment event into the experimentation pipeline so "
            "A/B tests can attribute metric changes back to specific treatments. "
            "Kill switches let operators disable any feature in seconds without "
            "a redeploy."
        ),
    ),
    Document(
        id="doc-12",
        title="Cost Optimization and FinOps",
        text=(
            "Cloud spend is tagged per team, service, and environment, and a "
            "nightly FinOps job emits a per-team dashboard with anomaly alerts. "
            "Reserved instances and committed-use discounts cover predictable "
            "baseline workloads, while spot and preemptible nodes absorb bursty "
            "batch jobs. Rightsizing recommendations are generated weekly from "
            "utilization telemetry, and idle resources are reaped automatically "
            "after a grace period."
        ),
    ),
    Document(
        id="doc-13",
        title="Data Privacy and GDPR Deletion",
        text=(
            "Personal data is catalogued with purpose tags so each field's legal "
            "basis is explicit. User deletion requests trigger a workflow that "
            "erases or anonymizes records across the online store, the warehouse, "
            "backups, and downstream analytics within the statutory deadline. "
            "Access requests are fulfilled by an automated export service that "
            "bundles all personal data into an encrypted archive for the user."
        ),
    ),
    Document(
        id="doc-14",
        title="Disaster Recovery and Backup Strategy",
        text=(
            "Every stateful service has a documented recovery point and recovery "
            "time objective. Daily full backups and continuous transaction-log "
            "shipping land in a separate account in a different region. Quarterly "
            "game-day exercises restore the largest databases into an isolated "
            "VPC and verify application-level integrity before signing off on "
            "the runbook. Backup artifacts are encrypted and their retention "
            "matches the data classification policy."
        ),
    ),
    Document(
        id="doc-15",
        title="Search Indexing and Query Performance",
        text=(
            "Product search is powered by an inverted index refreshed from the "
            "catalog via a change-data-capture stream. Synonyms and stemming are "
            "tuned per locale, and a reranking layer applies learning-to-rank "
            "signals from click-through data. Query-side caching collapses "
            "identical searches within a short window, and a circuit breaker "
            "falls back to a minimal result set when the cluster is degraded."
        ),
    ),
    Document(
        id="doc-16",
        title="Mobile Push Notifications and Delivery",
        text=(
            "Push notifications are fanned out through a dedicated notification "
            "service that batches APNs and FCM deliveries, respects per-user "
            "quiet hours, and deduplicates identical alerts across devices. "
            "Delivery receipts are correlated back to the originating event so "
            "campaign owners can measure open rates, and a throttle protects "
            "the provider APIs during coordinated launches."
        ),
    ),
)


__all__ = ["Document", "CORPUS"]
