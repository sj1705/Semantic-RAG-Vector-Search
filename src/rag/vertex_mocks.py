"""Deterministic mocks for ``vertexai.language_models``.

These classes mirror the method signatures and return shapes of the real Vertex AI
SDK so the rest of the pipeline — and the pytest suite — can talk to the same
interface whether the real SDK is installed or not.

Real-world equivalents:

- :class:`TextEmbeddingModel` <-> ``vertexai.language_models.TextEmbeddingModel``
- :class:`GenerativeModel`    <-> ``vertexai.generative_models.GenerativeModel``

The generative mock is *deterministic*: given the same prompt it always returns
the same response. This makes tests reproducible and lets us show meaningful
benchmark deltas between Strategy A and Strategy B without any model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# TextEmbeddingModel mock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextEmbedding:
    """Mirror of ``vertexai.language_models.TextEmbedding``."""

    values: list[float]
    statistics: dict | None = None


class TextEmbeddingModel:
    """Mock of ``vertexai.language_models.TextEmbeddingModel``.

    Uses a hashed bag-of-words projection that yields a deterministic,
    L2-normalized vector per input. The exact numbers are not semantically
    meaningful; the important contract is the **shape** and **method signature**
    so that application code written against this mock also works against the
    real SDK.
    """

    DEFAULT_DIMENSION = 768

    def __init__(self, model_name: str, dimension: int = DEFAULT_DIMENSION) -> None:
        self._model_name = model_name
        self._dimension = int(dimension)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "TextEmbeddingModel":
        return cls(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_embeddings(self, texts: Sequence[str]) -> list[TextEmbedding]:
        out: list[TextEmbedding] = []
        for text in texts:
            vec = self._encode(text)
            out.append(TextEmbedding(values=vec.tolist()))
        return out

    # -- internal -----------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        # Hashed bag-of-words into a deterministic feature space.
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            idx = hash(("mock-embed", token)) % self._dimension
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# GenerativeModel mock
# ---------------------------------------------------------------------------


@dataclass
class GenerateContentResponse:
    """Mirror of the real ``GenerateContentResponse``.

    The only attributes the rest of the pipeline reads are ``text`` and
    ``candidates``; both are provided here.
    """

    text: str
    candidates: list = field(default_factory=list)


# Keyword map used by the mock to produce topic-aware rewrites / HyDE passages.
# Each key is a lowercase token we search for in the incoming query; the
# associated synonyms steer Strategy B toward the right corpus document.
_KEYWORD_SYNONYMS: dict[str, list[str]] = {
    "peak": [
        "autoscaling", "horizontal pod autoscaler", "traffic spikes",
        "rate limiter", "throughput", "burst capacity", "load leveler",
    ],
    "load": ["capacity", "throughput", "concurrency", "scaling"],
    "database": [
        "postgres", "replication", "streaming replication", "standby",
        "failover", "patroni", "availability zone",
    ],
    "fail": ["failover", "promotion", "standby", "disaster recovery"],
    "encrypt": [
        "TLS 1.3", "mutual authentication", "AES-256",
        "KMS", "envelope encryption", "certificates",
    ],
    "data": ["customer data", "personally identifiable", "at rest", "in transit"],
    "protect": ["encryption", "confidentiality", "TLS", "KMS"],
    "transit": ["TLS 1.3", "mutual TLS", "certificates"],
    "observ": [
        "OpenTelemetry traces", "Prometheus metrics", "structured logs",
        "Grafana dashboards", "Alertmanager", "service level objectives",
    ],
    "diagnos": ["incidents", "tracing", "metrics", "logs"],
    "queue": ["Kafka", "backpressure", "dead-letter", "consumer offsets"],
    "deploy": ["Argo Rollouts", "canary", "progressive delivery", "rollback"],
    "rollout": ["canary", "progressive traffic shift", "Argo Rollouts"],
    "cache": ["CDN", "edge cache", "origin shield", "stale-while-revalidate"],
    "session": [
        "OpenID Connect", "JWT access tokens", "refresh tokens",
        "HTTP-only cookies", "session revocation",
    ],
    "token": [
        "JWT", "OIDC", "JWKS", "refresh tokens", "short-lived credentials",
    ],
    "access": [
        "authorization", "role-based access control", "attribute-based rules",
        "policy engine", "multi-factor authentication",
    ],
    "authent": [
        "OpenID Connect", "JWT", "multi-factor authentication", "JWKS",
    ],
    "secret": [
        "HashiCorp Vault", "dynamic secrets", "workload identity",
        "credential rotation", "key rotation",
    ],
    "credential": [
        "Vault", "workload identity", "dynamic secrets",
        "automated rotation",
    ],
    "rotat": [
        "automated rotation", "short-lived credentials",
        "scheduled rotation", "emergency rotation",
    ],
    "flag": [
        "feature flags", "kill switch", "progressive rollout",
        "cohort targeting", "A/B tests",
    ],
    "experiment": [
        "A/B tests", "treatment", "cohort targeting",
        "assignment events", "feature flags",
    ],
    "disaster": [
        "recovery point objective", "recovery time objective",
        "cross-region backups", "transaction-log shipping", "game-day",
    ],
    "backup": [
        "full backups", "transaction-log shipping",
        "cross-region replication", "retention",
    ],
    "recover": [
        "RPO", "RTO", "restore", "runbook", "cross-region",
    ],
    "personal": [
        "personally identifiable information", "GDPR deletion",
        "purpose tags", "data subject requests",
    ],
    "privacy": [
        "GDPR", "data subject requests", "anonymization",
        "purpose tags", "legal basis",
    ],
    "delet": [
        "erasure", "anonymization", "data subject request",
        "statutory deadline", "backups",
    ],
    "notif": [
        "APNs", "FCM", "push notifications", "delivery receipts",
        "quiet hours", "deduplication",
    ],
    "push": [
        "APNs", "FCM", "fan-out", "campaign", "throttle",
    ],
    "mobile": [
        "iOS", "Android", "APNs", "FCM", "push notifications",
    ],
    "cost": [
        "FinOps", "reserved instances", "committed-use discounts",
        "spot instances", "rightsizing",
    ],
    "search": [
        "inverted index", "synonyms", "stemming",
        "learning to rank", "click-through signals",
    ],
}

# Pre-seeded HyDE answers for the benchmark queries. These are deliberately short
# "hypothetical answer passages" in the HyDE style, and are selected by matching
# a trigger phrase in the prompt. The mock falls back to a generic set for any
# query that doesn't hit a trigger.
_HYDE_PASSAGES: dict[str, list[str]] = {
    "peak load": [
        "During peak load the platform adds replicas via a Kubernetes horizontal "
        "pod autoscaler driven by CPU and RPS metrics.",
        "Rate limiting and a queue-based load leveler absorb traffic spikes so "
        "downstream services stay within provisioned throughput.",
        "Burst capacity is achieved through horizontal scaling of stateless "
        "services plus backpressure on request queues.",
    ],
    "database fail": [
        "When the Postgres primary fails, Patroni promotes the synchronous "
        "standby in a second availability zone within thirty seconds.",
        "The service mesh is updated so new connections route to the promoted "
        "standby, and an async remote replica serves disaster recovery.",
        "Streaming replication keeps the hot standby caught up so failover is "
        "near-zero RPO for committed transactions.",
    ],
    "sensitive": [
        "Customer data in transit is protected by TLS 1.3 with mutual "
        "authentication between services and rotated certificates.",
        "At rest, disks are encrypted with AES-256 using customer-managed KMS "
        "keys, plus application-layer envelope encryption on PII fields.",
        "Certificate rotation uses an internal ACME pipeline every ninety days.",
    ],
    "observability": [
        "The observability stack ships structured JSON logs through Fluent Bit "
        "and scrapes Prometheus metrics from every pod.",
        "OpenTelemetry traces are correlated by a shared request ID and "
        "visualized in Grafana with Alertmanager driving paging.",
        "Service level objectives translate user-visible reliability into "
        "paging policy for on-call engineers.",
    ],
    "session": [
        "User sessions are managed with OpenID Connect issuing short-lived JWT "
        "access tokens and rotating refresh tokens stored in HTTP-only cookies.",
        "Authorization runs through a central policy engine evaluating "
        "role-based and attribute-based rules at request time.",
        "Session revocation is propagated within seconds through a pub/sub "
        "invalidation channel so compromised sessions can be killed quickly.",
    ],
    "secret": [
        "Service credentials live in HashiCorp Vault and are fetched at start "
        "via workload identity rather than baked into images.",
        "Dynamic short-lived secrets are preferred over static keys, and every "
        "long-lived secret rotates automatically on a schedule.",
        "Emergency rotation runs as a single operator command that updates "
        "dependent services and restarts pods gracefully.",
    ],
    "flag": [
        "Feature flags are evaluated at the edge and inside services via a "
        "low-latency SDK backed by a consistent-hash store.",
        "Rollouts target cohorts or percentage buckets and emit assignment "
        "events so A/B tests can attribute metric changes to treatments.",
        "Kill switches let operators disable any feature in seconds without "
        "a redeploy, limiting blast radius during incidents.",
    ],
    "disaster": [
        "Every stateful service has documented recovery point and recovery "
        "time objectives backed by cross-region replication.",
        "Daily full backups plus continuous transaction-log shipping land in "
        "a separate account in a different region for isolation.",
        "Quarterly game-day exercises restore the largest databases into an "
        "isolated VPC and verify application-level integrity.",
    ],
    "deletion": [
        "User deletion requests trigger a workflow that erases or anonymizes "
        "records across the online store, warehouse, backups, and analytics.",
        "Personal data is catalogued with purpose tags so each field's legal "
        "basis for processing is explicit.",
        "Access requests are fulfilled by an automated export service that "
        "bundles all personal data into an encrypted archive for the user.",
    ],
    "push": [
        "Push notifications are fanned out through a dedicated service that "
        "batches APNs and FCM deliveries and respects per-user quiet hours.",
        "Identical alerts are deduplicated across devices and delivery "
        "receipts are correlated back to the originating event.",
        "A throttle protects provider APIs during coordinated launches so "
        "burst campaigns don't trip rate limits.",
    ],
}

_GENERIC_HYDE = [
    "The answer involves operational controls, automation, and monitoring.",
    "Production systems address this with layered safeguards and observability.",
    "A runbook covers the detection, mitigation, and recovery steps.",
]


class GenerativeModel:
    """Mock of ``vertexai.generative_models.GenerativeModel``.

    Recognises two prompt shapes emitted by :class:`rag.query_expansion.QueryExpander`:

    * ``REWRITE: <user query>``   → returns a single domain-enriched rewrite.
    * ``HYDE: <user query>``      → returns three hypothetical answer passages
      separated by ``---``.

    Unknown prompts echo the input so callers still get a usable string.
    """

    _REWRITE_RE = re.compile(r"^REWRITE:\s*(.*)$", re.DOTALL)
    _HYDE_RE = re.compile(r"^HYDE:\s*(.*)$", re.DOTALL)

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_content(self, prompt: str) -> GenerateContentResponse:
        if (m := self._REWRITE_RE.match(prompt)):
            user_query = m.group(1).strip()
            return GenerateContentResponse(text=self._rewrite(user_query))
        if (m := self._HYDE_RE.match(prompt)):
            user_query = m.group(1).strip()
            return GenerateContentResponse(text=self._hyde(user_query))
        return GenerateContentResponse(text=prompt)

    # -- rewrite strategy ---------------------------------------------------

    def _rewrite(self, query: str) -> str:
        lowered = query.lower()
        extras: list[str] = []
        seen: set[str] = set()
        for key, synonyms in _KEYWORD_SYNONYMS.items():
            if key in lowered:
                for s in synonyms:
                    if s.lower() not in seen:
                        extras.append(s)
                        seen.add(s.lower())
        if not extras:
            return query
        return f"{query} [context: {', '.join(extras)}]"

    # -- HyDE strategy ------------------------------------------------------

    def _hyde(self, query: str) -> str:
        lowered = query.lower()
        for trigger, passages in _HYDE_PASSAGES.items():
            if trigger in lowered:
                return "\n---\n".join(passages)
        # No topical trigger: still produce three passages so callers can rely
        # on the shape of the response.
        return "\n---\n".join(_GENERIC_HYDE)


__all__ = [
    "TextEmbedding",
    "TextEmbeddingModel",
    "GenerateContentResponse",
    "GenerativeModel",
]
