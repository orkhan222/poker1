from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def metric_value(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


@dataclass(frozen=True)
class ModelComparisonRow:
    system: str
    family: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    cross_entropy: float
    latency_ms: float | None
    status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "family": self.family,
            "accuracy": rounded(self.accuracy),
            "macro_f1": rounded(self.macro_f1),
            "weighted_f1": rounded(self.weighted_f1),
            "cross_entropy": rounded(self.cross_entropy),
            "latency_ms": None if self.latency_ms is None else rounded(self.latency_ms, 2),
            "status": self.status,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ScopeGate:
    name: str
    status: str
    evidence: str
    risk: str
    next_action: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "risk": self.risk,
            "next_action": self.next_action,
        }


def status_from_metric(value: float, threshold: float, *, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def phase_status(gates: list[ScopeGate]) -> str:
    if not gates:
        return "UNKNOWN"
    if all(gate.status == "PASS" for gate in gates):
        return "PASS"
    if any(gate.status == "FAIL" for gate in gates):
        return "FAIL"
    return "PARTIAL"


def model_row_from_metrics(
    *,
    system: str,
    family: str,
    metrics: dict[str, Any],
    latency_ms: float | None,
    status: str,
    notes: str,
) -> ModelComparisonRow:
    return ModelComparisonRow(
        system=system,
        family=family,
        accuracy=metric_value(metrics, "accuracy"),
        macro_f1=metric_value(metrics, "macro_f1"),
        weighted_f1=metric_value(metrics, "weighted_f1"),
        cross_entropy=metric_value(metrics, "cross_entropy"),
        latency_ms=latency_ms,
        status=status,
        notes=notes,
    )
