from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchitectureOption:
    name: str
    production_fit: str
    strengths: list[str]
    limitations: list[str]
    operational_risk: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "production_fit": self.production_fit,
            "strengths": list(self.strengths),
            "limitations": list(self.limitations),
            "operational_risk": self.operational_risk,
        }


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    parameter_count: str
    role: str
    expected_use: str
    hardware_profile: str
    fine_tuning_fit: str
    risk: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "role": self.role,
            "expected_use": self.expected_use,
            "hardware_profile": self.hardware_profile,
            "fine_tuning_fit": self.fine_tuning_fit,
            "risk": self.risk,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class AgentBuildRecommendation:
    selected_architecture: str
    selected_model: str
    approval_status: str
    adapter_available: bool
    architecture_metrics: dict[str, Any] = field(default_factory=dict)
    architecture_options: list[ArchitectureOption] = field(default_factory=list)
    model_candidates: list[ModelCandidate] = field(default_factory=list)
    implementation_plan: list[str] = field(default_factory=list)
    acceptance_gates: list[str] = field(default_factory=list)

    @property
    def ready_for_stakeholder_review(self) -> bool:
        return bool(self.selected_architecture and self.selected_model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_architecture": self.selected_architecture,
            "selected_model": self.selected_model,
            "approval_status": self.approval_status,
            "adapter_available": self.adapter_available,
            "ready_for_stakeholder_review": self.ready_for_stakeholder_review,
            "architecture_metrics": self.architecture_metrics,
            "architecture_options": [item.to_dict() for item in self.architecture_options],
            "model_candidates": [item.to_dict() for item in sorted(self.model_candidates, key=lambda item: item.priority)],
            "implementation_plan": list(self.implementation_plan),
            "acceptance_gates": list(self.acceptance_gates),
        }


def default_architecture_options() -> list[ArchitectureOption]:
    return [
        ArchitectureOption(
            name="deterministic_parser",
            production_fit="Strong baseline, insufficient as the only path for corrupted OCR.",
            strengths=[
                "lowest latency",
                "fully deterministic",
                "easy to validate and debug",
            ],
            limitations=[
                "limited recovery on ambiguous OCR",
                "requires manual rule expansion",
            ],
            operational_risk="Low",
        ),
        ArchitectureOption(
            name="qlora_extractor",
            production_fit="Useful research baseline, not the safest first production path.",
            strengths=[
                "can recover noisy text patterns",
                "uses trained domain adapter",
            ],
            limitations=[
                "schema validity can degrade under malformed inputs",
                "higher latency and memory use than rules",
            ],
            operational_risk="Medium",
        ),
        ArchitectureOption(
            name="candidate_ranker",
            production_fit="Good controlled-output alternative when candidate generation is reliable.",
            strengths=[
                "bounded output space",
                "lower risk than free-form generation",
                "simple evaluation",
            ],
            limitations=[
                "cannot select the correct answer if no candidate is generated",
                "depends on parser coverage",
            ],
            operational_risk="Low to medium",
        ),
        ArchitectureOption(
            name="hybrid_parser_qlora",
            production_fit="Recommended production candidate for the current milestone.",
            strengths=[
                "keeps deterministic behavior for clean events",
                "uses LLM fallback only when needed",
                "validates every output against schema",
                "minimizes latency by lazy-loading the adapter",
            ],
            limitations=[
                "requires curated fallback examples for continued improvement",
                "does not replace the separate poker decision policy",
            ],
            operational_risk="Low to medium",
        ),
        ArchitectureOption(
            name="autonomous_decision_llm",
            production_fit="Not recommended for this milestone.",
            strengths=[
                "simple conceptual interface",
                "can reason over textual context",
            ],
            limitations=[
                "difficult to validate",
                "weak control over strategic outputs",
                "does not solve missing-card and data-quality issues",
            ],
            operational_risk="High",
        ),
    ]


def default_model_candidates(primary_model: str) -> list[ModelCandidate]:
    return [
        ModelCandidate(
            name=primary_model or "Qwen/Qwen2.5-1.5B-Instruct",
            parameter_count="1.5B",
            role="primary",
            expected_use="QLoRA adapter for event normalization fallback and candidate ranking.",
            hardware_profile="Runs locally with 4-bit quantization; GPU preferred for training, CPU acceptable for low-throughput fallback tests.",
            fine_tuning_fit="Strong fit for schema-guided extraction with small LoRA adapters.",
            risk="Needs a larger reviewed OCR/dealer-log set before production promotion.",
            priority=1,
        ),
        ModelCandidate(
            name="Qwen/Qwen3-1.7B",
            parameter_count="1.7B",
            role="comparison",
            expected_use="Second Qwen-family benchmark for structured extraction and fallback ranking.",
            hardware_profile="Similar deployment profile to Qwen2.5 1.5B under quantized inference.",
            fine_tuning_fit="Good LoRA candidate; should be benchmarked before replacing the primary model.",
            risk="Model-family behavior must be validated against the same schema gates.",
            priority=2,
        ),
        ModelCandidate(
            name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            parameter_count="1.7B",
            role="lightweight baseline",
            expected_use="Local baseline for latency and cost comparisons.",
            hardware_profile="Lightweight inference profile; practical for local reproducibility tests.",
            fine_tuning_fit="Usable LoRA baseline, but expected extraction quality is lower than Qwen on structured outputs.",
            risk="Previous zero-shot and few-shot runs were weak; do not select without fine-tuned evidence.",
            priority=3,
        ),
        ModelCandidate(
            name="google/gemma-3-1b-it",
            parameter_count="1B",
            role="fallback candidate",
            expected_use="Cost-sensitive fallback benchmark if deployment footprint dominates.",
            hardware_profile="Smallest candidate; lower memory pressure.",
            fine_tuning_fit="Useful for constrained extraction, but quality ceiling may be lower.",
            risk="License and deployment constraints must be reviewed before production use.",
            priority=4,
        ),
        ModelCandidate(
            name="meta-llama/Llama-3.2-1B-Instruct",
            parameter_count="1B",
            role="secondary fallback",
            expected_use="Edge-model comparison only after license and lifecycle checks.",
            hardware_profile="Small inference footprint.",
            fine_tuning_fit="Technically suitable for LoRA, but not the default recommendation.",
            risk="Access, license, and lifecycle constraints make it lower priority.",
            priority=5,
        ),
    ]


def adapter_artifact_available(model_dir: Path) -> bool:
    return (model_dir / "adapter_model.safetensors").exists() and (model_dir / "adapter_config.json").exists()


def select_metrics(decision_payload: dict[str, Any], selected_architecture: str) -> dict[str, Any]:
    decision_metrics = decision_payload.get("decision", {}).get("metrics")
    if isinstance(decision_metrics, dict) and decision_metrics.get("architecture") == selected_architecture:
        return decision_metrics
    for row in decision_payload.get("comparison", []):
        if isinstance(row, dict) and row.get("architecture") == selected_architecture:
            return row
    return {}


def build_recommendation(
    *,
    config: dict[str, Any],
    decision_payload: dict[str, Any],
    model_dir: Path,
    approval_status: str,
) -> AgentBuildRecommendation:
    configured_agent = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    selected_architecture = str(
        decision_payload.get("selected_architecture")
        or decision_payload.get("decision", {}).get("selected_architecture")
        or configured_agent.get("selected_architecture")
        or "hybrid_parser_qlora"
    )
    selected_model = str(config.get("base_model") or "Qwen/Qwen2.5-1.5B-Instruct")
    metrics = select_metrics(decision_payload, selected_architecture)
    return AgentBuildRecommendation(
        selected_architecture=selected_architecture,
        selected_model=selected_model,
        approval_status=approval_status,
        adapter_available=adapter_artifact_available(model_dir),
        architecture_metrics=metrics,
        architecture_options=default_architecture_options(),
        model_candidates=default_model_candidates(selected_model),
        implementation_plan=[
            "Freeze stakeholder-approved architecture and model family in config.",
            "Use deterministic parser for clean OCR/dealer-log records.",
            "Route incomplete or low-confidence events to the QLoRA fallback.",
            "Validate every event with the JSON schema before downstream ingestion.",
            "Log route, confidence, latency, schema validity, and unmatched rate for monitoring.",
            "Keep the poker decision policy separate from the event-normalization agent.",
        ],
        acceptance_gates=[
            "schema_validity_rate >= configured threshold",
            "simulation_readiness_rate >= configured threshold",
            "unmatched_rate within configured threshold",
            "average latency acceptable for service integration",
            "adapter artifact present before transformer fallback is enabled",
            "stakeholder approval recorded before building the production service path",
        ],
    )
