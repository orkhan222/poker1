from __future__ import annotations

from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "2026-06-19"


PREDICTION_RESPONSE_FIELDS: dict[str, dict[str, Any]] = {
    "action": {
        "type": "string",
        "allowed_values": ["fold", "call", "check", "bet", "raise"],
        "description": "Primary poker action selected by the policy for the submitted game state.",
        "engineering_note": "This is the top-ranked action after model inference and post-processing.",
    },
    "probabilities": {
        "type": "object[string,float]",
        "description": "Normalized probability distribution over the supported action space.",
        "engineering_note": "Values must sum to 1.0 within floating-point tolerance.",
    },
    "confidence": {
        "type": "float",
        "range": [0.0, 1.0],
        "description": "Confidence assigned to the selected action.",
        "engineering_note": "Usually equal to the highest probability in the response distribution.",
    },
    "bet_size": {
        "type": "float",
        "description": "Chip amount recommended for the selected action.",
        "engineering_note": "Zero for fold/check, call price for call, and a constrained sizing estimate for bet/raise.",
    },
    "wait_time_ms": {
        "type": "integer",
        "description": "Recommended delay before executing the action, expressed in milliseconds.",
        "engineering_note": "Calibrated from action complexity, street, betting history depth, confidence, and processing time.",
    },
    "sizing_method": {
        "type": "string",
        "description": "Name of the deterministic sizing policy used to produce bet_size.",
        "known_values": [
            "no_chip_commitment",
            "call_price",
            "bet_mapped_to_call_price",
            "pot_fraction_bet",
            "pressure_raise",
            "unsupported_action",
        ],
    },
    "timing_method": {
        "type": "string",
        "description": "Name of the deterministic timing policy used to produce wait_time_ms.",
        "known_values": ["complexity_calibrated"],
    },
    "model_status": {
        "type": "string",
        "description": "Model or fallback path that produced the response.",
        "engineering_note": "Used for observability and debugging, not as a user-facing strategy signal.",
    },
    "warnings": {
        "type": "array[string]",
        "description": "Optional warnings emitted when inference uses a conservative fallback or degraded input path.",
    },
}


DELIVERY_STATUS_FIELDS: dict[str, dict[str, str]] = {
    "delivery_verification=PASS": {
        "meaning": "The project delivery contract passed required file, compile, inference, report, hygiene, provenance, and ZIP checks.",
        "implication": "The package is suitable for technical handoff.",
    },
    "repo_hygiene=PASS": {
        "meaning": "The repository passed the client-delivery hygiene scan.",
        "implication": "Generated runtime artifacts and local tool metadata are not part of the deliverable.",
    },
    "zip_contract=PASS": {
        "meaning": "The release archive contains the required source, configs, reports, model artifacts, and verification files.",
        "implication": "The ZIP can be shared as the current delivery package.",
    },
    "production_gate=FAIL": {
        "meaning": "The software delivery passed, but the strategic poker policy is not approved as a production strategy model.",
        "implication": "The service and pipeline are usable for technical handoff; strategic deployment remains blocked until the readiness gates pass.",
    },
}


def prediction_response_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "response_fields": deepcopy(PREDICTION_RESPONSE_FIELDS),
        "example_response": {
            "action": "raise",
            "probabilities": {
                "fold": 0.02,
                "call": 0.24,
                "check": 0.04,
                "bet": 0.18,
                "raise": 0.52,
            },
            "confidence": 0.52,
            "bet_size": 4.5,
            "wait_time_ms": 1264,
            "sizing_method": "pressure_raise",
            "timing_method": "complexity_calibrated",
            "model_status": "hist_gradient_boosting",
        },
    }


def delivery_status_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "delivery_status_fields": deepcopy(DELIVERY_STATUS_FIELDS),
    }


def api_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "prediction_response": prediction_response_contract(),
        "delivery_status": delivery_status_contract(),
        "strategy_readiness": {
            "endpoint": "/strategy-readiness.json",
            "status_values": ["APPROVED", "NOT_APPROVED", "UNKNOWN"],
            "deployment_modes": ["production_policy", "technical_handoff_only", "unavailable"],
            "description": "Machine-readable approval boundary for the strategic poker policy.",
        },
        "delivery_readiness": {
            "endpoint": "/delivery-readiness.json",
            "overall_status_values": [
                "READY_FOR_TECHNICAL_HANDOFF",
                "READY_FOR_PRODUCTION_POLICY",
                "NOT_READY_FOR_HANDOFF",
            ],
            "description": "Machine-readable summary that separates service delivery readiness from strategy-policy approval.",
        },
        "scope_alignment": {
            "endpoint": "/scope-alignment.json",
            "source_documents": [
                "Poker ML Project.docx",
                "Poker_Agent_Development_EN_detailed.pdf",
            ],
            "description": "Traceability report mapping the client scope to implemented baselines, acceptance gates, and deployment artifacts.",
        },
        "approval_boundary": {
            "software_delivery": "PASS means the service, reports, packaging, and reproducibility checks are valid.",
            "strategy_model": "Production approval for autonomous strategic quality remains separate from software delivery.",
        },
    }
