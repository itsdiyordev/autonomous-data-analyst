"""Shared, JSON-serializable analytical contracts. Numerical truth comes from Python."""
from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["HIGH", "MEDIUM", "LOW"]


class TargetCandidate(BaseModel):
    name: str
    task: str
    score: float = Field(ge=0, le=1)
    confidence: Confidence
    reasons: list[str]
    score_kind: str = "heuristic suitability score, not a calibrated probability"


class AnalysisStep(BaseModel):
    id: str
    kind: Literal["profile", "quality", "temporal", "associations", "group_comparison", "statistics", "hypotheses", "anomalies", "leakage", "imbalance", "ml", "explainability", "insights", "recommendations", "report"]
    title: str
    reason: str
    columns: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    status: Literal["planned", "running", "completed", "skipped", "failed"] = "planned"


class MLDecision(BaseModel):
    use_ml: bool
    task: str = "descriptive"
    target: str | None = None
    reason: str
    limitations: list[str] = Field(default_factory=list)


class AnalysisPlan(BaseModel):
    version: str = "1.0"
    objective: str
    intent: str
    task: str
    target: str | None
    target_candidates: list[TargetCandidate]
    steps: list[AnalysisStep]
    ml_decision: MLDecision
    reasons: list[str]
    expected_outputs: list[str]
    seed: int = 42
    bounds: dict = Field(default_factory=lambda: {"statistical_rows": 5000, "association_pairs": 24, "anomaly_rows": 10000})


class Evidence(BaseModel):
    id: str
    kind: str
    title: str
    method: str
    columns: list[str] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
    sample_rows: int = 0
    population_rows: int = 0
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    status: str = "completed"
    provenance: Literal["computed_python"] = "computed_python"


class Insight(BaseModel):
    id: str
    type: str
    severity: Literal["high", "medium", "low"]
    confidence: Confidence
    finding: str
    why_it_matters: str
    evidence_ids: list[str] = Field(min_length=1)
    confidence_reasons: list[str]
    limitations: list[str]


class Recommendation(BaseModel):
    id: str
    insight_id: str
    action: str
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)
    limitation: str


def record_evidence(evidence: list[Evidence], kind: str, title: str, method: str, **kwargs) -> Evidence:
    item = Evidence(id=f"e{len(evidence) + 1:03d}", kind=kind, title=title, method=method, **kwargs)
    evidence.append(item)
    return item
