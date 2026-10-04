"""Objective → selected analytics → optional ML → evidence → findings → actions."""
import logging
import time

import pandas as pd
from sqlalchemy import update
from threadpoolctl import threadpool_limits

from .anomalies import detect_anomalies
from .data import safe_json
from .db import AnalysisRun, Dataset, SessionLocal, now
from .evidence import AnalysisPlan, record_evidence
from .hypothesis import generate_hypotheses
from .insights import generate_insights
from .leakage import detect_leakage
from .llm import grounded_completion
from .ml import Cancelled, _train, publish_result, train_run, update_progress
from .model_assessment import imbalance
from .planner import create_plan
from .reports import analytical_report
from .reproducibility import metadata
from .statistics import run_statistics
from .time_series import analyze_temporal
from .errors import encode_failure
from .version import ENGINE_VERSION

logger = logging.getLogger(__name__)


def execute_run(run_id):
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        if not run or run.status != "running":
            return
        autonomous = run.config.get("analysis_mode") == "autonomous"
    if not autonomous:
        return train_run(run_id)
    try:
        with threadpool_limits(limits=1):
            _analyze(run_id)
    except Cancelled:
        logger.info("Cancelled analytical run %s", run_id)
    except Exception as exc:
        logger.exception("Analytical run failed")
        with SessionLocal() as db:
            db.execute(update(AnalysisRun).where(AnalysisRun.id == run_id, AnalysisRun.status == "running").values(status="failed", error=encode_failure(exc), stage="Analysis failed", finished_at=now()))
            db.commit()


def _analyze(run_id):
    started = time.monotonic()
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        dataset = db.get(Dataset, run.dataset_id)
        profile, config, objective = dataset.profile, dict(run.config), run.objective
        frame = pd.read_parquet(dataset.path)
    plan = AnalysisPlan.model_validate(config["analysis_plan"]) if config.get("analysis_plan") else create_plan(profile, objective, run.target or None, run.task, config)
    evidence, hypotheses, model, package = [], [], None, None
    leakage = {"flags": []}
    selected = next((step.columns for step in plan.steps if step.kind == "associations"), [])
    outcome_task = config.get("requested_task", "auto")
    if outcome_task == "auto":
        outcome_task = plan.ml_decision.task if plan.ml_decision.use_ml else next((candidate.task for candidate in plan.target_candidates if candidate.name == plan.target), None)
    statistical_profile = {**profile, "columns": [{**column, "kind": "categorical"} if column["name"] == plan.target and outcome_task == "classification" else column for column in profile["columns"]]}
    statistics_done = False
    for index, step in enumerate(plan.steps):
        if step.kind in {"insights", "recommendations", "report", "explainability"}:
            continue
        step.status = "running"
        _save_plan(run_id, plan)
        update_progress(run_id, min(48, 5 + index * 3), step.title, step.reason, step_id=step.id, decision=step.kind)
        before = len(evidence)
        if step.kind == "profile":
            record_evidence(evidence, "dataset", "Dataset understanding", "Schema and immutable dataset profile", values={"rows": len(frame), "columns": len(frame.columns), "type_ratios": profile.get("type_ratios", {}), "target_candidates": [candidate.model_dump() for candidate in plan.target_candidates]}, sample_rows=len(frame), population_rows=len(frame))
        elif step.kind == "quality":
            record_evidence(evidence, "data_quality", "Data quality assessment", "Direct counts and per-column IQR", values={"rows": len(frame), "columns": len(frame.columns), **{key: profile.get(key, 0) for key in ("missing_cells", "missing_pct", "duplicates", "quality_score", "invalid_values", "outlier_rows")}, **{key: profile.get(key, []) for key in ("constant_columns", "near_constant_columns", "possible_ids", "high_cardinality_columns", "free_text_columns")}, "policy": "Original data retained; structural ML exclusions are recorded separately."}, sample_rows=len(frame), population_rows=len(frame))
        elif step.kind == "temporal":
            temporal = analyze_temporal(frame, *step.columns[:2], evidence, forecast_requested=plan.intent == "forecast" and plan.target == step.columns[1])
            if plan.intent == "forecast" and plan.target != step.columns[1] and temporal.status == "completed":
                temporal.values["forecast"]["reason"] = "No reliable numeric outcome was identified for the requested forecast; specify the measure. The displayed temporal overview is exploratory."
        elif step.kind == "associations":
            run_statistics(frame, statistical_profile, selected, evidence, plan.bounds["association_pairs"])
            statistics_done = True
        elif step.kind == "group_comparison":
            groups = [item.id for item in evidence if item.kind == "group_difference"]
            record_evidence(evidence, "group_summary", "Group comparison provenance", "Group summaries included with their corresponding tests", columns=step.columns, values={"evidence_ids": groups, "reason": "Selected groups have adequate complete observations." if groups else "No selected group comparison met the complete-case/group-size requirements."}, status="completed" if groups else "skipped", sample_rows=min(5000, len(frame)), population_rows=len(frame))
        elif step.kind == "statistics":
            if not statistics_done:
                run_statistics(frame, statistical_profile, selected, evidence, plan.bounds["association_pairs"])
            record_evidence(evidence, "testing_policy", "Statistical inference policy", "Benjamini–Hochberg adjusted exploratory tests", values={"tests": sum("q_value" in item.values for item in evidence), "alpha": 0.05, "effect_sizes_required": True}, population_rows=len(frame), limitations=["Statistical significance is distinct from practical importance and does not imply causality."])
        elif step.kind == "hypotheses":
            hypotheses = generate_hypotheses(evidence)
        elif step.kind == "anomalies":
            detect_anomalies(frame, step.columns, evidence)
        elif step.kind == "leakage":
            leakage = detect_leakage(frame, plan.target, profile)
            record_evidence(evidence, "leakage", "Potential target leakage", "Semantic, temporal and extreme-association review", columns=[plan.target], values=leakage, sample_rows=min(5000, len(frame)), population_rows=len(frame), limitations=["These are review flags, not proof of leakage; feature availability needs domain confirmation."])
        elif step.kind == "imbalance":
            record_evidence(evidence, "imbalance", "Class imbalance", "Observed labeled class distribution", columns=[plan.target], values=imbalance(frame, plan.target), sample_rows=int(frame[plan.target].notna().sum()), population_rows=len(frame))
        elif step.kind == "ml":
            if plan.ml_decision.use_ml:
                model, package = _train(run_id, time.monotonic(), publish=False)
                primary = model["primary_metric"]
                baseline = (model.get("baseline_metrics") or {}).get(primary)
                beats = baseline is not None and (model["metrics"][primary] < baseline if model["task"] == "regression" else model["metrics"][primary] > baseline)
                record_evidence(evidence, "model", "Held-out model evaluation", "Development-only selection, independent test evaluation", values={"model_name": model["model_name"], "selection_metric": primary, "primary_metric": primary, "test_score": model["metrics"][primary], "metrics": model["metrics"], "cross_validation": model["cross_validation"], "stability": model["model_stability"], "calibration": model["calibration"], "leakage_flags": len(leakage["flags"]), "beats_baseline": beats}, sample_rows=model["split"]["test"], population_rows=len(frame), limitations=model["model_limitations"])
            else:
                step.status = "skipped"
                record_evidence(evidence, "ml_decision", "Machine-learning decision", "Objective and dataset suitability rules", values=plan.ml_decision.model_dump(), population_rows=len(frame))
        if step.status != "skipped":
            step.status = "skipped" if len(evidence) > before and all(item.status == "skipped" for item in evidence[before:]) else "completed"
        _save_plan(run_id, plan)
        update_progress(run_id, 49 if model else min(48, 8 + index * 3), step.title, f"{step.title}: {step.status}. " + (f"Produced {len(evidence) - before} evidence records." if step.kind != "hypotheses" else f"Tested {len(hypotheses)} exploratory hypotheses."), step_id=step.id, evidence_ids=[item.id for item in evidence[before:]])
    insights = generate_insights(evidence)
    from .recommendations import generate_recommendations
    recommendations = generate_recommendations(insights)
    for step in plan.steps:
        if step.kind in {"explainability", "insights", "recommendations", "report"}:
            step.status = "completed" if step.kind != "explainability" or model else "skipped"
    limitations = list(dict.fromkeys(["All findings are observational/exploratory. Association and model-based scenarios do not establish causal effects.", "Confidence levels are explicit rule assessments, not probabilities; significance does not imply practical importance.", "Statistical tests use at most 5,000 deterministic sampled rows and 24 selected column pairs; missingness and dependence may limit inference.", "Hypotheses use the same exploratory sample; independent replication is needed.", *plan.ml_decision.limitations, *(model.get("model_limitations", []) if model else [])]))
    analysis = safe_json({"plan": plan.model_dump(), "ml_decision": plan.ml_decision.model_dump(), "evidence": [item.model_dump() for item in evidence], "hypotheses": hypotheses, "insights": [item.model_dump() for item in insights], "recommendations": [item.model_dump() for item in recommendations], "limitations": limitations, "reproducibility": metadata(run_id, dataset, plan.model_dump(), config), "leakage": leakage,
                          "association_matrix": [{"x": item.columns[0], "y": item.columns[1], "method": item.method, "value": item.values.get("effect_size"), "evidence_id": item.id} for item in evidence if item.kind in {"association", "group_difference"}]})
    result = model or {"model_name": None, "task": "descriptive", "target": plan.target, "metrics": {}, "experiments": [], "feature_importance": [], "input_schema": [], "pipeline": [], "classes": [], "seed": 42, "dataset_fingerprint": dataset.fingerprint, "version": "3.0"}
    result["analysis"] = analysis
    result["engine_version"] = ENGINE_VERSION
    result["report"] = analytical_report(dataset.name, objective, analysis, model)
    result["report_source"] = "computed"
    result["llm_explanation"] = grounded_completion(objective, {"narrative_blocks": [{"id": item.id, "text": item.finding} for item in insights]})
    result["duration_seconds"] = round(time.monotonic() - started, 2)
    _save_plan(run_id, plan)
    update_progress(run_id, 97, "Findings and recommendations", f"Generated {len(insights)} evidence-backed findings and {len(recommendations)} recommendations.", evidence_ids=[item.id for item in evidence])
    publish_result(run_id, result, package, dataset.name)


def _save_plan(run_id, plan):
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        if not run or run.status != "running":
            raise Cancelled()
        changed = db.execute(update(AnalysisRun).where(AnalysisRun.id == run_id, AnalysisRun.status == "running").values(config={**run.config, "analysis_plan": plan.model_dump()}))
        if changed.rowcount != 1:
            raise Cancelled()
        db.commit()
