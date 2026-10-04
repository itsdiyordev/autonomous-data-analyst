import pandas as pd

from app.data import profile_frame
from app.planner import create_plan


def sales_profile():
    return profile_frame(pd.DataFrame({"order_date": pd.date_range("2025-01-01", periods=100), "sales": range(100, 200), "price": range(10, 110), "region": ["A", "B"] * 50}))


def test_diagnostic_sales_objective_selects_analysis_without_forcing_ml():
    plan = create_plan(sales_profile(), "Why did sales decrease?")
    assert {"temporal", "group_comparison", "anomalies", "statistics", "hypotheses"} <= {step.kind for step in plan.steps}
    assert not plan.ml_decision.use_ml
    assert all(step.reason and step.expected_outputs for step in plan.steps)
    assert plan.target == "sales"


def test_prediction_selects_classification_imbalance_cv_explanation():
    profile = profile_frame(pd.DataFrame({"monthly_charges": range(100), "churn": ["No"] * 90 + ["Yes"] * 10}))
    plan = create_plan(profile, "Predict customer churn.")
    assert plan.ml_decision.use_ml and plan.task == "classification"
    assert {"ml", "imbalance", "explainability"} <= {step.kind for step in plan.steps}
    assert "CV" in plan.expected_outputs


def test_segments_select_clustering_and_ambiguous_target_stays_descriptive():
    profile = sales_profile()
    assert create_plan(profile, "Find customer segments.").task == "clustering"
    plan = create_plan(profile, "Predict a useful business outcome.")
    assert not plan.ml_decision.use_ml and plan.target is None
    assert all("probability" in candidate.score_kind for candidate in plan.target_candidates)


def test_small_dataset_retains_exploration_and_does_not_train():
    profile = profile_frame(pd.DataFrame({"value": [1], "target": [2]}))
    plan = create_plan(profile, "Predict target outcomes.", target="target")
    assert not plan.ml_decision.use_ml
    assert "profile" in {step.kind for step in plan.steps}


def test_focused_quality_objective_does_not_run_unrelated_analyses():
    plan = create_plan(sales_profile(), "Check missing values and duplicate records.")
    assert plan.intent == "quality" and not plan.ml_decision.use_ml
    assert not {"temporal", "associations", "statistics", "anomalies"} & {step.kind for step in plan.steps}
    assert {"profile", "quality", "recommendations"} <= {step.kind for step in plan.steps}


def test_forecast_intent_does_not_use_random_tabular_ml_and_clusters_ignore_targets():
    plan = create_plan(sales_profile(), "Predict sales next month.")
    assert plan.intent == "forecast" and not plan.ml_decision.use_ml
    cluster = create_plan(sales_profile(), "Find similar sales segments.", "sales")
    assert cluster.task == "clustering" and cluster.ml_decision.target is None


def test_duplicate_minority_rows_do_not_create_false_training_eligibility():
    frame = pd.DataFrame({"value": list(range(50)) + [100] * 8, "churn": ["No"] * 50 + ["Yes"] * 8})
    plan = create_plan(profile_frame(frame), "Predict customer churn.")
    assert not plan.ml_decision.use_ml
    assert "insufficient supported class examples" in plan.ml_decision.reason


def test_forecast_without_dates_explains_why_forecasting_cannot_run():
    profile = profile_frame(pd.DataFrame({"sales": range(100), "region": ["A", "B"] * 50}))
    plan = create_plan(profile, "Forecast future sales.")
    assert "forecasting was not attempted" in plan.ml_decision.reason
    assert "temporal" not in {step.kind for step in plan.steps}
