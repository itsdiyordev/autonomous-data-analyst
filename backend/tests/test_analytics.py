import numpy as np
import pandas as pd
import pytest

from app.anomalies import detect_anomalies
from app.data import normalize_frame, profile_frame
from app.insights import generate_insights
from app.leakage import detect_leakage
from app.model_assessment import calibration, eligibility, imbalance, stability
from app.planner import create_plan
from app.statistics import run_statistics
from app.time_series import analyze_temporal


@pytest.mark.parametrize("frame", [pd.DataFrame({"constant": [1] * 12, "empty": [np.nan] * 12}), pd.DataFrame({"value": [1, np.inf, -np.inf, 100000]}), pd.DataFrame({"extreme": [-1e308, 0, 1e308]}), pd.DataFrame({"mixed": ["1", "2", "oops", "4"]}), pd.DataFrame({"description": [f"Unique free-text content with more than sixty characters describing a different item {i}" for i in range(150)]})])
def test_profiler_edge_cases_are_json_safe_and_explicit(frame):
    import json
    profile = profile_frame(normalize_frame(frame.copy()))
    json.dumps(profile, allow_nan=False)
    assert profile["rows"] == len(frame) and profile["profile_version"] == 4
    if "empty" in frame:
        assert {"constant", "empty"} <= set(profile["constant_columns"])
    if "value" in frame:
        assert profile["invalid_values"] == 2 and profile["columns"][0]["infinite_values"] == 2
    if "mixed" in frame:
        assert profile["columns"][0]["mixed_numeric_text"]
    if "description" in frame:
        assert "description" in profile["free_text_columns"] and "description" in profile["high_cardinality_columns"]
    if "extreme" in frame:
        assert profile["columns"][0]["extreme_magnitude"] and profile["columns"][0]["mean"] == 0
        evidence = []
        run_statistics(frame, profile, ["extreme"], evidence)
        assert evidence[0].status == "skipped" and "measurement units" in evidence[0].limitations[0]


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame({"value": []})])
def test_empty_ingestion_is_rejected(frame):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as failure:
        normalize_frame(frame)
    assert failure.value.status_code == 422


def test_correlations_intervals_and_weak_significant_effects_are_distinct():
    rng = np.random.default_rng(42)
    x = rng.normal(size=5000)
    x = (x - x.mean()) / x.std()
    noise = rng.normal(size=len(x))
    noise -= noise.mean()
    noise -= np.dot(noise, x) / np.dot(x, x) * x
    noise /= noise.std()
    frame = pd.DataFrame({"measurement": x, "outcome": 0.05 * x + np.sqrt(1 - 0.05 ** 2) * noise})
    evidence = []
    run_statistics(frame, profile_frame(frame), list(frame), evidence)
    pearson = next(item for item in evidence if item.method == "Pearson correlation")
    assert pearson.values["coefficient"] == pytest.approx(0.05)
    assert pearson.values["q_value"] < 0.05 and pearson.values["practical_strength"] == "negligible"
    assert pearson.values["confidence_interval"]["lower"] < 0.05 < pearson.values["confidence_interval"]["upper"]
    findings = generate_insights(evidence)
    assert any("practically weak" in finding.finding for finding in findings)
    assert all(finding.confidence == "LOW" for finding in findings)


def test_mixed_group_comparison_uses_welch_and_records_cohens_d():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"region": ["A"] * 120 + ["B"] * 120, "sales": np.r_[rng.normal(0, 1, 120), rng.normal(2, 1.3, 120)]})
    evidence = []
    run_statistics(frame, profile_frame(frame), list(frame), evidence)
    item = evidence[0]
    assert item.method == "Welch's t-test" and abs(item.values["cohens_d"]) > 1
    assert item.values["q_value"] < 0.01 and len(item.values["groups"]) == 2
    assert any("causality" in limitation or "causality" in assumption for limitation in item.limitations for assumption in item.assumptions)


def test_skewed_groups_choose_rank_test_and_sparse_categories_use_fisher():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"group": ["A"] * 20 + ["B"] * 20, "value": np.r_[rng.exponential(1, 20), rng.exponential(5, 20)]})
    evidence = []
    run_statistics(frame, profile_frame(frame), list(frame), evidence)
    assert evidence[0].method == "Mann–Whitney U" and "rank_biserial" in evidence[0].values
    frame = pd.DataFrame({"category": ["A"] * 48 + ["B"] * 2, "outcome": ["No"] * 46 + ["Yes"] * 4})
    evidence = []
    run_statistics(frame, profile_frame(frame), list(frame), evidence)
    assert evidence[0].method == "Fisher exact association"
    assert 0 <= evidence[0].values["effect_size"] <= 1


def test_multigroup_statistics_report_omnibus_effect_and_fdr():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"segment": ["A"] * 50 + ["B"] * 50 + ["C"] * 50, "value": np.r_[rng.normal(0, 1, 50), rng.normal(2, 1, 50), rng.normal(4, 1, 50)]})
    evidence = []
    run_statistics(frame, profile_frame(frame), list(frame), evidence)
    assert evidence[0].method in {"One-way ANOVA", "Kruskal–Wallis"}
    assert evidence[0].values["effect_measure"] in {"eta_squared", "epsilon_squared"}
    assert evidence[0].values["q_value"] < 0.01
    assert any("omnibus" in note for note in evidence[0].limitations)


def test_temporal_irregularity_missing_dates_duplicates_and_future_dates():
    dates = pd.date_range("2090-01-01", periods=35).delete([10, 20])
    frame = pd.DataFrame({"event_date": list(dates) + [dates[0]], "sales": np.arange(len(dates) + 1) + 10})
    profile = profile_frame(frame)
    temporal = profile["columns"][0]["temporal"]
    assert not temporal["regular"] and temporal["future_timestamps"] == len(frame)
    evidence = []
    item = analyze_temporal(frame, "event_date", "sales", evidence, True)
    assert item.values["missing_periods"] == 2 and item.values["duplicate_dates"] == 1
    assert item.values["forecast"]["status"] == "skipped"
    assert any(point["value"] is None for point in item.values["points"])


def test_semantic_temporal_and_target_copy_leakage_are_flagged_without_removal():
    frame = pd.DataFrame({"target": np.arange(100, dtype=float), "target_copy": np.arange(100, dtype=float), "prediction_time": pd.date_range("2025-01-01", periods=100), "cancellation_date": pd.date_range("2025-02-01", periods=100)})
    original = frame.copy()
    review = detect_leakage(frame, "target", profile_frame(frame))
    flags = {item["feature"]: item for item in review["flags"]}
    assert {"target_copy", "cancellation_date"} <= set(flags)
    assert flags["target_copy"]["evidence"]["correlation"] == pytest.approx(1)
    assert flags["cancellation_date"]["evidence"]["after_prediction_fraction"] == 1
    pd.testing.assert_frame_equal(frame, original)
    binary = pd.DataFrame({"outcome": ["No"] * 50 + ["Yes"] * 50, "suspicious_measurement": list(range(100))})
    binary_review = detect_leakage(binary, "outcome", profile_frame(binary))
    assert any(flag["feature"] == "suspicious_measurement" for flag in binary_review["flags"])


def test_anomalies_are_explained_and_original_rows_retained():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"amount": np.r_[rng.normal(0, 1, 99), 100], "visits": np.r_[rng.normal(0, 1, 99), -100]})
    evidence = []
    item = detect_anomalies(frame, list(frame), evidence)
    assert item.values["flagged_rows"] > 0
    assert any(example["row_index"] == 99 for example in item.values["examples"])
    assert all(example["reasons"] for example in item.values["examples"])
    assert len(frame) == 100


def test_severe_imbalance_rare_classes_constant_target_and_model_suitability():
    frame = pd.DataFrame({"value": range(1000), "churn": ["No"] * 980 + ["Yes"] * 20})
    balance = imbalance(frame, "churn")
    assert balance["severe"] and balance["minority_share"] == 0.02
    assert "PR-AUC" in balance["policy"]
    rare = frame.copy()
    rare["churn"] = ["No"] * 999 + ["Yes"]
    assert not create_plan(profile_frame(rare), "Predict churn.").ml_decision.use_ml
    constant = pd.DataFrame({"input": range(100), "target": [2] * 100})
    assert not create_plan(profile_frame(constant), "Predict target.", "target", "regression").ml_decision.use_ml
    assert not create_plan(profile_frame(frame), "Predict churn.", "churn", "regression").ml_decision.use_ml
    incomplete = pd.DataFrame({"input": range(100), "target": list(range(30)) + [np.nan] * 70})
    assert not create_plan(profile_frame(incomplete), "Predict target.", "target", "regression").ml_decision.use_ml
    assert eligibility("regression", "baseline", 10, 5000)[0]
    assert not eligibility("classification", "neural_network", 80, 30, [40, 40])[0]
    assert not eligibility("regression", "ensemble", 100, 1500)[0]
    assert stability({"mean": 0.91, "std": 0.02})["status"] == "stable"
    assert stability({"mean": 0.91, "std": 0.3})["status"] == "unstable"


def test_probability_calibration_is_evaluated_not_assumed():
    class Model:
        def __init__(self, probability):
            self.probability = probability

        def predict_proba(self, frame):
            return np.c_[1 - self.probability, self.probability]

    frame = pd.DataFrame({"value": range(100)})
    y = np.array([0, 1] * 50)
    perfect = calibration(Model(y), frame, y, ["No", "Yes"])
    poor = calibration(Model(np.where(y == 1, 0.51, 0.49)), frame, y, ["No", "Yes"])
    assert perfect["brier_score"] == 0 and perfect["status"] == "good"
    assert poor["status"] == "poor" and poor["brier_score"] > 0.2
    assert sum(bin["rows"] for bin in poor["reliability"]) == len(frame)


def test_large_dataset_workloads_are_bounded_without_discarding_data():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"a": rng.normal(size=100000), "b": rng.normal(size=100000), "segment": rng.choice(["A", "B", "C"], 100000)})
    profile = profile_frame(frame)
    assert profile["rows"] == 100000 and profile["sampled_rows"] == 10000
    evidence = []
    run_statistics(frame, profile, list(frame), evidence)
    assert all(item.sample_rows <= 5000 and item.population_rows == 100000 for item in evidence)
    assert detect_anomalies(frame, ["a", "b"], evidence).sample_rows == 10000
