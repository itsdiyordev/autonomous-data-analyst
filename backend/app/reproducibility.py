import hashlib
import importlib.metadata
import json
import platform


def metadata(run_id, dataset, plan, config):
    versions = {}
    for library in ("numpy", "pandas", "scipy", "scikit-learn", "shap", "joblib"):
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            versions[library] = "unavailable"
    contract = {"dataset_hash": dataset.fingerprint, "plan": plan, "configuration": {key: value for key, value in config.items() if key not in {"parent_run_id", "experiment_id"}}, "seed": 42, "library_versions": versions, "engine_version": "3.0"}
    return {"analysis_id": run_id, "dataset_id": dataset.id, "dataset_hash": dataset.fingerprint, "dataset_version": 1,
            "analysis_plan": plan, "configuration": config, "random_seed": 42, "python_version": platform.python_version(),
            "library_versions": versions, "feature_schema": dataset.profile["columns"], "dataset_timestamp": dataset.created_at.isoformat(),
            "configuration_hash": hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest(),
            "engine_version": "3.0", "equivalence": "Same immutable dataset/configuration/versions reproduce analytical values; timestamps and execution times may differ."}
