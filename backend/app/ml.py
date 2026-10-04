"""The autonomous profiler → detector → training → explanation → report workflow."""
import logging
import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, calinski_harabasz_score, confusion_matrix, davies_bouldin_score, f1_score, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sqlalchemy import update
from threadpoolctl import threadpool_limits

from .artifacts import export_solution
from .data import id_like, profile_frame, safe_json
from .db import AnalysisRun, Dataset, SessionLocal, now
from .explain import error_analysis, shap_explanation
from .features import FeatureEngineer, date_columns
from .llm import grounded_completion
from .leakage import detect_leakage
from .model_assessment import calibration, eligibility, imbalance, inference_timing, stability, threshold_analysis

logger = logging.getLogger(__name__)

PIPELINE_STAGES = [
    ("Data profiler", "Types, missing values, duplicates, outliers, correlations, and distributions."),
    ("Problem detector", "Choose regression, classification, or clustering from the goal and target."),
    ("Preprocessing engine", "Fit missing-value handling, encoding, scaling, and date feature engineering."),
    ("Model selector", "Choose task-appropriate candidates from supported model families."),
    ("Training engine", "Fit candidate pipelines on development data within the search budget."),
    ("Validation / Cross-validation", "Validate candidates using leakage-safe folds and a separate validation split."),
    ("Model comparison", "Rank candidates using their recorded development scores."),
    ("Best model", "Refit the selected complete pipeline on training plus validation rows."),
    ("Prediction", "Generate predictions and diagnostics on the untouched test set."),
    ("Explainability", "Compute permutation importance, SHAP, and error or assignment analysis."),
    ("Final analysis report", "Write a grounded explanation and export the complete reusable solution."),
]


class Cancelled(Exception):
    pass


def update_progress(run_id, progress, stage, detail, **trace):
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        if not run or run.status == "cancelled":
            raise Cancelled()
        if run.config.get("analysis_mode") == "autonomous" and stage in {item[0] for item in PIPELINE_STAGES}:
            progress = 50 + int(progress * 0.35)
        run.progress = max(run.progress, progress)
        run.stage = stage
        run.events = [*run.events, {"at": now().isoformat(), "stage": stage, "detail": detail, **trace}]
        run.updated_at = now()
        db.commit()


def preprocessing(numeric, categorical):
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=24, sparse_output=False, dtype=np.float32)),
        ]), categorical))
    return ColumnTransformer(transforms, remainder="drop")


def silhouette_validation(pipeline, X, y=None):
    labels = pipeline.predict(X)
    transformed = pipeline[:-1].transform(X)
    if not 1 < len(np.unique(labels)) < len(labels):
        raise ValueError("At least two non-empty clusters are needed for silhouette evaluation.")
    return float(silhouette_score(transformed, labels, sample_size=min(600, len(X)), random_state=42))


def calculate_metrics(pipeline, X, y, task, class_count):
    prediction = pipeline.predict(X)
    if task == "clustering":
        transformed = pipeline[:-1].transform(X)
        return {"silhouette": silhouette_validation(pipeline, X),
                "davies_bouldin": float(davies_bouldin_score(transformed, prediction)),
                "calinski_harabasz": float(calinski_harabasz_score(transformed, prediction)),
                "clusters": int(len(np.unique(prediction)))}
    if task == "regression":
        return {"mae": float(mean_absolute_error(y, prediction)), "rmse": float(np.sqrt(mean_squared_error(y, prediction))), "r2": float(r2_score(y, prediction))}
    result = {"accuracy": float(accuracy_score(y, prediction)), "f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
              "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
              "precision": float(precision_score(y, prediction, average="weighted", zero_division=0)),
              "recall": float(recall_score(y, prediction, average="weighted", zero_division=0)),
              "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0))}
    if class_count == 2 and len(np.unique(y)) == 2:
        probability = pipeline.predict_proba(X)[:, 1]
        result.update(roc_auc=float(roc_auc_score(y, probability)), pr_auc=float(average_precision_score(y, probability)),
                      positive_precision=float(precision_score(y, prediction, zero_division=0)), positive_recall=float(recall_score(y, prediction, zero_division=0)))
    return result


def split_data(df, target, task, config):
    strategy, column = config.get("split_strategy", "random"), config.get("split_column")
    size = config.get("test_size", 0.2)
    if strategy == "chronological":
        dates = pd.to_datetime(df[column], errors="coerce", utc=True, format="mixed")
        if dates.isna().any():
            raise ValueError("Choose a complete date column without invalid values.")
        df = df.assign(__sort_time=dates).sort_values("__sort_time").drop(columns="__sort_time")
        cut = int(len(df) * (1 - size))
        development, test = df.iloc[:cut], df.iloc[cut:]
        cut = int(len(development) * 0.8)
        train, validation = development.iloc[:cut], development.iloc[cut:]
    elif strategy == "group":
        if df[column].isna().any() or df[column].nunique() < 5:
            raise ValueError("Group splitting needs at least five complete groups.")
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=size, random_state=42).split(df, groups=df[column]))
        development, test = df.iloc[a], df.iloc[b]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=43).split(development, groups=development[column]))
        train, validation = development.iloc[a], development.iloc[b]
    else:
        development, test = train_test_split(df, test_size=size, random_state=42, stratify=df[target] if task == "classification" else None)
        train, validation = train_test_split(development, test_size=0.2, random_state=43, stratify=development[target] if task == "classification" else None)
    if min(len(train), len(validation), len(test)) < 5:
        raise ValueError("The selected split creates too few rows. Use a larger dataset.")
    if task == "classification":
        classes = set(df[target].unique())
        if any(set(part[target].unique()) != classes for part in (train, validation, test)):
            raise ValueError("Every split must contain all target classes. Change the split or add examples.")
    return train, validation, test


def cv_plan(train, target, task, config):
    strategy = config.get("split_strategy", "random")
    folds = config.get("cv_folds", 3)
    y = train[target] if task != "clustering" else None
    if strategy == "chronological":
        splitter, name = TimeSeriesSplit(n_splits=folds), "TimeSeriesSplit"
        splits = list(splitter.split(train))
    elif strategy == "group":
        groups = train[config["split_column"]]
        folds = min(folds, groups.nunique())
        if folds < 2:
            return None, {"status": "unavailable", "reason": "Too few development groups for cross-validation."}
        splits, name = list(GroupKFold(n_splits=folds).split(train, y, groups)), "GroupKFold"
    elif task == "classification":
        folds = min(folds, int(y.value_counts().min()))
        if folds < 2:
            return None, {"status": "unavailable", "reason": "Too few examples per class for cross-validation."}
        splits, name = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(train, y)), "StratifiedKFold"
    else:
        splits, name = list(KFold(n_splits=folds, shuffle=True, random_state=42).split(train)), "KFold"
    if task == "classification":
        classes = set(y.unique())
        if any(set(y.iloc[a].unique()) != classes or set(y.iloc[b].unique()) != classes for a, b in splits):
            return None, {"status": "unavailable", "reason": "Some group/time folds lack target classes; selection uses the separate validation set."}
    return splits, {"status": "completed", "strategy": name, "folds": len(splits),
                    "fold_rows": [{"train": len(a), "validation": len(b)} for a, b in splits]}


def model_candidates(task):
    neural = {"hidden_layer_sizes": (48, 24), "max_iter": 120, "early_stopping": True,
              "n_iter_no_change": 10, "random_state": 42}
    if task == "classification":
        return [
            ("Baseline", "baseline", DummyClassifier(strategy="prior"), {}),
            ("Logistic regression", "linear", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42), {"class_weight": "balanced"}),
            ("Decision tree", "tree", DecisionTreeClassifier(max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42), {"max_depth": 8}),
            ("Random forest", "ensemble", RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=3, class_weight="balanced", n_jobs=1, random_state=42), {"n_estimators": 100}),
            ("Neural network", "neural_network", MLPClassifier(**neural), {"hidden_layers": [48, 24], "max_iter": 120}),
            ("Gradient boosting", "ensemble", HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=23, l2_regularization=1, class_weight="balanced", random_state=42), {"max_iter": 130, "class_weight": "balanced"}),
        ]
    if task == "regression":
        return [
            ("Baseline", "baseline", DummyRegressor(), {}),
            ("Ridge regression", "linear", Ridge(alpha=10), {"alpha": 10}),
            ("Decision tree", "tree", DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=42), {"max_depth": 8}),
            ("Random forest", "ensemble", RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=3, n_jobs=1, random_state=42), {"n_estimators": 100}),
            ("Neural network", "neural_network", TransformedTargetRegressor(regressor=MLPRegressor(**neural), transformer=StandardScaler()), {"hidden_layers": [48, 24], "target_scaling": True}),
            ("Gradient boosting", "ensemble", HistGradientBoostingRegressor(max_iter=130, max_leaf_nodes=23, l2_regularization=1, random_state=42), {"max_iter": 130}),
        ]
    return [(f"K-Means · {k} groups", "baseline" if k == 2 else "centroid", KMeans(n_clusters=k, n_init=10, random_state=42), {"clusters": k}) for k in (2, 3, 4, 5)] + [
        (f"Gaussian mixture · {k} groups", "probabilistic", GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, random_state=42), {"clusters": k}) for k in (2, 3, 4)
    ]


def train_run(run_id):
    started = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            _train(run_id, started)
    except Cancelled:
        logger.info("Cancelled run %s", run_id)
    except Exception as exc:
        logger.exception("Pipeline failed for %s", run_id)
        with SessionLocal() as db:
            db.execute(update(AnalysisRun).where(AnalysisRun.id == run_id, AnalysisRun.status != "cancelled").values(
                status="failed", error=str(exc)[:1000], stage="Pipeline failed", finished_at=now()))
            db.commit()


def _train(run_id, started, publish=True):
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        dataset = db.get(Dataset, run.dataset_id)
        target, task, config, objective = run.target or None, run.task, run.config, run.objective
        name, fingerprint, path = dataset.name, dataset.fingerprint, dataset.path
    update_progress(run_id, 5, "Data profiler", "Calculating types, missing values, duplicates, IQR outliers, correlations, and distributions.")
    raw = pd.read_parquet(path)
    profile = profile_frame(raw)
    update_progress(run_id, 7, "Data profiler", f"Loaded {len(raw):,} original rows and {len(raw.columns)} columns; {profile['missing_cells']:,} missing cells and {profile['duplicates']:,} exact duplicate rows.")
    df = raw.drop_duplicates()
    if target:
        df = df.dropna(subset=[target])
    if len(df) < 40:
        raise ValueError("At least 40 distinct usable rows are needed for this workflow.")
    update_progress(run_id, 12, "Problem detector", config.get("problem_detection", {}).get("reason", f"Preparing a {task} workflow."))
    encoder, classes, class_balance = None, [], None
    if task == "classification":
        df = df.copy()
        df[target] = df[target].astype(str)
        counts = df[target].value_counts()
        if not 2 <= len(counts) <= 20 or counts.min() < 8:
            raise ValueError("Classification needs 2–20 classes with at least 8 examples in each.")
        class_balance = imbalance(df, target)
        class_balance["original_distribution"] = imbalance(raw, target)["distribution"]
        class_balance["distribution_scope"] = "Usable labeled rows after documented exact-duplicate/target handling; original distribution is stored separately."
        encoder = LabelEncoder().fit(df[target])
        if len(counts) == 2:
            positive = config.get("positive_label") or str(counts.index[-1])
            if positive not in counts.index:
                raise ValueError("The requested positive class is not in the target labels.")
            encoder.classes_ = np.asarray([str(label) for label in encoder.classes_ if label != positive] + [positive])
        classes = encoder.classes_.tolist()
        df[target] = encoder.transform(df[target])
        if len(classes) == 2:
            update_progress(run_id, 14, "Problem detector", f"Binary positive outcome: {classes[1]}. Selected from the explicit label or the minority class among usable labeled rows.")
    elif task == "regression":
        df = df.copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=[target])
        if len(df) < 40 or df[target].nunique() < 2:
            raise ValueError("Regression needs at least 40 rows with varying numeric target values.")
    dates = date_columns(df.drop(columns=[target] if target else []))
    excluded, features = [], []
    for column in df.columns:
        if column == target:
            continue
        reason = None
        if config.get("split_strategy") == "group" and column == config.get("split_column"):
            reason = "group split identifier"
        elif df[column].nunique() <= 1:
            reason = "constant or empty column"
        elif id_like(column, df[column]):
            reason = "unique identifier"
        elif column not in dates and not pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() > 100:
            reason = "high-cardinality text (more than 100 categories)"
        if reason:
            excluded.append({"feature": column, "reason": reason})
        else:
            features.append(column)
    if not 1 <= len(features) <= 100:
        raise ValueError("The workflow needs between 1 and 100 usable features after excluding unsuitable columns.")
    if excluded:
        update_progress(run_id, 18, "Preprocessing engine", "Explicit model-only column exclusions: " + "; ".join(f"{item['feature']}: {item['reason']}" for item in excluded))
    dates = [column for column in dates if column in features]
    raw_numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    raw_categorical = [column for column in features if column not in raw_numeric]
    for column in raw_categorical:
        df[column] = df[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    engineered = FeatureEngineer(tuple(dates)).fit_transform(df[features].head(100))
    numeric = list(engineered.select_dtypes(include="number").columns)
    categorical = [column for column in engineered.columns if column not in numeric]
    feature_engineering = {"date_columns": dates, "generated_date_features": [column for column in engineered if column not in features],
                           "missing_indicators": True, "categorical_whitespace_trimmed": True,
                           "outlier_policy": "Flagged with IQR bounds and retained; no target-aware outlier removal."}
    feature_engineering["row_handling"] = {"original_rows": len(raw), "usable_rows": len(df), "removed_exact_duplicates": int(raw.duplicated().sum()), "excluded_missing_or_invalid_target_rows": len(raw.drop_duplicates()) - len(df), "policy": "Only exact duplicates and unusable supervised targets are excluded from ML; original data and all outliers are retained."}
    leakage = detect_leakage(raw, target, profile)
    update_progress(run_id, 20, "Preprocessing engine", f"ML row handling: {len(raw):,} original rows → {len(df):,} usable rows after exact duplicates and unusable targets; outliers retained.")
    update_progress(run_id, 20, "Preprocessing engine", f"Preparing {len(features)} raw inputs; {len(dates)} date columns produce calendar and cyclic features. Imputation and scaling fit inside each fold.")
    train, validation, test = split_data(df, target, task, config)
    search = train
    if len(train) > 20_000:
        indexes = np.sort(np.random.default_rng(42).choice(len(train), 20_000, replace=False))
        sample = train.iloc[indexes]
        if task != "classification" or (sample[target].nunique() == len(classes) and sample[target].value_counts().min() >= 3):
            search = sample
    X_train, X_val, X_test = search[features], validation[features], test[features]
    if len(search) < len(train):
        update_progress(run_id, 23, "Model selector", f"Candidate search uses {len(search):,} seeded development rows out of {len(train):,}; final refit still uses all development rows.")
    y_train = search[target] if target else None
    y_val = validation[target] if target else None
    y_test = test[target] if target else None
    binary = task == "classification" and len(classes) == 2
    primary = "silhouette" if task == "clustering" else "pr_auc" if binary else "f1_macro" if task == "classification" else "rmse"
    scoring = silhouette_validation if task == "clustering" else "average_precision" if binary else "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    folds, cv_metadata = cv_plan(search, target, task, config)
    candidates = model_candidates(task)
    width = len(numeric) * 2 + sum(min(24, int(search[column].nunique())) for column in raw_categorical if column not in dates)
    if width > 1200 or len(df) * width * 8 > 512 * 1024 * 1024:
        raise ValueError("The encoded feature matrix exceeds the supported 512 MiB / 1,200-input memory contract. Use fewer selected source columns or autonomous exploratory analysis.")
    update_progress(run_id, 25, "Model selector", f"Selected {len(candidates)} candidates: " + ", ".join(dict.fromkeys(candidate[1].replace("_", " ") for candidate in candidates)))
    experiments, fitted = [], {}
    for index, (model_name, family, estimator, parameters) in enumerate(candidates):
        allowed, suitability = eligibility(task, family, len(search), width, search[target].value_counts().tolist() if task == "classification" else None)
        if not allowed:
            experiments.append({"name": model_name, "family": family, "status": "skipped", "parameters": parameters, "reason": suitability, "suitability": suitability})
            update_progress(run_id, 28, "Model selector", f"Skipped {model_name}: {suitability}", decision="eligibility")
            continue
        if index > 1 and time.monotonic() - started >= config.get("budget_seconds", 180):
            experiments.append({"name": model_name, "family": family, "status": "skipped", "parameters": parameters, "reason": "Model search budget reached."})
            continue
        progress = 28 + int(index / len(candidates) * 34)
        update_progress(run_id, progress, "Training engine", f"Fitting {model_name} with a complete preprocessing pipeline.")
        began = time.monotonic()
        pipeline = Pipeline([("features", FeatureEngineer(tuple(dates))), ("preprocess", preprocessing(numeric, categorical)), ("model", estimator)])
        try:
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always")
                pipeline.fit(X_train, y_train)
            validation_metrics = calculate_metrics(pipeline, X_val, y_val, task, len(classes))
            update_progress(run_id, progress + 2, "Validation / Cross-validation", f"{model_name}: " + (f"{len(folds)} leakage-safe {cv_metadata['strategy']} folds." if folds else "separate hold-out validation; cross-validation is unavailable for this split."))
            cv = dict(cv_metadata)
            if folds:
                values = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring=scoring, n_jobs=1, error_score="raise")
                if task == "regression":
                    values = -values
                if not np.isfinite(values).all():
                    raise ValueError("Cross-validation produced non-finite scores.")
                cv.update(mean=float(values.mean()), std=float(values.std()), scores=values.tolist())
            score = cv["mean"] if folds else validation_metrics[primary]
            experiments.append({"name": model_name, "family": family, "status": "completed", "validation_metrics": validation_metrics,
                                 "cross_validation": cv, "selection_score": score, "parameters": parameters,
                                 "suitability": suitability, "stability": stability(cv), "inference_time": inference_timing(pipeline, X_val),
                                 "complexity": {"family": family, "estimated_encoded_inputs": width},
                                 "test_metrics": None, "test_note": "Only the selected model is evaluated on test rows; these rows never select a candidate.",
                                "duration_seconds": round(time.monotonic() - began, 2)})
            fitted[model_name] = pipeline
            experiments[-1]["warnings"] = list(dict.fromkeys(str(item.message)[:250] for item in fit_warnings))
        except Exception as exc:
            logger.warning("Candidate %s failed: %s", model_name, exc)
            experiments.append({"name": model_name, "family": family, "status": "failed", "error": str(exc)[:250], "parameters": parameters})
    completed = [experiment for experiment in experiments if experiment["status"] == "completed"]
    if not completed:
        raise ValueError("All model candidates failed. Review the data, target, and split strategy.")
    completed.sort(key=lambda experiment: experiment["selection_score"], reverse=task != "regression")
    winner = completed[0]
    update_progress(run_id, 67, "Model comparison", f"Compared {len(completed)} successful candidates by {'cross-validation mean' if folds else 'validation'} {primary.upper()}.")
    pipeline = fitted[winner["name"]]
    thresholds = threshold_analysis(pipeline, X_val, y_val) if binary else None
    development = pd.concat([train, validation])
    update_progress(run_id, 73, "Best model", f"Refitting {winner['name']} on all {len(development):,} development rows.")
    with warnings.catch_warnings(record=True) as refit_warnings:
        warnings.simplefilter("always")
        pipeline.fit(development[features], development[target] if target else None)
    update_progress(run_id, 79, "Prediction", "Generating predictions and metrics on the untouched test set.")
    test_metrics = calculate_metrics(pipeline, X_test, y_test, task, len(classes))
    winner["test_metrics"] = test_metrics
    baseline = next((item for item in completed if item["family"] == "baseline"), None)
    baseline_metrics = None
    if baseline:
        baseline_pipeline = fitted[baseline["name"]]
        if baseline["name"] != winner["name"]:
            baseline_pipeline.fit(development[features], development[target] if target else None)
        baseline_metrics = calculate_metrics(baseline_pipeline, X_test, y_test, task, len(classes))
    prediction = pipeline.predict(X_test)
    diagnostics, cluster_profiles = diagnostics_for(pipeline, X_test, y_test, prediction, task, classes)
    update_progress(run_id, 84, "Explainability", "Calculating permutation importance, SHAP contributions, and error or ambiguous-assignment examples.")
    sample = X_test.sample(min(300, len(X_test)), random_state=42)
    importance = permutation_importance(pipeline, sample, y_test.loc[sample.index] if target else None,
                                       scoring=scoring, n_repeats=3, random_state=42, n_jobs=1)
    feature_importance = sorted([{"feature": feature, "importance": float(mean), "std": float(std)}
                                for feature, mean, std in zip(features, importance.importances_mean, importance.importances_std)], key=lambda item: item["importance"], reverse=True)
    shap = shap_explanation(pipeline, development[features], X_test, task, features)
    errors = error_analysis(pipeline, X_test, y_test, prediction, task, classes)
    input_schema = [{"name": column, "type": "number" if column in raw_numeric else "string", "nullable": True,
                     "example": safe_json(df[column].dropna().iloc[0]) if not df[column].dropna().empty else None,
                     "categories": sorted(df[column].dropna().astype(str).unique().tolist())[:30] if column in raw_categorical and column not in dates else None}
                    for column in features]
    selection_note = (f"Models were ranked by {cv_metadata.get('folds', 0)}-fold {cv_metadata.get('strategy', 'hold-out')} mean {primary}. " if folds else "Models were ranked by separate validation scores. ") + "The selected pipeline was refitted on training + validation rows, then evaluated once on unseen test rows."
    stages = [{"name": stage, "detail": detail, "status": "completed"} for stage, detail in PIPELINE_STAGES]
    stages[1]["detail"] = config.get("problem_detection", {}).get("reason", task)
    stages[5]["detail"] = f"{cv_metadata.get('folds', 0)} folds · {cv_metadata.get('strategy', cv_metadata.get('reason'))}"
    result = safe_json({"model_name": winner["name"], "task": task, "target": target, "primary_metric": primary,
                        "metrics": test_metrics, "experiments": experiments, "feature_importance": feature_importance,
                        "shap": shap, "error_analysis": errors, "diagnostics": diagnostics, "cluster_profiles": cluster_profiles,
                        "excluded_features": excluded, "input_schema": input_schema, "classes": classes,
                        "positive_class": classes[1] if binary else None, "pipeline": stages,
                        "problem_detection": config.get("problem_detection", {}), "feature_engineering": feature_engineering,
                        "data_profile": {key: profile[key] for key in ("rows", "missing_cells", "duplicates", "outlier_rows", "quality_score")},
                         "cross_validation": winner["cross_validation"],
                         "selection_score": winner["selection_score"], "model_stability": stability(winner["cross_validation"]),
                         "class_imbalance": class_balance, "leakage": leakage, "baseline_metrics": baseline_metrics,
                         "calibration": calibration(pipeline, X_test, y_test, classes) if task == "classification" else None,
                         "threshold_analysis": thresholds, "inference_time": inference_timing(pipeline, X_test),
                         "search_sampling": {"search_rows": len(search), "training_rows": len(train), "seed": 42, "final_refit_full_development": True},
                         "model_limitations": ["Held-out scores do not guarantee future performance or causal effects.", "Fold stability and calibration statuses are transparent heuristics, not confidence probabilities.", "Potential leakage is flagged for review and not automatically removed."],
                         "training_warnings": list(dict.fromkeys([*winner.get("warnings", []), *(str(item.message)[:250] for item in refit_warnings)])),
                        "split": {"train": len(train), "validation": len(validation), "test": len(test), "strategy": config.get("split_strategy", "random")},
                        "dataset_fingerprint": fingerprint, "seed": 42, "version": "3.0", "python_version": platform.python_version(),
                        "sklearn_version": sklearn.__version__, "duration_seconds": round(time.monotonic() - started, 2), "selection_note": selection_note})
    top = ", ".join(item["feature"] for item in feature_importance[:3])
    report = (f"FINAL ANALYSIS REPORT — {name}\n\nGOAL\n{objective}\n\nDATA PROFILE\n{len(raw):,} original rows; {profile['missing_cells']:,} missing cells; {profile['duplicates']:,} duplicates; {profile['outlier_rows']:,} rows with potential IQR outliers. Exact duplicates were removed. Outliers were retained.\n\nPROBLEM\n{task.title()}: {stages[1]['detail']}\n\nPREPARATION\n{len(features)} raw inputs; {len(dates)} date columns with calendar/cyclic features; imputation, missing indicators, encoding, and scaling fit inside each fold.\n\nMODEL COMPARISON\nSelected {winner['name']} from {len(completed)} successful candidates. {selection_note}\n\nTEST RESULTS\n{primary.upper()}: {test_metrics[primary]:.4f}. {len(test):,} unseen rows.\n\nEXPLAINABILITY\nTop permutation-importance features: {top}. SHAP: {shap.get('method', shap.get('reason', 'unavailable'))}, using {shap.get('sample_rows', 0)} test examples. Predictive contributions do not establish causal effects.\n\nERROR ANALYSIS\n{errors.get('note', f'{len(errors["examples"])} representative difficult or incorrect test examples are included in the detailed evidence.')}\n\nDELIVERABLES\nComplete preprocessing + model pipeline, input schema, model comparison, SHAP evidence, error analysis, and standalone prediction script.")
    update_progress(run_id, 94, "Final analysis report", "Writing the natural-language report and packaging the complete ML solution.")
    enhanced = grounded_completion(f"Write a practical final analysis report for this objective: {objective}. Cover profiling, task detection, preprocessing, CV comparison, best model, test predictions, SHAP, and errors. Use only computed evidence.", result)
    result["report"], result["report_source"] = report, "computed"
    result["llm_explanation"] = enhanced
    result["duration_seconds"] = round(time.monotonic() - started, 2)
    package = {"pipeline": pipeline, "features": features, "numeric": raw_numeric, "categorical": raw_categorical,
               "encoder": encoder, "task": task, "schema": input_schema, "run_id": run_id, "version": "3.0",
               "evaluation_context": {"model_name": result["model_name"], "test_metrics": test_metrics, "test_rows": len(test), "note": "Held-out error and raw class probabilities do not guarantee an individual outcome or causal impact."}}
    if not publish:
        return result, package
    publish_result(run_id, result, package, name)


def publish_result(run_id, result, package, name):
    directory = export_solution(run_id, package, result, name)
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        events = [*run.events, {"at": now().isoformat(), "stage": "Completed", "detail": "The final analytical report and computed evidence are ready." if package is None else "The final report, trained model, and prediction service are ready."}]
        published = db.execute(update(AnalysisRun).where(AnalysisRun.id == run_id, AnalysisRun.status == "running").values(
            status="completed", progress=100, stage="Solution ready" if package else "Analysis ready", result=result, artifact_path=str(directory), finished_at=now(), updated_at=now(), events=events))
        db.commit()
        if published.rowcount != 1:
            raise Cancelled()


def diagnostics_for(pipeline, X, y, prediction, task, classes):
    diagnostics, profiles = {}, []
    if task == "classification":
        diagnostics.update(confusion_matrix=confusion_matrix(y, prediction, labels=list(range(len(classes)))).tolist(), classes=classes)
        if len(classes) == 2:
            probability = pipeline.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probability)
            indexes = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
            diagnostics["roc_curve"] = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in indexes]
            precision, recall, _ = precision_recall_curve(y, probability)
            indexes = np.linspace(0, len(recall) - 1, min(100, len(recall))).astype(int)
            diagnostics["pr_curve"] = [{"recall": float(recall[i]), "precision": float(precision[i])} for i in indexes]
    elif task == "regression":
        diagnostics["predicted_vs_actual"] = [{"actual": float(actual), "predicted": float(predicted), "residual": float(actual - predicted)} for actual, predicted in zip(y.iloc[:300], prediction[:300])]
    else:
        transformed = pipeline[:-1].transform(X)
        pca = PCA(n_components=min(2, transformed.shape[1]), random_state=42).fit(pipeline[:-1].transform(X.head(min(1000, len(X)))))
        projection = pca.transform(transformed[:400])
        diagnostics["cluster_projection"] = [{"x": float(row[0]), "y": float(row[1]) if len(row) > 1 else 0.0, "cluster": int(cluster)} for row, cluster in zip(projection, prediction[:400])]
        for cluster in sorted(np.unique(prediction)):
            frame = X.iloc[np.flatnonzero(prediction == cluster)]
            summaries = {}
            for column in X.columns:
                series = frame[column].dropna()
                if len(series):
                    summaries[column] = round(float(series.mean()), 3) if pd.api.types.is_numeric_dtype(series) else str(series.astype(str).mode().iloc[0])
            profiles.append({"cluster": int(cluster), "name": f"Group {int(cluster) + 1}", "rows": len(frame), "share": float(len(frame) / len(X)), "features": summaries})
    return safe_json(diagnostics), safe_json(profiles)
