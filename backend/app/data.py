import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .config import settings
from .db import Dataset, new_id
from .features import date_columns


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def profile_frame(df: pd.DataFrame) -> dict:
    sample = df.sample(min(len(df), 10_000), random_state=42)
    columns = []
    dates = set(date_columns(sample))
    total_missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    outlier_rows = pd.Series(False, index=df.index)
    for name in df.columns:
        series = df[name]
        non_null = series.dropna()
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        kind = "numeric" if numeric else "datetime" if name in dates else "categorical"
        info = {
            "name": name, "dtype": str(series.dtype), "kind": kind,
            "missing": int(series.isna().sum()), "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique()),
            "examples": [str(v)[:100] for v in non_null.head(3).tolist()],
        }
        if numeric and len(non_null):
            q1, q3 = non_null.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            outliers = series.notna() & ((series < lower) | (series > upper))
            outlier_rows |= outliers
            info.update({"outliers": int(outliers.sum()), "outlier_pct": round(float(outliers.mean() * 100), 2),
                         "outlier_bounds": {"lower": lower, "upper": upper}, "outlier_method": "1.5 × IQR"})
            info.update({"mean": float(non_null.mean()), "median": float(non_null.median()),
                         "std": float(non_null.std()), "min": float(non_null.min()), "max": float(non_null.max())})
            values = sample[name].dropna().to_numpy(dtype=float)
            counts, edges = np.histogram(values, bins=min(18, max(1, int(series.nunique()))))
            info["distribution"] = [
                {"label": f"{edges[i]:.3g}", "value": int(count), "upper": float(edges[i + 1])}
                for i, count in enumerate(counts)
            ]
        else:
            counts = sample[name].fillna("(missing)").astype(str).value_counts().head(10)
            info["distribution"] = [{"label": str(k)[:50], "value": int(v)} for k, v in counts.items()]
        columns.append(info)
    numeric_df = sample.select_dtypes(include="number").iloc[:, :16]
    corr = numeric_df.corr().round(3)
    correlations = [{"x": x, "y": y, "value": corr.loc[y, x]} for y in corr.index for x in corr.columns]
    numeric_names = list(numeric_df.columns)
    scatter = []
    if len(numeric_names) >= 2:
        scatter = sample[numeric_names[:2]].dropna().head(500).rename(
            columns={numeric_names[0]: "x", numeric_names[1]: "y"}
        ).to_dict("records")
    missing_pct = total_missing / max(df.size, 1) * 100
    duplicate_pct = duplicates / max(len(df), 1) * 100
    quality = round(max(0, 100 - missing_pct - duplicate_pct), 1)
    insights = [
        {"title": "Dataset overview", "text": f"{len(df):,} records across {len(df.columns)} features, including {len(numeric_names)} numeric features in the correlation view.", "kind": "info"},
        {"title": "Completeness", "text": f"{100 - missing_pct:.1f}% of cells contain a value. {sum(c['missing'] > 0 for c in columns)} columns have missing values.", "kind": "warning" if missing_pct > 5 else "success"},
        {"title": "Duplicate records", "text": f"{duplicates:,} exact duplicate rows found. Training removes exact duplicates before splitting.", "kind": "warning" if duplicates else "success"},
    ]
    pairs = [(x, y, float(corr.loc[x, y])) for i, x in enumerate(corr.index) for y in corr.index[i + 1:] if pd.notna(corr.loc[x, y])]
    insights.append({"title": "Potential outliers", "text": f"{int(outlier_rows.sum()):,} rows contain at least one numeric value outside 1.5 × IQR bounds. They are flagged for review and retained for training.", "kind": "warning" if outlier_rows.any() else "success"})
    if pairs:
        x, y, value = max(pairs, key=lambda p: abs(p[2]))
        insights.append({"title": "Strongest numeric relationship", "text": f"{x} and {y} have a Pearson correlation of {value:.2f}. This describes association, not causation.", "kind": "info"})
    return safe_json({
        "rows": len(df), "columns": columns, "missing_cells": total_missing, "missing_pct": round(missing_pct, 2),
        "duplicates": duplicates, "quality_score": quality, "correlations": correlations,
        "correlation_columns": numeric_names, "scatter": scatter, "scatter_columns": numeric_names[:2],
        "sampled_rows": len(sample), "insights": insights,
        "outlier_rows": int(outlier_rows.sum()), "outlier_method": "1.5 × IQR per numeric column",
        "profile_version": 2,
    })


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        raise HTTPException(422, "The file has no data rows.")
    if len(df) > settings.max_dataset_rows:
        raise HTTPException(422, f"Maximum dataset size is {settings.max_dataset_rows:,} rows.")
    if len(df.columns) > 200:
        raise HTTPException(422, "Maximum dataset width is 200 columns.")
    names = [str(c).strip()[:255] or f"column_{i}" for i, c in enumerate(df.columns)]
    if len(set(names)) != len(names):
        raise HTTPException(422, "Column names must be unique after trimming.")
    df.columns = names
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].map(lambda v: str(v) if pd.notna(v) else None)
    return df.replace([np.inf, -np.inf], np.nan)


def register_frame(db, owner_id: str, df: pd.DataFrame, filename: str, raw: bytes | None = None):
    df = normalize_frame(df)
    identifier = new_id()
    path = settings.data_dir / "datasets" / f"{identifier}.parquet"
    df.to_parquet(path, index=False)
    digest = hashlib.sha256(raw if raw is not None else path.read_bytes()).hexdigest()
    dataset = Dataset(
        id=identifier, owner_id=owner_id,
        name=Path(filename).stem.replace("_", " ").title(), filename=Path(filename).name,
        path=str(path), row_count=len(df), column_count=len(df.columns),
        size_bytes=len(raw) if raw is not None else path.stat().st_size,
        fingerprint=digest, profile=profile_frame(df),
    )
    db.add(dataset)
    db.commit()
    return dataset


def parse_upload(raw: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(io.BytesIO(raw), nrows=settings.max_dataset_rows + 1)
        if suffix in (".xlsx", ".xlsm"):
            return pd.read_excel(io.BytesIO(raw), nrows=settings.max_dataset_rows + 1)
        raise HTTPException(415, "Upload a CSV or XLSX file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read the file: {str(exc)[:180]}") from exc


def demo_frame(kind: str = "churn") -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(42)
    n = 2200
    tenure = rng.integers(1, 73, n)
    charges = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Monthly", "Annual", "Two year"], n, p=[0.55, 0.3, 0.15])
    support = rng.poisson(1.2, n)
    usage = rng.gamma(3, 8, n).round(1)
    score = -0.4 - tenure / 35 + (charges - 60) / 55 + (contract == "Monthly") * 1.1 + support * 0.45
    churn = rng.binomial(1, 1 / (1 + np.exp(-score)))
    df = pd.DataFrame({"customer_id": [f"CUS-{i:05}" for i in range(n)], "tenure_months": tenure,
                       "monthly_charges": charges, "contract_type": contract,
                       "support_tickets": support, "usage_gb": usage,
                       "payment_method": rng.choice(["Card", "Bank transfer", "Electronic check"], n),
                       "churn": np.where(churn, "Yes", "No")})
    df.loc[rng.choice(n, 55, replace=False), "usage_gb"] = np.nan
    if kind == "sales":
        df = pd.DataFrame({"advertising_spend": rng.uniform(100, 10000, n).round(2),
                           "store_size": rng.integers(400, 4000, n),
                           "region": rng.choice(["North", "South", "East", "West"], n),
                           "employees": rng.integers(3, 30, n)})
        df["monthly_revenue"] = (df.advertising_spend * 3.2 + df.store_size * 12 + df.employees * 450 + rng.normal(0, 6000, n)).round(2)
        return df, "retail_sales.csv"
    return df, "customer_churn.csv"


def infer_task(series: pd.Series):
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "classification"
    unique = series.dropna().nunique()
    return "classification" if unique <= 15 and unique / max(len(series.dropna()), 1) < 0.1 else "regression"


def id_like(name: str, series: pd.Series) -> bool:
    return bool(re.search(r"(^id$|_id$|^id_|identifier|uuid)", name, re.I)) and series.nunique() / max(len(series), 1) > 0.8
