import hashlib
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .config import settings
from .db import Dataset, new_id
from .features import date_columns

PROFILE_VERSION = 4


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
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    return value


def profile_frame(df: pd.DataFrame) -> dict:
    sample = df.sample(min(len(df), 10_000), random_state=42)
    columns = []
    dates = set(date_columns(sample))
    total_missing = int(df.isna().sum().sum())
    duplicate_mask = df.duplicated()
    duplicates = int(duplicate_mask.sum())
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
            "usable_non_missing_rows": int(series.loc[~duplicate_mask].notna().sum()),
            "examples": [str(v)[:100] for v in non_null.head(3).tolist()],
        }
        counts_all = non_null.astype(str).value_counts()
        dominant_share = float(counts_all.iloc[0] / max(len(non_null), 1)) if len(counts_all) else 0
        info.update(constant=series.nunique() <= 1, near_constant=dominant_share >= 0.95,
                    possible_id=id_like(name, series) or (not numeric and name not in dates and series.nunique() / max(len(df), 1) > 0.98 and non_null.astype(str).str.len().mean() < 40),
                    cardinality_ratio=float(series.nunique() / max(len(df), 1)),
                    high_cardinality=not numeric and name not in dates and series.nunique() > 100,
                    possible_free_text=not numeric and name not in dates and bool(len(non_null)) and float(non_null.astype(str).str.len().median()) > 60)
        if 1 <= len(counts_all) <= 20:
            usable_counts = series.loc[~duplicate_mask].dropna().astype(str).value_counts()
            info.update(minimum_class_count=int(counts_all.min()), usable_minimum_class_count=int(usable_counts.min()) if len(usable_counts) else 0, category_counts={str(key): int(value) for key, value in counts_all.items()})
        if numeric:
            info["infinite_values"] = int(np.isinf(non_null.to_numpy(dtype=float)).sum()) + int(df.attrs.get("nonfinite_counts", {}).get(name, 0))
        if not numeric and name not in dates and len(non_null):
            parsed_share = float(pd.to_numeric(non_null, errors="coerce").notna().mean())
            info["mixed_numeric_text"] = 0.2 <= parsed_share < 0.95
        if numeric and len(non_null):
            non_null = non_null.replace([np.inf, -np.inf], np.nan).dropna()
            if non_null.empty:
                info.update(distribution=[], outliers=0, invalid_numeric=True)
                columns.append(info)
                continue
            scale = max(float(non_null.abs().max()), 1.0)
            scaled = non_null / scale
            q1, q3 = scaled.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_scaled, upper_scaled = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            lower = lower_scaled * scale if abs(lower_scaled) <= np.finfo(float).max / scale else None
            upper = upper_scaled * scale if abs(upper_scaled) <= np.finfo(float).max / scale else None
            normalized_series = series.replace([np.inf, -np.inf], np.nan) / scale
            outliers = normalized_series.notna() & ((normalized_series < lower_scaled) | (normalized_series > upper_scaled))
            outlier_rows |= outliers
            info.update({"outliers": int(outliers.sum()), "outlier_pct": round(float(outliers.mean() * 100), 2),
                         "outlier_bounds": {"lower": lower, "upper": upper}, "outlier_method": "1.5 × IQR"})
            info.update({"mean": float(scaled.mean() * scale), "median": float(scaled.median() * scale),
                         "std": float(scaled.std() * scale), "variance": float(scaled.var() * scale * scale) if scale <= np.sqrt(np.finfo(float).max) else None,
                         "skewness": float(scaled.skew()), "kurtosis": float(scaled.kurt()),
                         "zero_pct": float((non_null == 0).mean() * 100), "negative_pct": float((non_null < 0).mean() * 100),
                         "quantiles": {str(q): float(scaled.quantile(q) * scale) for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
                         "min": float(non_null.min()), "max": float(non_null.max())})
            values = sample[name].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            counts, edges = np.histogram(values / scale, bins=min(18, max(1, int(series.nunique()))))
            edges = edges * scale
            info["extreme_magnitude"] = scale > 1e100
            if info["extreme_magnitude"]:
                info["numeric_scale_note"] = "Extreme finite magnitudes use scaled profiling; unrepresentable variance/bounds are null and predictive ML may be unsuitable."
            info["distribution"] = [
                {"label": f"{edges[i]:.3g}", "value": int(count), "upper": float(edges[i + 1])}
                for i, count in enumerate(counts)
            ]
        else:
            counts = sample[name].fillna("(missing)").astype(str).value_counts().head(10)
            info["distribution"] = [{"label": str(k)[:50], "value": int(v)} for k, v in counts.items()]
            if name in dates:
                parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed").dropna().sort_values()
                distinct = parsed.drop_duplicates()
                gaps = distinct.diff().dropna().dt.total_seconds()
                frequency = pd.infer_freq(distinct) if len(distinct) >= 3 else None
                median_gap = float(gaps.median()) if len(gaps) else None
                missing_periods = int(np.maximum(np.rint(gaps / median_gap) - 1, 0).sum()) if median_gap and median_gap > 0 else 0
                info["temporal"] = {"min": parsed.min().isoformat() if len(parsed) else None, "max": parsed.max().isoformat() if len(parsed) else None,
                                    "invalid_dates": int(series.notna().sum() - len(parsed)), "duplicate_dates": int(parsed.duplicated().sum()),
                                    "frequency": frequency, "median_interval_seconds": median_gap, "regular": bool(frequency),
                                    "missing_periods_estimate": missing_periods, "trend_suitable": len(distinct) >= 8,
                                    "seasonality_suitable": len(distinct) >= 24,
                                    "future_timestamps": int((parsed > pd.Timestamp.now(tz="UTC")).sum())}
            else:
                probabilities = counts_all / max(len(non_null), 1)
                info.update(entropy=float(-(probabilities * np.log2(probabilities)).sum()),
                            dominant_category=str(counts_all.index[0]) if len(counts_all) else None, dominant_share=dominant_share,
                            rare_category_pct=float(counts_all[counts_all / max(len(non_null), 1) < 0.01].sum() / max(len(non_null), 1) * 100))
        columns.append(info)
    numeric_df = sample.select_dtypes(include="number").iloc[:, :16].replace([np.inf, -np.inf], np.nan)
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
    result = safe_json({
        "rows": len(df), "columns": columns, "missing_cells": total_missing, "missing_pct": round(missing_pct, 2),
        "duplicates": duplicates, "quality_score": quality, "correlations": correlations,
        "correlation_columns": numeric_names, "scatter": scatter, "scatter_columns": numeric_names[:2],
        "sampled_rows": len(sample), "insights": insights,
        "outlier_rows": int(outlier_rows.sum()), "outlier_method": "1.5 × IQR per numeric column",
        "profile_version": PROFILE_VERSION, "memory_bytes": int(df.memory_usage(deep=True).sum()), "duplicate_pct": duplicate_pct,
        "type_ratios": {kind: sum(column["kind"] == kind for column in columns) / max(len(columns), 1) for kind in ("numeric", "categorical", "datetime")},
        "constant_columns": [column["name"] for column in columns if column.get("constant")],
        "near_constant_columns": [column["name"] for column in columns if column.get("near_constant")],
        "possible_ids": [column["name"] for column in columns if column.get("possible_id")],
        "high_cardinality_columns": [column["name"] for column in columns if column.get("high_cardinality")],
        "free_text_columns": [column["name"] for column in columns if column.get("possible_free_text")],
        "invalid_values": sum(column.get("infinite_values", 0) + column.get("temporal", {}).get("invalid_dates", 0) for column in columns),
        "ingestion_note": "Non-finite numeric inputs are recorded and represented as missing values; outliers are retained.",
    })
    from .problem import rank_targets
    result["target_candidates"] = [candidate.model_dump() for candidate in rank_targets(result)]
    return result


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
    counts = {name: int(np.isinf(df[name].to_numpy(dtype=float)).sum()) for name in df.select_dtypes(include="number")}
    normalized = df.replace([np.inf, -np.inf], np.nan)
    normalized.attrs["nonfinite_counts"] = counts
    return normalized


def register_frame(db, owner_id: str, df: pd.DataFrame, filename: str, raw: bytes | None = None):
    df = normalize_frame(df)
    identifier = new_id()
    path = settings.data_dir / "datasets" / f"{identifier}.parquet"
    df.to_parquet(path, index=False)
    digest = hashlib.sha256(raw if raw is not None else path.read_bytes()).hexdigest()
    dataset = Dataset(
        id=identifier, owner_id=owner_id,
        name=Path(safe_filename(filename)).stem.replace("_", " ").title(), filename=safe_filename(filename),
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
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if len(members) > 4096 or sum(member.file_size for member in members) > 128 * 1024 * 1024:
                    raise HTTPException(422, "The spreadsheet archive exceeds the safe expanded-size limit.")
                for member in members:
                    path = PurePosixPath(member.filename.replace("\\", "/"))
                    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]) or member.flag_bits & 1:
                        raise HTTPException(422, "The spreadsheet archive contains an unsafe path or encrypted member.")
                    if member.file_size > 64 * 1024 * 1024 or member.file_size / max(member.compress_size, 1) > 1000:
                        raise HTTPException(422, "The spreadsheet archive contains an oversized compressed member.")
            return pd.read_excel(io.BytesIO(raw), nrows=settings.max_dataset_rows + 1)
        raise HTTPException(415, "Upload a CSV or XLSX file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read the file: {str(exc)[:180]}") from exc


def safe_filename(filename):
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)[:255]
    return name or "dataset.csv"


def demo_frame(kind: str = "churn") -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(42)
    if kind == "temporal":
        dates = np.repeat(pd.date_range("2022-01-01", periods=48, freq="MS"), 4)
        frame = pd.DataFrame({"order_date": dates, "region": ["North", "South", "East", "West"] * 48,
                              "price": rng.uniform(10, 30, len(dates)).round(2)})
        frame["sales"] = (np.repeat(np.linspace(240, 120, 48), 4) + np.tile([15, 0, -10, 5], 48) + rng.normal(0, 5, len(frame))).round(2)
        return frame, "monthly_sales.csv"
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
