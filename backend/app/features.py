"""Portable feature engineering included with every exported model package."""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def date_columns(frame):
    result = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_datetime64_any_dtype(series):
            result.append(name)
        elif not pd.api.types.is_numeric_dtype(series) and (re.search(r"date|timestamp|(^|_)time($|_)", name, re.I) or series.dropna().head(80).astype(str).str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() >= 0.9):
            sample = series.dropna().head(80)
            if len(sample) and pd.to_datetime(sample, errors="coerce", utc=True, format="mixed").notna().mean() >= 0.9:
                result.append(name)
    return result


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self, dates=()):
        self.dates = dates

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        frame = X.copy()
        for name in frame.select_dtypes(include="bool"):
            frame[name] = frame[name].astype(float)
        for name in self.dates:
            parsed = pd.to_datetime(frame[name], errors="coerce", utc=True, format="mixed")
            frame[f"{name}__year"] = parsed.dt.year.astype(float)
            frame[f"{name}__month"] = parsed.dt.month.astype(float)
            frame[f"{name}__weekday"] = parsed.dt.dayofweek.astype(float)
            frame[f"{name}__month_sin"] = np.sin(2 * np.pi * parsed.dt.month / 12)
            frame[f"{name}__month_cos"] = np.cos(2 * np.pi * parsed.dt.month / 12)
            frame = frame.drop(columns=name)
        for name in frame.select_dtypes(exclude="number"):
            frame[name] = frame[name].map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
        return frame.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        result = [name for name in names if name not in self.dates]
        for name in self.dates:
            result.extend(f"{name}__{suffix}" for suffix in ("year", "month", "weekday", "month_sin", "month_cos"))
        return np.asarray(result, dtype=object)
