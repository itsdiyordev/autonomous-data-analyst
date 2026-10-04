"""Bounded temporal diagnostics and chronological, evaluated baseline forecasts."""
import numpy as np
import pandas as pd

from .evidence import record_evidence


def _prediction(history, method, alpha=0.3):
    if method == "Naive":
        return float(history[-1])
    if method == "Moving average":
        return float(np.mean(history[-3:]))
    state = float(history[0])
    for value in history[1:]:
        state = alpha * value + (1 - alpha) * state
    return state


def _evaluate(history, actual, method):
    history = list(history)
    predictions = []
    for value in actual:
        predictions.append(_prediction(history, method))
        history.append(value)
    error = np.asarray(actual) - predictions
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def analyze_temporal(frame, date, measure, evidence, forecast_requested=False):
    dates = pd.to_datetime(frame[date], errors="coerce", utc=True, format="mixed")
    values = pd.to_numeric(frame[measure], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.abs().max() > 1e100:
        return record_evidence(evidence, "temporal", f"{measure} over time", "Numeric range suitability", columns=[date, measure], status="skipped", population_rows=len(frame), limitations=["Extreme numeric magnitudes exceed the stable temporal-analysis contract; review measurement units. Original values are retained."])
    usable = pd.DataFrame({"date": dates, "value": values}).dropna().sort_values("date")
    if usable.date.nunique() < 8:
        return record_evidence(evidence, "temporal", f"{measure} over time", "Temporal profiling", columns=[date, measure], status="skipped", sample_rows=len(usable), population_rows=len(frame), limitations=["At least eight distinct valid dates and finite measurements are required."])
    gap = usable.date.drop_duplicates().diff().dropna().dt.total_seconds().median() / 86400
    span = (usable.date.max() - usable.date.min()).days
    frequency = "YS" if gap > 300 or span > 60000 else "MS" if gap >= 25 or span > 2000 else "W-MON" if gap >= 6 else "h" if gap < 1 and span * 24 <= 2000 else "D"
    additive = any(token in measure.lower() for token in ("sales", "revenue", "quantity", "profit", "volume"))
    series = usable.set_index("date").value.resample(frequency)
    counts = series.count()
    aggregated = series.sum(min_count=1) if additive else series.mean()
    complete = aggregated.dropna()
    if not np.isfinite(complete.to_numpy()).all():
        return record_evidence(evidence, "temporal", f"{measure} over time", "Calendar aggregation", columns=[date, measure], status="skipped", sample_rows=len(usable), population_rows=len(frame), limitations=["Aggregate values exceed the supported finite numeric range; review measurement units before trend/forecast inference."])
    if len(complete) < 2:
        return record_evidence(evidence, "temporal", f"{measure} over time", "Calendar aggregation", columns=[date, measure], status="skipped", sample_rows=len(usable), population_rows=len(frame), limitations=["Too few distinct calendar periods after the recorded aggregation."])
    x = np.arange(len(aggregated))[aggregated.notna()]
    slope = float(np.polyfit(x, complete.to_numpy(), 1)[0]) if len(complete) > 1 else 0.0
    prior = float(aggregated.iloc[-2])
    growth = (float(aggregated.iloc[-1]) - prior) / abs(prior) * 100 if pd.notna(prior) and prior else None
    rolling = aggregated.rolling(3, min_periods=2).mean()
    residual = aggregated - aggregated.shift(1).rolling(3, min_periods=2).mean()
    scale = float((residual - residual.median()).abs().median() * 1.4826)
    anomaly_mask = (residual - residual.median()).abs() > 3 * scale if scale > 0 else pd.Series(False, index=aggregated.index)
    lag = 12 if frequency == "MS" else 7 if frequency == "D" else 52 if frequency == "W-MON" else 0
    seasonality = None
    if lag and len(aggregated) >= lag * 2 and aggregated.notna().all():
        detrended = aggregated.to_numpy() - np.polyval(np.polyfit(np.arange(len(aggregated)), aggregated, 1), np.arange(len(aggregated)))
        if np.std(detrended) > 1e-9:
            seasonality = float(np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1])
    changes = rolling.diff().abs()
    change_scale = float((changes - changes.median()).abs().median() * 1.4826)
    candidates = changes[changes > changes.median() + 3 * change_scale].nlargest(5) if change_scale > 0 else changes.iloc[:0]
    forecast = {"status": "skipped", "reason": "Evaluated forecasting needs at least 24 complete regular aggregated periods without missing periods."}
    if forecast_requested and len(aggregated) >= 24 and aggregated.notna().all():
        array = aggregated.to_numpy()
        a, b = int(len(array) * 0.6), int(len(array) * 0.8)
        models = [{"name": method, "validation": _evaluate(array[:a], array[a:b], method)} for method in ("Naive", "Moving average", "Exponential smoothing")]
        winner = min(models, key=lambda item: item["validation"]["mae"])
        test = _evaluate(array[:b], array[b:], winner["name"])
        history = array.tolist()
        horizon = []
        for _ in range(6):
            value = _prediction(history, winner["name"])
            horizon.append(value)
            history.append(value)
        future = pd.date_range(aggregated.index[-1], periods=7, freq=frequency)[1:]
        forecast = {"status": "completed", "selected_model": winner["name"], "candidates": models, "test_metrics": test,
                    "split": {"train": a, "validation": b - a, "test": len(array) - b},
                    "horizon": [{"date": date.isoformat(), "value": value} for date, value in zip(future, horizon)],
                    "evaluation": "Rolling one-step chronological validation selects the baseline; only its separate later test periods are evaluated. Final output is a six-period iterative forecast.",
                    "limitation": "Baselines do not incorporate external drivers; forecast uncertainty intervals are not estimated. Historical one-step error is not a six-step guarantee."}
    elif not forecast_requested:
        forecast["reason"] = "The objective did not request forecasting; descriptive temporal analysis was selected."
    points = [{"date": date.isoformat(), "value": float(value) if pd.notna(value) else None, "rows": int(counts.loc[date]), "rolling_average": float(rolling.loc[date]) if pd.notna(rolling.loc[date]) else None} for date, value in aggregated.items()]
    return record_evidence(evidence, "temporal", f"{measure} over time", "Calendar aggregation, linear descriptive trend and robust residual review", columns=[date, measure],
                           values={"frequency": frequency, "aggregation": "sum" if additive else "mean", "periods": len(aggregated), "missing_periods": int(aggregated.isna().sum()), "duplicate_dates": int(usable.date.duplicated().sum()),
                                   "invalid_rows": len(frame) - len(usable), "trend": "positive" if slope > 0 else "negative" if slope < 0 else "flat", "slope_per_period": slope, "latest_growth_pct": growth,
                                   "points": points[-400:], "seasonality_lag": lag, "seasonality_autocorrelation": seasonality,
                                   "anomalies": [date.isoformat() for date in aggregated.index[anomaly_mask.fillna(False)]][:20],
                                   "change_point_candidates": [date.isoformat() for date in candidates.index], "forecast": forecast},
                           sample_rows=len(usable), population_rows=len(frame), assumptions=["Calendar buckets and aggregation are appropriate for the measurement; verify with domain knowledge."],
                           limitations=["Latest or first periods may be partial; changing observation counts can change aggregate totals.", "Trend, seasonality and change-point candidates are descriptive, not causal or confirmed structural breaks.", "Missing periods are displayed and are not silently filled; future timestamps may represent planned events."])
