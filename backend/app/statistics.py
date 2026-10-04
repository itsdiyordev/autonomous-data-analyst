"""Bounded, type-aware exploratory inference with effect sizes and FDR control."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .evidence import record_evidence


def strength(value, measure):
    cutoffs = (0.2, 0.5, 0.8) if measure == "cohens_d" else (0.01, 0.06, 0.14) if measure in {"eta_squared", "epsilon_squared"} else (0.1, 0.3, 0.5)
    magnitude = abs(value)
    return "large" if magnitude >= cutoffs[2] else "moderate" if magnitude >= cutoffs[1] else "small" if magnitude >= cutoffs[0] else "negligible"


def adjust_fdr(items, alpha=0.05):
    tested = [item for item in items if item.values.get("p_value") is not None and np.isfinite(item.values["p_value"])]
    order = sorted(tested, key=lambda item: item.values["p_value"])
    previous = 1.0
    for index in range(len(order) - 1, -1, -1):
        item = order[index]
        previous = min(previous, item.values["p_value"] * len(order) / (index + 1))
        item.values.update(q_value=float(previous), significant=previous < alpha, multiple_testing="Benjamini–Hochberg", tests_in_family=len(order))


def run_statistics(frame, profile, selected, evidence, max_pairs=24):
    metadata = {column["name"]: column for column in profile["columns"]}
    for name in dict.fromkeys(selected):
        if name in metadata and metadata[name].get("extreme_magnitude"):
            record_evidence(evidence, "statistics", f"Stable inference review: {name}", "Numeric range suitability", columns=[name], status="skipped", population_rows=len(frame), limitations=["Extreme numeric magnitudes exceed the supported inferential range; review measurement units. Scaled descriptive profiling remains available."])
    columns = [name for name in dict.fromkeys(selected) if name in metadata and metadata[name]["kind"] != "datetime" and not metadata[name].get("possible_id") and not metadata[name].get("extreme_magnitude") and metadata[name]["unique"] > 1]
    sample = frame.sample(min(5000, len(frame)), random_state=42)
    output = []
    for left, right in itertools.islice(itertools.combinations(columns, 2), max_pairs):
        kinds = [metadata[name]["kind"] for name in (left, right)]
        pair = sample[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
        assumptions = ["Rows are independent observations; observational association does not establish causality."]
        limitations = ["Exploratory analysis on pairwise complete cases; unmeasured confounders may affect results."]
        if len(sample) < len(frame):
            limitations.append("Computed on a deterministic bounded sample rather than the full dataset.")
        if any(column["kind"] == "datetime" for column in metadata.values()):
            limitations.append("Temporal dependence may violate independent-observation assumptions; p-values require caution.")
        if profile.get("duplicates", 0):
            limitations.append("Duplicate records may violate independent-observation assumptions and inflate apparent statistical evidence.")
        if len(pair) < 8 or min(pair.nunique()) < 2:
            record_evidence(evidence, "statistics", f"{left} ↔ {right}", "Insufficient complete observations", columns=[left, right], sample_rows=len(pair), population_rows=len(frame), status="skipped", limitations=["At least eight complete pairs are required."])
            continue
        if kinds == ["numeric", "numeric"]:
            for method, function in (("Pearson correlation", stats.pearsonr), ("Spearman correlation", stats.spearmanr)):
                result = function(pair[left].to_numpy(dtype=float), pair[right].to_numpy(dtype=float))
                coefficient, p = float(result.statistic), float(result.pvalue)
                if not np.isfinite(coefficient) or not np.isfinite(p):
                    continue
                values = {"coefficient": coefficient, "p_value": p, "effect_size": coefficient, "effect_measure": "correlation", "practical_strength": strength(coefficient, "correlation"), "missing_pairs": len(sample) - len(pair)}
                if method.startswith("Pearson") and len(pair) > 3:
                    z = np.arctanh(np.clip(coefficient, -0.999999, 0.999999))
                    radius = 1.96 / np.sqrt(len(pair) - 3)
                    values["confidence_interval"] = {"level": 0.95, "lower": float(np.tanh(z - radius)), "upper": float(np.tanh(z + radius)), "method": "Fisher z; assumes independent bivariate-normal observations"}
                output.append(record_evidence(evidence, "association", f"{left} ↔ {right}", method, columns=[left, right], values=values, sample_rows=len(pair), population_rows=len(frame), assumptions=assumptions, limitations=limitations + (["Spearman p-value uses an asymptotic approximation; small samples need independent confirmation."] if method.startswith("Spearman") else ["Pearson inference assumes a linear relationship and approximately bivariate-normal observations."])))
        elif kinds == ["categorical", "categorical"]:
            if max(pair.nunique()) > 20:
                continue
            table = pd.crosstab(pair[left].astype(str), pair[right].astype(str))
            if min(table.shape) < 2:
                continue
            chi, p, dof, expected = stats.chi2_contingency(table, correction=False)
            sparse = float((expected < 5).mean()) > 0.2 or float(expected.min()) < 1
            method = "Chi-square association"
            if sparse and table.shape == (2, 2):
                _, p = stats.fisher_exact(table.to_numpy())
                method = "Fisher exact association"
            elif sparse:
                limitations.append("Expected counts are too sparse for reliable chi-square inference; treat significance as provisional.")
            effect = float(np.sqrt(chi / (len(pair) * (min(table.shape) - 1))))
            values = {"p_value": float(p), "statistic": float(chi), "degrees_of_freedom": int(dof), "effect_size": effect, "effect_measure": "cramers_v", "practical_strength": strength(effect, "cramers_v"), "expected_min": float(expected.min()), "sparse_expected": sparse, "contingency": {"rows": table.index.tolist(), "columns": table.columns.tolist(), "counts": table.to_numpy().tolist()}}
            output.append(record_evidence(evidence, "association", f"{left} ↔ {right}", method, columns=[left, right], values=values, sample_rows=len(pair), population_rows=len(frame), assumptions=assumptions + ["Category counts represent independent observations."], limitations=limitations))
        else:
            category, numeric = (left, right) if kinds[0] == "categorical" else (right, left)
            if pair[category].nunique() > 20:
                continue
            groups = [(str(label), group[numeric].to_numpy(dtype=float)) for label, group in pair.groupby(category, sort=True)]
            if len(groups) < 2 or min(len(values) for _, values in groups) < 3:
                continue
            arrays = [values for _, values in groups]
            normal = all(np.std(values) > 0 and len(values) >= 8 and (abs(float(stats.skew(values))) < 2 and len(values) >= 30 or stats.shapiro(values[:500]).pvalue >= 0.05) for values in arrays)
            summaries = [{"group": label, "n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))} for label, values in groups]
            if len(groups) == 2:
                pooled = np.sqrt(sum((len(values) - 1) * np.var(values, ddof=1) for values in arrays) / (len(pair) - 2))
                effect = float((np.mean(arrays[0]) - np.mean(arrays[1])) / pooled) if pooled > 0 else 0.0
                if normal:
                    variance_ratio = max(np.var(values, ddof=1) for values in arrays) / min(np.var(values, ddof=1) for values in arrays)
                    equal_variance = variance_ratio <= 1.15 and stats.levene(*arrays).pvalue >= 0.1
                    result, method = stats.ttest_ind(*arrays, equal_var=equal_variance), "Student's t-test" if equal_variance else "Welch's t-test"
                else:
                    result, method = stats.mannwhitneyu(*arrays, alternative="two-sided", method="auto"), "Mann–Whitney U"
                measure = "cohens_d"
                extra = {"cohens_d": effect if pooled > 0 else None}
                if method == "Mann–Whitney U":
                    extra["rank_biserial"] = float(2 * result.statistic / (len(arrays[0]) * len(arrays[1])) - 1)
                    effect, measure = extra["rank_biserial"], "rank_biserial"
                    limitations.append("Mann–Whitney tests distribution/rank differences, not necessarily differences in means.")
            else:
                equal_variance = all(np.var(values) > 0 for values in arrays) and stats.levene(*arrays).pvalue >= 0.05
                if normal and equal_variance:
                    result, method, measure = stats.f_oneway(*arrays), "One-way ANOVA", "eta_squared"
                    total = float(np.sum((pair[numeric] - pair[numeric].mean()) ** 2))
                    effect = sum(len(values) * (values.mean() - pair[numeric].mean()) ** 2 for values in arrays) / total if total else 0.0
                else:
                    if pair[numeric].nunique() < 2:
                        continue
                    result, method, measure = stats.kruskal(*arrays), "Kruskal–Wallis", "epsilon_squared"
                    effect = max(0, float((result.statistic - len(groups) + 1) / (len(pair) - len(groups))))
                extra = {}
                limitations.append("An omnibus test does not identify which individual groups differ; no uncorrected post-hoc claims are made.")
            if not np.isfinite(result.pvalue):
                continue
            values = {"p_value": float(result.pvalue), "statistic": float(result.statistic), "effect_size": float(effect), "effect_measure": measure, "practical_strength": strength(effect, measure), "groups": summaries, "normality_screen_passed": normal, **extra}
            if len(groups) == 2 and normal:
                variances = [np.var(array, ddof=1) / len(array) for array in arrays]
                se = float(np.sqrt(sum(variances)))
                degrees = float(sum(variances) ** 2 / sum(variance ** 2 / (len(array) - 1) for variance, array in zip(variances, arrays)))
                difference = float(arrays[0].mean() - arrays[1].mean())
                radius = float(stats.t.ppf(0.975, degrees) * se)
                values["mean_difference_interval"] = {"level": 0.95, "lower": difference - radius, "upper": difference + radius, "method": "Welch uncertainty interval; independent observations and adequate mean sampling behavior assumed"}
            if measure == "eta_squared":
                values["partial_eta_squared"] = float(effect)
            output.append(record_evidence(evidence, "group_difference", f"{numeric} by {category}", method, columns=[category, numeric], values=values, sample_rows=len(pair), population_rows=len(frame), assumptions=assumptions + (["Welch allows unequal variances; mean inference needs adequate sampling behavior."] if normal else ["Rank methods do not require normality; distribution shapes affect interpretation."]), limitations=limitations))
    adjust_fdr(output)
    return output
