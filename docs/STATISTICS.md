# Statistical methods and interpretation

The deterministic engine performs exploratory inference on up to 5,000 sampled rows and 24 schema/objective-selected column pairs. It uses pairwise complete cases and records sample/population sizes. A p-value is not a probability that a hypothesis is true, and significance does not establish causal impact.

An explicit classification outcome is nominal even when stored as numeric label codes. Statistical comparisons use categorical/group methods for that outcome; its arbitrary codes are not used as numeric anomaly distances. This honors the existing task contract rather than inferring label meaning from numeric storage alone.

| Variable types | Test / method | Effect / uncertainty |
|---|---|---|
| Numeric ↔ numeric | Pearson and Spearman | Correlation coefficient; Pearson Fisher-z 95% interval |
| Categorical ↔ categorical | Chi-square; sparse 2×2 tables use Fisher exact | Cramér's V, contingency and expected-count checks |
| Two groups ↔ numeric | Student t only under the recorded normality/variance screen; otherwise Welch t or Mann–Whitney U | Cohen's d for mean tests; rank-biserial for rank tests; Welch mean-difference interval when appropriate |
| Multiple groups ↔ numeric | ANOVA with normality/equal-variance screen; otherwise Kruskal–Wallis | Eta squared (equal to partial eta squared for this one-way design) or epsilon squared |

## Assumptions and edge cases

- At least eight complete pairs and varying values are required; group comparisons require at least three observations per group.
- Normality screening uses bounded Shapiro samples or an adequate-size/skewness screen. Levene and a conservative variance-ratio screen gate Student t; Welch is the unequal-variance mean test.
- Sparse expected counts are recorded. Larger sparse contingency tables are provisional rather than treated as reliable chi-square evidence.
- Pearson inference assumes independent, approximately bivariate-normal observations and a linear relationship. Spearman p-values use an asymptotic approximation and need caution for small samples.
- Mann–Whitney tests ranks/distributions, not necessarily equality of means. An omnibus ANOVA/Kruskal result does not identify specific differing pairs; the engine does not invent uncorrected post-hoc conclusions.
- Missingness, repeated records, temporal dependence, selected variables and confounding can invalidate simple inferential assumptions. These concerns are recorded and limit confidence.

## Multiplicity and practical importance

All valid selected tests share a Benjamini–Hochberg adjustment family. Raw `p_value`, adjusted `q_value`, family size and alpha 0.05 are stored. BH relies on suitable dependence assumptions and does not make exploratory variable selection confirmatory.

The practical-strength rules are explicitly heuristic:

- Correlation / Cramér's V / rank-biserial: negligible below 0.10, small from 0.10, moderate from 0.30, large from 0.50.
- Cohen's d: thresholds 0.20 / 0.50 / 0.80.
- Eta/epsilon squared: thresholds 0.01 / 0.06 / 0.14.

An adjusted-significant relationship with negligible effect is described as statistically detectable but practically weak. Domain-specific thresholds may differ; no business value is inferred from p-values alone.

## Hypotheses and confidence

Each exploratory hypothesis stores the observed summaries, null/alternative statement, exact test/evidence ID, p/q-values, effect, deterministic confidence/reasons and conclusion. It references the same sample's computed test, not an independent confirmation. Failure to reject the null is not proof of no relationship.

HIGH relationship confidence requires q < .01, moderate/large effect, at least 100 complete observations and no recorded severe assumption concern. MEDIUM requires adjusted evidence, non-negligible effect and an adequate smaller sample; other cases are LOW. These labels are rule-based evidence assessments, not calibrated truth probabilities. Direct quality counts, temporal patterns, anomaly flags and model evidence have their own documented rules in `confidence.py`.

Independent replication, experimental design and domain expertise are necessary before causal or operational claims.
