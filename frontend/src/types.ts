export type User = { id: string; name: string; email: string }
export type Column = {
  name: string; dtype: string; kind: 'numeric' | 'categorical' | 'datetime'; missing: number;
  missing_pct: number; unique: number; examples: string[]; mean?: number; median?: number;
  min?: number; max?: number; distribution: { label: string; value: number }[]
  outliers?: number; outlier_pct?: number
  constant?: boolean; near_constant?: boolean; possible_id?: boolean; high_cardinality?: boolean; possible_free_text?: boolean;
  infinite_values?: number; entropy?: number; skewness?: number; variance?: number; mixed_numeric_text?: boolean;
  temporal?: { min: string; max: string; regular: boolean; frequency: string | null; missing_periods_estimate: number; future_timestamps: number }
}
export type Profile = {
  rows: number; columns: Column[]; missing_cells: number; missing_pct: number; duplicates: number;
  quality_score: number; correlations: { x: string; y: string; value: number | null }[];
  correlation_columns: string[]; sampled_rows: number; scatter: { x: number; y: number }[];
  scatter_columns: string[]; insights: { title: string; text: string; kind: string }[]
  outlier_rows?: number; outlier_method?: string
  invalid_values?: number; memory_bytes?: number; duplicate_pct?: number;
  constant_columns?: string[]; near_constant_columns?: string[]; possible_ids?: string[]; high_cardinality_columns?: string[]; free_text_columns?: string[];
  target_candidates?: TargetCandidate[]
}
export type Dataset = {
  id: string; name: string; filename: string; row_count: number; column_count: number;
  size_bytes: number; fingerprint: string; quality_score: number; created_at: string; profile?: Profile
}
export type Experiment = {
  name: string; status: string; validation_metrics: Record<string, number>; training_metrics: Record<string, number>;
  parameters: Record<string, unknown>; duration_seconds: number; error?: string
  family?: string; selection_score?: number; reason?: string; cross_validation?: CrossValidation
  suitability?: string; stability?: ModelStability; inference_time?: { seconds: number; rows: number }; test_metrics?: Record<string, number> | null;
}
export type CrossValidation = { status: string; strategy?: string; folds?: number; mean?: number; std?: number; scores?: number[]; reason?: string }
export type ModelResult = {
  model_name: string | null; task: string; target: string | null; primary_metric: string; metrics: Record<string, number>;
  experiments: Experiment[]; feature_importance: { feature: string; importance: number; std: number }[];
  diagnostics: { confusion_matrix?: number[][]; classes?: string[]; roc_curve?: { fpr: number; tpr: number }[];
    pr_curve?: { recall: number; precision: number }[];
    predicted_vs_actual?: { actual: number; predicted: number; residual: number }[];
    cluster_projection?: { x: number; y: number; cluster: number }[] };
  excluded_features: { feature: string; reason: string }[];
  input_schema: { name: string; type: string; nullable: boolean; example: unknown; categories?: string[] | null }[];
  classes: string[]; positive_class: string | null; split: { train: number; validation: number; test: number; strategy: string };
  duration_seconds: number; report: string; report_source: string; selection_note: string;
  dataset_fingerprint: string; sklearn_version: string; seed: number
  cross_validation?: CrossValidation;
  pipeline?: { name: string; detail: string; status: string }[];
  problem_detection?: { task: string; target: string | null; reason: string };
  feature_engineering?: { date_columns: string[]; generated_date_features: string[]; missing_indicators: boolean; outlier_policy: string };
  shap?: { status: string; method?: string; output?: string; reason?: string; sample_rows?: number;
    feature_importance: { feature: string; importance: number }[];
    examples: { row_index: number; prediction: number; contributions: { feature: string; value: number }[] }[]; directions?: FeatureDirection[] };
  error_analysis?: { kind: string; note?: string; error_count?: number; p95_absolute_error?: number;
    examples: { row_index: number; actual?: string | number; predicted: string | number; error: number; record: Record<string, unknown> }[] };
  cluster_profiles?: { cluster: number; name: string; rows: number; share: number; features: Record<string, string | number> }[]
  analysis?: AnalyticalResult; calibration?: Calibration | null; model_stability?: ModelStability; selection_score?: number;
  class_imbalance?: { distribution: { class: string; rows: number; share: number }[]; minority_share: number; severe: boolean; policy: string } | null;
  leakage?: Leakage; baseline_metrics?: Record<string, number> | null; model_limitations?: string[];
  inference_time?: { seconds: number; rows: number }; llm_explanation?: string | null;
  training_warnings?: string[];
  threshold_analysis?: { rows: { threshold: number; precision: number; recall: number; f1: number; balanced_accuracy: number }[]; data: string; policy: string } | null
}
export type Run = {
  id: string; dataset_id: string; objective: string; target: string; task: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'; progress: number; stage: string;
  config: Record<string, unknown>; events: { at: string; stage: string; detail: string; step_id?: string; evidence_ids?: string[]; decision?: string }[];
  result: ModelResult | null; error: string | null; created_at: string; finished_at: string | null
}
export type TargetCandidate = { name: string; task: string; score: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW'; reasons: string[]; score_kind: string }
export type AnalysisStep = { id: string; kind: string; title: string; reason: string; columns: string[]; expected_outputs: string[]; status: string }
export type MLDecision = { use_ml: boolean; task: string; target: string | null; reason: string; limitations: string[] }
export type AnalysisPlan = { version: string; objective: string; intent: string; task: string; target: string | null; target_candidates: TargetCandidate[]; steps: AnalysisStep[]; ml_decision: MLDecision; reasons: string[]; expected_outputs: string[]; seed: number }
export type Evidence = { id: string; kind: string; title: string; method: string; columns: string[]; values: Record<string, unknown>; sample_rows: number; population_rows: number; assumptions: string[]; limitations: string[]; status: string; provenance: string }
export type Finding = { id: string; type: string; severity: string; confidence: string; finding: string; why_it_matters: string; evidence_ids: string[]; confidence_reasons: string[]; limitations: string[] }
export type Recommendation = { id: string; insight_id: string; action: string; rationale: string; evidence_ids: string[]; limitation: string }
export type AnalyticalResult = { plan: AnalysisPlan; ml_decision: MLDecision; evidence: Evidence[]; insights: Finding[]; recommendations: Recommendation[]; hypotheses: { id: string; hypothesis: string; observation: string; conclusion: string; status: string; evidence_id: string; p_value: number; q_value: number; effect_size: number; effect_measure: string; limitation: string }[]; limitations: string[]; executive_summary: string; reproducibility: Record<string, unknown>; report_sections: { title: string; text: string }[]; leakage: Leakage }
export type Leakage = { target?: string | null; flags: { feature: string; reasons: string[]; evidence: Record<string, number>; action: string }[]; policy: string }
export type ModelStability = { status: string; relative_std?: number; rule?: string; reason?: string }
export type Calibration = { status: string; brier_score: number; expected_calibration_error?: number; reliability: { predicted_probability: number; observed_frequency: number; rows: number }[]; note: string; rule?: string }
export type FeatureDirection = { feature: string; direction: string; coefficient: number; sample_rows: number; output: string; evidence: string; limitation: string }
export type ExperimentGroup = { id: string; name: string; dataset_id: string; created_at: string; runs: Run[]; best_run_id: string | null; selection_note: string }
export type PredictionItem = { label?: string; value?: number; cluster?: number; probabilities?: Record<string, number> }
export type PredictionResponse = { predictions: PredictionItem[]; model_id: string; model_name: string; model_version: string; evaluation_context: { test_metrics: Record<string, number>; test_rows: number; note: string } }
export type ScenarioResponse = { label: string; baseline: PredictionItem; scenario: PredictionItem; changes: Record<string, unknown>; model_name: string; model_version: string; limitation: string; evaluation_context: PredictionResponse['evaluation_context'] }
export type DashboardData = {
  datasets: Dataset[]; runs: Run[];
  stats: { datasets: number; rows: number; models: number; active_runs: number; average_quality: number }
}
