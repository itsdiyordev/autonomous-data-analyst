export type User = { id: string; name: string; email: string }
export type Column = {
  name: string; dtype: string; kind: 'numeric' | 'categorical' | 'datetime'; missing: number;
  missing_pct: number; unique: number; examples: string[]; mean?: number; median?: number;
  min?: number; max?: number; distribution: { label: string; value: number }[]
  outliers?: number; outlier_pct?: number
}
export type Profile = {
  rows: number; columns: Column[]; missing_cells: number; missing_pct: number; duplicates: number;
  quality_score: number; correlations: { x: string; y: string; value: number | null }[];
  correlation_columns: string[]; sampled_rows: number; scatter: { x: number; y: number }[];
  scatter_columns: string[]; insights: { title: string; text: string; kind: string }[]
  outlier_rows?: number; outlier_method?: string
}
export type Dataset = {
  id: string; name: string; filename: string; row_count: number; column_count: number;
  size_bytes: number; fingerprint: string; quality_score: number; created_at: string; profile?: Profile
}
export type Experiment = {
  name: string; status: string; validation_metrics: Record<string, number>; training_metrics: Record<string, number>;
  parameters: Record<string, unknown>; duration_seconds: number; error?: string
  family?: string; selection_score?: number; reason?: string; cross_validation?: CrossValidation
}
export type CrossValidation = { status: string; strategy?: string; folds?: number; mean?: number; std?: number; scores?: number[]; reason?: string }
export type ModelResult = {
  model_name: string; task: string; target: string | null; primary_metric: string; metrics: Record<string, number>;
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
    examples: { row_index: number; prediction: number; contributions: { feature: string; value: number }[] }[] };
  error_analysis?: { kind: string; note?: string; error_count?: number; p95_absolute_error?: number;
    examples: { row_index: number; actual?: string | number; predicted: string | number; error: number; record: Record<string, unknown> }[] };
  cluster_profiles?: { cluster: number; name: string; rows: number; share: number; features: Record<string, string | number> }[]
}
export type Run = {
  id: string; dataset_id: string; objective: string; target: string; task: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'; progress: number; stage: string;
  config: Record<string, unknown>; events: { at: string; stage: string; detail: string }[];
  result: ModelResult | null; error: string | null; created_at: string; finished_at: string | null
}
export type DashboardData = {
  datasets: Dataset[]; runs: Run[];
  stats: { datasets: number; rows: number; models: number; active_runs: number; average_quality: number }
}
