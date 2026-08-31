/**
 * TypeScript mirrors of the Furina ML backend API.
 *
 * This file is the single source of truth for API shapes on the frontend.
 * Pages must import from here rather than declaring their own local copies —
 * a duplicated shape is how the frontend and backend silently drift apart.
 */

export type ColumnKind =
  | "numeric"
  | "categorical"
  | "boolean"
  | "datetime"
  | "text";

export type TaskType = "classification" | "regression";

/** What the user may ask for on the training form; "auto" is resolved server-side. */
export type RequestedTaskType = TaskType | "auto";

export interface TopValue {
  value: string;
  count: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  kind: ColumnKind;
  missing: number;
  missing_pct: number;
  unique: number;
  sample_values: string[];
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  top_values: TopValue[] | null;
  suitable_as_target: boolean;
  suggested_task: TaskType | null;
}

export interface DatasetPreview {
  headers: string[];
  rows: (string | number | null)[][];
}

export interface DatasetSummary {
  id: string;
  name: string;
  source: "upload" | "preprocessed";
  n_rows: number;
  n_cols: number;
  created_at: string;
}

export interface DatasetDetail extends DatasetSummary {
  /** null only for a dry-run preprocessing preview, which is never persisted. */
  id: string;
  parent: string | null;
  size_bytes: number;
  columns: ColumnProfile[];
  preview: DatasetPreview;
  download_url: string;
}

/** A dry-run preview is DatasetDetail-shaped but has no id yet. */
export type DatasetPreviewResult = Omit<DatasetDetail, "id"> & {
  id: string | null;
};

export interface HyperparamOption {
  value: string;
  label: string;
}

export interface HyperparamSpec {
  name: string;
  label: string;
  type: "int" | "float" | "select" | "bool";
  default: number | string | boolean | null;
  min: number | null;
  max: number | null;
  step: number | null;
  options: HyperparamOption[] | null;
  description: string;
  advanced: boolean;
}

export interface AlgorithmSpec {
  key: string;
  label: string;
  family: string;
  tasks: TaskType[];
  description: string;
  supports_proba: boolean;
  supports_feature_importance: boolean;
  hyperparameters: HyperparamSpec[];
}

export interface PredictField {
  name: string;
  kind: ColumnKind;
  required: boolean;
  options: string[] | null;
  min: number | null;
  max: number | null;
  default: string | number | null;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface PerClassMetric {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ResidualPoint {
  actual: number;
  predicted: number;
}

export interface CrossValidation {
  folds: number;
  scoring: string;
  scores: number[];
  mean: number;
  std: number;
}

/** Metric maps are open-ended: keys differ by task, values may be null. */
export type MetricMap = Record<string, number | null>;

export interface TrainingRunSummary {
  id: string;
  name: string;
  dataset_id: string;
  dataset_name: string;
  algorithm: string;
  algorithm_label: string;
  task_type: TaskType;
  target_column: string;
  status: "success" | "failed";
  primary_metric_name: string;
  primary_metric: number | null;
  /** The full test-split metric blob, flat `{name: value}`. */
  metrics: Record<string, number | null>;
  /** The same keys measured on the training split — the overfitting signal. */
  train_metrics: Record<string, number | null>;
  n_train: number;
  n_test: number;
  duration_ms: number;
  cv_folds: number | null;
  created_at: string;
}

export interface TrainingRunDetail extends TrainingRunSummary {
  feature_columns: string[];
  hyperparameters: Record<string, unknown>;
  effective_hyperparameters: Record<string, unknown>;
  test_size: number;
  random_state: number | null;
  stratify: boolean;
  scale_features: boolean;
  n_train: number;
  n_test: number;
  duration_ms: number;
  error: string | null;
  metrics: MetricMap;
  train_metrics: MetricMap;
  cv: CrossValidation | null;
  classes: string[] | null;
  confusion_matrix: ConfusionMatrixData | null;
  per_class: PerClassMetric[] | null;
  feature_importance: FeatureImportance[] | null;
  residual_sample: ResidualPoint[] | null;
  input_schema: PredictField[];
  warnings: string[];
  model_download_url: string;
  report_download_url: string;
}

/* ------------------------------------------------------------------ */
/* Requests                                                            */
/* ------------------------------------------------------------------ */

export type PreprocessOp =
  | "remove_duplicates"
  | "drop_columns"
  | "drop_missing_rows"
  | "fill_mean"
  | "fill_median"
  | "fill_mode"
  | "fill_constant"
  | "fill_empty_string"
  | "strip_whitespace"
  | "lowercase"
  | "remove_punctuation"
  | "remove_outliers_iqr"
  | "remove_outliers_zscore"
  | "minmax_scale"
  | "standard_scale"
  | "robust_scale"
  | "one_hot_encode"
  | "label_encode"
  | "pca"
  | "select_k_best"
  | "lda";

export interface PreprocessStep {
  op: PreprocessOp;
  /** null means "every column this operation sensibly applies to". */
  columns: string[] | null;
  params?: Record<string, unknown> | null;
}

export interface PreprocessRequest {
  steps: PreprocessStep[];
  dry_run?: boolean;
  name?: string | null;
}

export interface PreprocessStepReport {
  op: string;
  columns: string[];
  message: string;
  skipped: boolean;
  rows_after: number;
  cols_after: number;
}

export interface PreprocessReport {
  rows_before: number;
  rows_after: number;
  cols_before: number;
  cols_after: number;
  steps: PreprocessStepReport[];
  warnings: string[];
}

export interface PreprocessResponse {
  dataset: DatasetPreviewResult | null;
  report: PreprocessReport;
}

export interface TrainRequest {
  dataset_id: string;
  target_column: string;
  feature_columns: string[] | null;
  algorithm: string;
  task_type: RequestedTaskType;
  test_size: number;
  random_state: number | null;
  hyperparameters: Record<string, unknown>;
  stratify: boolean;
  scale_features: boolean;
  cv_folds: number | null;
  name?: string | null;
}

export interface PredictionResult {
  prediction: string | number;
  probabilities: { label: string; probability: number }[] | null;
}

export interface PredictResponse {
  predictions: PredictionResult[];
}

export interface HealthResponse {
  status: string;
  sklearn: string;
  pandas: string;
  numpy: string;
}

/** One turn of the assistant conversation. */
export interface AssistantMessage {
  role: "user" | "assistant";
  content: string;
}

/** What the assistant is told the user is looking at. */
export interface AssistantContext {
  tool?: string;
  dataset_id?: string | null;
  run_id?: string | null;
}

export interface AssistantStatus {
  enabled: boolean;
  model: string | null;
}

export interface AssistantReply {
  reply: string;
  model: string;
}

/** Normalized error surfaced by the service layer. */
export interface ApiError {
  detail: string;
  errors?: Record<string, string[]>;
  status?: number;
}
