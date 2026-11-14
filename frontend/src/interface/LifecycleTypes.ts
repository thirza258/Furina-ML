/**
 * TypeScript mirrors of the Furina ML lifecycle API (`/api/lifecycle/`).
 *
 * Like `MlTypes.ts`, this is the single source of truth for these shapes on
 * the frontend. Field names and nullability mirror
 * `backend/lifecycle/serializers.py` exactly — a hand-copied shape here is how
 * the two sides silently drift apart.
 */

import type { MetricMap, PredictField, TaskType, TrainingRunDetail } from "./MlTypes";

/* ------------------------------------------------------------------ */
/* Dataset versions                                                    */
/* ------------------------------------------------------------------ */

export type VersionSource = "upload" | "preprocessed" | "derived";

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  dataset_name: string;
  version: number;
  parent: string | null;
  name: string;
  description: string;
  source_type: VersionSource;
  content_hash: string;
  schema_hash: string;
  row_count: number;
  column_count: number;
  recipe: unknown | null;
  created_at: string;
}

export interface DatasetVersionLineage {
  version: DatasetVersion;
  lineage: {
    id: string;
    version: number;
    name: string;
    source_type: VersionSource;
    created_at: string;
  }[];
}

/* ------------------------------------------------------------------ */
/* Leakage & resources                                                 */
/* ------------------------------------------------------------------ */

export type LeakageSeverity = "info" | "warning" | "critical";

export interface LeakageFinding {
  code: string;
  severity: LeakageSeverity;
  message: string;
  columns: string[];
}

export interface LeakageSummary {
  severity: LeakageSeverity;
  count: number;
  findings: LeakageFinding[];
}

export interface ResourceEstimate {
  estimated_gb: number;
  available_gb: number;
  risk: "low" | "medium" | "high";
  rows: number;
  columns: number;
}

/* ------------------------------------------------------------------ */
/* Experiments                                                         */
/* ------------------------------------------------------------------ */

export type ObjectiveDirection = "maximize" | "minimize";

export interface ObjectiveChoice {
  key: string;
  label: string;
  direction: ObjectiveDirection;
}

export interface Experiment {
  id: string;
  dataset_version_id: string;
  name: string;
  description: string;
  task_type: TaskType;
  objective_metric: string;
  objective_direction: ObjectiveDirection;
  tags: string[];
  best_run_id: string | null;
  best_metric_value: number | null;
  n_runs: number;
  created_at: string;
}

export interface ExperimentDetail extends Experiment {
  dataset_version: DatasetVersion;
  dataset_name: string;
  objective_choices: ObjectiveChoice[];
}

export interface CreateExperimentRequest {
  dataset_id: string;
  dataset_version_id?: string | null;
  name: string;
  description?: string;
  task_type?: TaskType | "auto";
  objective_metric?: string;
  objective_direction?: ObjectiveDirection;
  tags?: string[];
}

/* ------------------------------------------------------------------ */
/* Runs                                                                */
/* ------------------------------------------------------------------ */

export type RunStatus = "queued" | "running" | "success" | "failed" | "cancelled";

export interface RunSummary {
  id: string;
  experiment_id: string;
  experiment_name: string;
  number: number;
  trial_index: number | null;
  algorithm: string;
  algorithm_label: string;
  task_type: TaskType;
  target_column: string;
  status: RunStatus;
  objective_metric: string;
  primary_metric: number | null;
  metrics: MetricMap;
  train_metrics: MetricMap;
  duration_ms: number;
  cv_folds: number | null;
  config_hash: string;
  n_train: number;
  n_test: number;
  created_at: string;
}

/** One curve's worth of points, downsampled server-side. */
export interface RocoCurve {
  fpr: number[];
  tpr: number[];
  auc: number | null;
}

export interface ThresholdPoint {
  threshold: number;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  predicted_positive_rate: number | null;
}

export interface Diagnosis {
  verdict:
    | "good_fit"
    | "overfitting"
    | "underfitting"
    | "potential_leakage"
    | "unknown";
  confidence: "low" | "medium" | "high";
  reasons: string[];
  train_metric: number | null;
  test_metric: number | null;
  gap: number | null;
  metric_name?: string;
}

export interface RunDiagnostics {
  diagnosis: Diagnosis;
  note?: string;
  roc?: RocoCurve | null;
  precision_recall?: { recall: number[]; precision: number[] } | null;
  calibration?: {
    prob_true_mean: number[];
    prob_pred_mean: number[];
    brier: number | null;
  } | null;
  threshold_analysis?: ThresholdPoint[] | null;
  residuals?: { mean: number | null; std: number | null; bias: number | null } | null;
  absolute_error?: { mean: number | null; median: number | null; max: number | null } | null;
  percentage_error?: number | null;
  worst_predictions?: { actual: number | null; predicted: number | null; error: number | null }[] | null;
  residual_distribution?: number[] | null;
}

export interface RunDetail extends RunSummary {
  dataset_version_id: string;
  dataset_hash: string;
  dataset_row_count: number;
  feature_columns: string[];
  hyperparameters: Record<string, unknown>;
  effective_hyperparameters: Record<string, unknown>;
  test_size: number;
  random_state: number | null;
  stratify: boolean;
  scale_features: boolean;
  preprocessing: Record<string, unknown>;
  environment: {
    python?: string;
    os?: string;
    furina_version?: string;
    git_commit?: string | null;
    packages?: Record<string, string | null>;
    migrated?: boolean;
  } | null;
  cv: TrainingRunDetail["cv"];
  classes: string[] | null;
  confusion_matrix: TrainingRunDetail["confusion_matrix"];
  per_class: TrainingRunDetail["per_class"];
  feature_importance: TrainingRunDetail["feature_importance"];
  diagnostics: RunDiagnostics | null;
  residual_sample: TrainingRunDetail["residual_sample"];
  input_schema: PredictField[];
  warnings: string[];
  error: string | null;
  model_size_bytes: number;
  model_download_url: string;
  reproduce_url: string;
}

export interface CompareTable {
  task_type: TaskType;
  metrics: { key: string; label: string }[];
  objective_metric: string;
  objective_direction: ObjectiveDirection;
  rows: Record<string, string | number | null>[];
}

export interface RunComparison {
  task_types: TaskType[];
  tables: Partial<Record<TaskType, CompareTable>>;
}

/* ------------------------------------------------------------------ */
/* Jobs                                                                */
/* ------------------------------------------------------------------ */

export type JobKind = "train" | "hpo" | "automl" | "reproduce";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  progress: number;
  total_units: number;
  completed_units: number;
  cancel_requested: boolean;
  error: string | null;
  suggested_action: string | null;
  best_run_id: string | null;
  payload: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainJobRequest {
  target_column: string;
  feature_columns?: string[] | null;
  algorithm?: string;
  task_type?: TaskType | "auto";
  test_size?: number;
  random_state?: number | null;
  hyperparameters?: Record<string, unknown>;
  stratify?: boolean;
  cv_folds?: number | null;
  name?: string | null;
  run_now?: boolean;
}

export interface HpoJobRequest extends TrainJobRequest {
  method?: "random" | "grid" | "successive_halving";
  n_trials?: number;
  seed?: number;
  param_space?: Record<string, unknown> | null;
}

export interface AutoMlJobRequest {
  target_column: string;
  feature_columns?: string[] | null;
  task_type?: TaskType | "auto";
  time_budget_seconds?: number;
  max_models?: number;
  cv_folds?: number;
  seed?: number;
  test_size?: number;
  run_now?: boolean;
}

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

export type ModelState = "candidate" | "validated" | "staging" | "production" | "archived";

export interface ModelCard {
  model: string;
  version: number;
  task: string;
  algorithm: string;
  target_column: string;
  dataset: { dataset_version: string; rows: number; columns: number };
  training_method: Record<string, unknown>;
  features: string[];
  metrics: MetricMap;
  validation_strategy: string;
  known_limitations: string[];
  intended_use: string;
  out_of_scope_use: string[];
  deployment_status: string;
  training_timestamp: string | null;
  data_quality_note?: string;
}

export interface ModelSignatureField {
  name: string;
  type: string;
  required: boolean;
  categories: string[] | null;
  minimum: number | null;
  maximum: number | null;
  default: string | number | null;
}

export interface ModelSignature {
  task_type: TaskType;
  target_column: string;
  classes: string[] | null;
  features: ModelSignatureField[];
  example_request: Record<string, string | number | null>;
}

export interface ModelVersion {
  id: string;
  model_id: string;
  model_name: string;
  version: number;
  status: ModelState;
  status_label: string;
  source_run_id: string | null;
  dataset_version_id: string | null;
  metrics: MetricMap;
  model_card: ModelCard;
  signature: ModelSignature;
  created_at: string;
  states: ModelState[];
}

export interface ModelSummary {
  id: string;
  name: string;
  task_type: TaskType;
  description: string;
  production_version: number | null;
  n_versions: number;
  created_at: string;
}

export interface ModelDetail extends ModelSummary {
  versions: ModelVersion[];
}

/* ------------------------------------------------------------------ */
/* Deployments & monitoring                                            */
/* ------------------------------------------------------------------ */

export interface Deployment {
  id: string;
  slug: string;
  model_version_id: string;
  model_name: string;
  model_version: number;
  status: ModelState;
  is_active: boolean;
  total_requests: number;
  total_errors: number;
  last_predicted_at: string | null;
  signature: ModelSignature;
  predict_url: string;
  batch_predict_url: string;
  openapi_url: string;
  client_snippets_url: string;
  created_at: string;
}

export interface DeploymentPrediction {
  prediction: string | number | null;
  probabilities: { label: string; probability: number }[] | null;
  probabilities_map?: Record<string, number>;
  model_version: string;
  notes?: string[];
}

export interface DriftFeature {
  feature: string;
  kind: string | null;
  status: "stable" | "moderate" | "significant" | "missing" | "unknown";
  psi: number | null;
  missing_delta: number | null;
  message: string;
}

export interface DriftReport {
  overall: "stable" | "moderate" | "significant" | "unknown";
  max_psi: number;
  rows: number;
  features: DriftFeature[];
}

export interface MonitoringSummary {
  deployment: Deployment;
  total_requests: number;
  total_errors: number;
  error_rate: number;
  latest_drift: DriftReport | null;
  latest_performance: Record<string, unknown> | null;
}

export interface ClientSnippets {
  predict_url: string;
  snippets: { curl: string; python: string; javascript: string };
}