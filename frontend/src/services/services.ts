/**
 * The single HTTP layer for the Furina ML API.
 *
 * Pages call these methods; nothing else in the app should construct an axios
 * request. Every method throws a normalized `ApiError` on failure so callers
 * can render `err.detail` directly.
 */
import axios, { AxiosError } from "axios";
import type { AxiosInstance } from "axios";
import type {
  Credentials,
  NotebookLaunch,
  ProgressResponse,
  Registration,
  SessionInfo,
  Submission,
} from "../interface/AccountTypes";

import type {
  AlgorithmSpec,
  ApiError,
  AssistantContext,
  AssistantMessage,
  AssistantReply,
  AssistantStatus,
  DatasetDetail,
  DatasetSummary,
  HealthResponse,
  PredictResponse,
  PreprocessRequest,
  PreprocessResponse,
  TaskType,
  TrainRequest,
  TrainingRunDetail,
  TrainingRunSummary,
} from "../interface/MlTypes";

import type {
  AutoMlJobRequest,
  ClientSnippets,
  CreateExperimentRequest,
  DatasetVersion,
  DatasetVersionLineage,
  Deployment,
  DeploymentPrediction,
  DriftReport,
  Experiment,
  ExperimentDetail,
  HpoJobRequest,
  Job,
  LeakageSummary,
  ModelDetail,
  ModelState,
  ModelSummary,
  ModelVersion,
  MonitoringSummary,
  ResourceEstimate,
  RunComparison,
  RunDetail,
  RunSummary,
  TrainJobRequest,
} from "../interface/LifecycleTypes";

declare global {
  interface Window {
    /** Set by the container's generated `/config.js`; absent in `npm run dev`. */
    __FURINA_API_BASE_URL__?: string;
    /** `owner/repo/blob/branch/notebooks` on GitHub, for "Open in Colab". */
    __FURINA_NOTEBOOKS_GITHUB__?: string;
  }
}

/**
 * Where the API lives, resolved in the order the deployment needs:
 *
 * 1. `window.__FURINA_API_BASE_URL__` -- written into `/config.js` when the
 *    nginx container boots, so the built bundle can be pointed at a different
 *    backend without being rebuilt.
 * 2. `VITE_API_BASE_URL` -- baked in at build time, for a split deployment.
 * 3. `/api` on the same origin -- what the container defaults to (nginx proxies
 *    it to Django) and what the Vite dev proxy forwards to `localhost:8000`.
 */
function resolveApiBaseUrl(): string {
  const runtime =
    typeof window !== "undefined" ? window.__FURINA_API_BASE_URL__ : undefined;
  const buildTime = import.meta.env.VITE_API_BASE_URL as string | undefined;
  return runtime?.trim() || buildTime?.trim() || "/api";
}

export const API_BASE_URL: string = resolveApiBaseUrl();

/**
 * Where the course notebooks are published on GitHub, as
 * `owner/repo/blob/branch/notebooks`, so a lesson can offer "Open in Colab".
 * Empty (the default) hides that button: Colab can only open notebooks from a
 * public repository that contains them.
 */
export const NOTEBOOKS_GITHUB_PATH: string = (
  (typeof window !== "undefined" ? window.__FURINA_NOTEBOOKS_GITHUB__ : undefined)?.trim() ||
  (import.meta.env.VITE_NOTEBOOKS_GITHUB as string | undefined)?.trim() ||
  ""
).replace(/^\/+|\/+$/g, "");

const client: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  // Training a large model can legitimately take a while.
  timeout: 300000,
  headers: { Accept: "application/json" },
  // Django's CSRF protection. Once someone is signed in, every POST and
  // DELETE has to echo the csrftoken cookie (set by `session()`) back in this
  // header; axios does that for same-origin requests.
  xsrfCookieName: "csrftoken",
  xsrfHeaderName: "X-CSRFToken",
  withCredentials: true,
});

/** Turn any axios failure into the flat `{ detail, errors, status }` we render. */
export function toApiError(error: unknown): ApiError {
  // `request()` below already rejects with a normalized ApiError, and callers
  // (WorkspaceProvider's restore) legitimately run a caught error back through
  // here. Without this guard such an object matches no branch and falls to
  // `String(error)` -> the literal "[object Object]", while `status` is lost so
  // `status === 404` could never clear a stale saved id. An AxiosError carries
  // `message`, never a string `detail`, so this cannot swallow a real failure.
  if (
    typeof error === "object" &&
    error !== null &&
    typeof (error as { detail?: unknown }).detail === "string"
  ) {
    return error as ApiError;
  }

  const axiosError = error as AxiosError<Record<string, unknown>>;

  if (axiosError?.isAxiosError) {
    const status = axiosError.response?.status;
    const data = axiosError.response?.data;

    if (data && typeof data === "object") {
      const detail = data.detail;
      if (typeof detail === "string") {
        return {
          detail,
          errors: data.errors as Record<string, string[]> | undefined,
          status,
        };
      }
      // DRF's default ValidationError shape: { field: ["message", ...] }
      const fieldErrors: Record<string, string[]> = {};
      for (const [key, value] of Object.entries(data)) {
        fieldErrors[key] = Array.isArray(value)
          ? value.map(String)
          : [String(value)];
      }
      const first = Object.entries(fieldErrors)[0];
      if (first) {
        return {
          detail: `${first[0]}: ${first[1][0]}`,
          errors: fieldErrors,
          status,
        };
      }
    }

    if (axiosError.code === "ECONNABORTED") {
      return { detail: "The request timed out. Try a smaller dataset or a simpler model.", status };
    }
    if (!axiosError.response) {
      return {
        detail: `Cannot reach the API at ${API_BASE_URL}. Is the Django server running?`,
      };
    }
    return { detail: axiosError.message, status };
  }

  return { detail: error instanceof Error ? error.message : String(error) };
}

async function request<T>(fn: () => Promise<{ data: T }>): Promise<T> {
  try {
    return (await fn()).data;
  } catch (error) {
    throw toApiError(error);
  }
}

const services = {
  health(): Promise<HealthResponse> {
    return request(() => client.get<HealthResponse>("/health/"));
  },

  /* ---------------- datasets ---------------- */

  uploadDataset(
    file: File,
    options: { name?: string; delimiter?: string; onProgress?: (pct: number) => void } = {}
  ): Promise<DatasetDetail> {
    const form = new FormData();
    form.append("file", file);
    if (options.name) form.append("name", options.name);
    if (options.delimiter) form.append("delimiter", options.delimiter);

    return request(() =>
      client.post<DatasetDetail>("/datasets/upload/", form, {
        onUploadProgress: (event) => {
          if (options.onProgress && event.total) {
            options.onProgress(Math.round((event.loaded / event.total) * 100));
          }
        },
      })
    );
  },

  listDatasets(): Promise<DatasetSummary[]> {
    return request(() => client.get<DatasetSummary[]>("/datasets/"));
  },

  getDataset(id: string, previewRows = 10): Promise<DatasetDetail> {
    return request(() =>
      client.get<DatasetDetail>(`/datasets/${id}/`, {
        params: { preview_rows: previewRows },
      })
    );
  },

  deleteDataset(id: string): Promise<void> {
    return request(() => client.delete<void>(`/datasets/${id}/`));
  },

  datasetDownloadUrl(id: string): string {
    return `${API_BASE_URL}/datasets/${id}/download/`;
  },

  preprocess(id: string, body: PreprocessRequest): Promise<PreprocessResponse> {
    return request(() =>
      client.post<PreprocessResponse>(`/datasets/${id}/preprocess/`, body)
    );
  },

  /* ---------------- algorithms & training ---------------- */

  listAlgorithms(task?: TaskType): Promise<AlgorithmSpec[]> {
    return request(() =>
      client.get<AlgorithmSpec[]>("/algorithms/", {
        params: task ? { task } : undefined,
      })
    );
  },

  train(body: TrainRequest): Promise<TrainingRunDetail> {
    return request(() => client.post<TrainingRunDetail>("/train/", body));
  },

  listRuns(datasetId?: string): Promise<TrainingRunSummary[]> {
    return request(() =>
      client.get<TrainingRunSummary[]>("/runs/", {
        params: datasetId ? { dataset_id: datasetId } : undefined,
      })
    );
  },

  getRun(id: string): Promise<TrainingRunDetail> {
    return request(() => client.get<TrainingRunDetail>(`/runs/${id}/`));
  },

  deleteRun(id: string): Promise<void> {
    return request(() => client.delete<void>(`/runs/${id}/`));
  },

  predict(id: string, rows: Record<string, unknown>[]): Promise<PredictResponse> {
    return request(() =>
      client.post<PredictResponse>(`/runs/${id}/predict/`, { rows })
    );
  },

  modelDownloadUrl(id: string): string {
    return `${API_BASE_URL}/runs/${id}/download_model/`;
  },

  reportDownloadUrl(id: string, format: "md" | "json" = "md"): string {
    return `${API_BASE_URL}/runs/${id}/report/?format=${format}`;
  },

  /** Whether this server has an OpenRouter key, so the UI can explain itself. */
  assistantStatus(): Promise<AssistantStatus> {
    return request(() => client.get<AssistantStatus>("/assistant/"));
  },

  /**
   * Ask the in-app helper. The key lives on the server, so this never carries
   * one; `context` sends ids, and the server decides what about them the model
   * is told.
   */
  askAssistant(
    messages: AssistantMessage[],
    context: AssistantContext
  ): Promise<AssistantReply> {
    return request(() =>
      client.post<AssistantReply>("/assistant/", { messages, context })
    );
  },

  /* ---------------- accounts ---------------- */

  /** Who is signed in. Also sets the CSRF cookie the other calls need. */
  session(): Promise<SessionInfo> {
    return request(() => client.get<SessionInfo>("/auth/session/"));
  },

  login(credentials: Credentials): Promise<SessionInfo> {
    return request(() => client.post<SessionInfo>("/auth/login/", credentials));
  },

  register(registration: Registration): Promise<SessionInfo> {
    return request(() => client.post<SessionInfo>("/auth/register/", registration));
  },

  logout(): Promise<SessionInfo> {
    return request(() => client.post<SessionInfo>("/auth/logout/"));
  },

  /* ---------------- course progress (signed in) ---------------- */

  progress(): Promise<ProgressResponse> {
    return request(() => client.get<ProgressResponse>("/progress/"));
  },

  /** Mark lessons complete; keys already saved are left alone. */
  saveProgress(lessons: string[]): Promise<ProgressResponse> {
    return request(() => client.post<ProgressResponse>("/progress/", { lessons }));
  },

  /* ---------------- notebooks ---------------- */

  /** The URL that opens a lesson notebook in the student's own JupyterHub workspace. */
  launchNotebook(path: string): Promise<NotebookLaunch> {
    return request(() => client.post<NotebookLaunch>("/notebooks/launch/", { path }));
  },

  /* ---------------- grading (signed in) ---------------- */

  /** Hand in a lesson notebook (the learner's .ipynb) for automated grading. */
  submitNotebook(path: string, file: File): Promise<Submission> {
    const form = new FormData();
    form.append("path", path);
    form.append("notebook", file);
    return request(() => client.post<Submission>("/submissions/", form));
  },

  /** The signed-in learner's latest submissions of one lesson notebook, newest first. */
  submissions(path: string): Promise<Submission[]> {
    return request(() =>
      client.get<{ results: Submission[] }>("/submissions/", { params: { path } })
    ).then((page) => page.results);
  },

  submission(id: string): Promise<Submission> {
    return request(() => client.get<Submission>(`/submissions/${encodeURIComponent(id)}/`));
  },

  /* ---------------- lifecycle: dataset versions ---------------- */

  listVersions(datasetId: string): Promise<DatasetVersion[]> {
    return request(() =>
      client.get<DatasetVersion[]>(`/lifecycle/datasets/${datasetId}/versions/`)
    );
  },

  createVersion(
    datasetId: string,
    body: {
      name?: string;
      description?: string;
      source_type?: string;
      parent_version_id?: string | null;
      recipe?: unknown;
    } = {}
  ): Promise<DatasetVersion> {
    return request(() =>
      client.post<DatasetVersion>(`/lifecycle/datasets/${datasetId}/versions/`, body)
    );
  },

  versionLineage(versionId: string): Promise<DatasetVersionLineage> {
    return request(() =>
      client.get<DatasetVersionLineage>(`/lifecycle/versions/${versionId}/lineage/`)
    );
  },

  versionDownloadUrl(versionId: string): string {
    return `${API_BASE_URL}/lifecycle/versions/${versionId}/download/`;
  },

  checkLeakage(
    versionId: string,
    targetColumn: string,
    featureColumns?: string[] | null
  ): Promise<LeakageSummary> {
    return request(() =>
      client.post<LeakageSummary>(`/lifecycle/versions/${versionId}/leakage/`, {
        target_column: targetColumn,
        feature_columns: featureColumns ?? null,
      })
    );
  },

  estimateResources(
    versionId: string,
    algorithm: string,
    nTrials = 1
  ): Promise<ResourceEstimate> {
    return request(() =>
      client.post<ResourceEstimate>(`/lifecycle/versions/${versionId}/resources/`, {
        algorithm,
        n_trials: nTrials,
      })
    );
  },

  /* ---------------- lifecycle: experiments ---------------- */

  listExperiments(datasetId?: string): Promise<Experiment[]> {
    return request(() =>
      client.get<Experiment[]>("/lifecycle/experiments/", {
        params: datasetId ? { dataset_id: datasetId } : undefined,
      })
    );
  },

  createExperiment(body: CreateExperimentRequest): Promise<ExperimentDetail> {
    return request(() =>
      client.post<ExperimentDetail>("/lifecycle/experiments/", body)
    );
  },

  getExperiment(id: string): Promise<ExperimentDetail> {
    return request(() => client.get<ExperimentDetail>(`/lifecycle/experiments/${id}/`));
  },

  deleteExperiment(id: string): Promise<void> {
    return request(() => client.delete<void>(`/lifecycle/experiments/${id}/`));
  },

  experimentRuns(id: string): Promise<RunSummary[]> {
    return request(() =>
      client.get<RunSummary[]>(`/lifecycle/experiments/${id}/runs/`)
    );
  },

  trainExperiment(id: string, body: TrainJobRequest): Promise<Job> {
    return request(() =>
      client.post<Job>(`/lifecycle/experiments/${id}/train/`, body)
    );
  },

  searchHyperparameters(id: string, body: HpoJobRequest): Promise<Job> {
    return request(() =>
      client.post<Job>(`/lifecycle/experiments/${id}/search/`, body)
    );
  },

  runAutoMl(id: string, body: AutoMlJobRequest): Promise<Job> {
    return request(() =>
      client.post<Job>(`/lifecycle/experiments/${id}/automl/`, body)
    );
  },

  /* ---------------- lifecycle: runs ---------------- */

  getLifecycleRun(id: string): Promise<RunDetail> {
    return request(() => client.get<RunDetail>(`/lifecycle/runs/${id}/`));
  },

  compareRuns(runIds: string[]): Promise<RunComparison> {
    return request(() =>
      client.post<RunComparison>("/lifecycle/runs/compare/", { run_ids: runIds })
    );
  },

  lifecycleModelDownloadUrl(runId: string): string {
    return `${API_BASE_URL}/lifecycle/runs/${runId}/download_model/`;
  },

  lifecycleNotebookUrl(runId: string): string {
    return `${API_BASE_URL}/lifecycle/runs/${runId}/notebook/`;
  },

  registerModel(runId: string, modelName?: string): Promise<ModelVersion> {
    return request(() =>
      client.post<ModelVersion>(`/lifecycle/runs/${runId}/register/`, {
        model_name: modelName ?? "",
      })
    );
  },

  reproduceRun(runId: string, runNow = false): Promise<Job> {
    return request(() =>
      client.post<Job>(`/lifecycle/runs/${runId}/reproduce/`, { run_now: runNow })
    );
  },

  /* ---------------- lifecycle: jobs ---------------- */

  getJob(id: string): Promise<Job> {
    return request(() => client.get<Job>(`/lifecycle/jobs/${id}/`));
  },

  cancelJob(id: string): Promise<Job> {
    return request(() => client.post<Job>(`/lifecycle/jobs/${id}/cancel/`, {}));
  },

  /* ---------------- lifecycle: registry ---------------- */

  listModels(): Promise<ModelSummary[]> {
    return request(() => client.get<ModelSummary[]>("/lifecycle/models/"));
  },

  getModel(id: string): Promise<ModelDetail> {
    return request(() => client.get<ModelDetail>(`/lifecycle/models/${id}/`));
  },

  getModelVersion(modelId: string, version: number): Promise<ModelVersion> {
    return request(() =>
      client.get<ModelVersion>(`/lifecycle/models/${modelId}/versions/${version}/`)
    );
  },

  promoteModelVersion(
    modelId: string,
    version: number,
    target: ModelState,
    confirmed = false
  ): Promise<ModelVersion> {
    return request(() =>
      client.post<ModelVersion>(
        `/lifecycle/models/${modelId}/versions/${version}/promote/`,
        { target, confirmed }
      )
    );
  },

  deployModelVersion(
    modelId: string,
    version: number,
    slug?: string
  ): Promise<Deployment> {
    return request(() =>
      client.post<Deployment>(
        `/lifecycle/models/${modelId}/versions/${version}/deploy/`,
        { slug: slug ?? "" }
      )
    );
  },

  /* ---------------- lifecycle: deployments ---------------- */

  listDeployments(): Promise<Deployment[]> {
    return request(() => client.get<Deployment[]>("/lifecycle/deployments/"));
  },

  getDeployment(slug: string): Promise<Deployment> {
    return request(() => client.get<Deployment>(`/lifecycle/deployments/${slug}/`));
  },

  archiveDeployment(slug: string): Promise<void> {
    return request(() => client.delete<void>(`/lifecycle/deployments/${slug}/`));
  },

  predictDeployment(
    slug: string,
    data: Record<string, unknown>
  ): Promise<DeploymentPrediction> {
    return request(() =>
      client.post<DeploymentPrediction>(`/lifecycle/deployments/${slug}/predict/`, { data })
    );
  },

  batchPredictDeployment(slug: string, file: File): Promise<Blob> {
    const form = new FormData();
    form.append("file", file);
    return request(() =>
      client.post<Blob>(`/lifecycle/deployments/${slug}/batch-predict/`, form, {
        responseType: "blob",
      })
    );
  },

  deploymentSnippets(slug: string): Promise<ClientSnippets> {
    return request(() =>
      client.get<ClientSnippets>(`/lifecycle/deployments/${slug}/snippets/`)
    );
  },

  deploymentOpenApiUrl(slug: string): string {
    return `${API_BASE_URL}/lifecycle/deployments/${slug}/openapi/`;
  },

  /* ---------------- lifecycle: monitoring ---------------- */

  monitoringOverview(): Promise<MonitoringSummary[]> {
    return request(() => client.get<MonitoringSummary[]>("/lifecycle/monitoring/"));
  },

  deploymentMonitoring(slug: string): Promise<{
    summary: MonitoringSummary;
    records: { kind: string; payload: Record<string, unknown>; created_at: string }[];
  }> {
    return request(() =>
      client.get(`/lifecycle/deployments/${slug}/monitoring/`)
    );
  },

  checkDrift(slug: string, rows: Record<string, unknown>[]): Promise<DriftReport> {
    return request(() =>
      client.post<DriftReport>(`/lifecycle/deployments/${slug}/drift-check/`, { rows })
    );
  },
};

export default services;
