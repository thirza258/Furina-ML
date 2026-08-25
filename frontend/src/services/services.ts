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
  AlgorithmSpec,
  ApiError,
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

declare global {
  interface Window {
    /** Set by the container's generated `/config.js`; absent in `npm run dev`. */
    __FURINA_API_BASE_URL__?: string;
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

const client: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  // Training a large model can legitimately take a while.
  timeout: 300000,
  headers: { Accept: "application/json" },
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
};

export default services;
