/**
 * The top bar. Besides the product mark it answers the three questions a user
 * of this app asks most often: is the backend up, what data am I working on,
 * and did my model train?
 *
 * The backend check is the important one. Every page here is useless without
 * the Django server, and "nothing happens when I click" is a miserable way to
 * discover it is not running — so a failed health check is stated in words,
 * with the reason and the API base URL, in a bar nobody can miss.
 */
import React, { useCallback, useEffect, useState } from "react";

import services, { API_BASE_URL } from "../services/services";
import type { ApiError, HealthResponse } from "../interface/MlTypes";
import { useWorkspace } from "../context/workspaceContext";

interface NavBarProps {
  selectedTool: string;
}

type HealthState =
  | { status: "checking" }
  | { status: "online"; info: HealthResponse }
  | { status: "offline"; reason: string };

/**
 * Service methods already throw a normalized ApiError, so the value caught
 * here is that object — not an Error. Read `detail` off it directly; passing
 * it back through the normalizer would stringify it to "[object Object]".
 */
function errorDetail(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as ApiError).detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error
    ? error.message
    : "The backend could not be reached.";
}

/** Metrics are 0..1 ratios almost everywhere; 3dp is enough to compare runs. */
function formatMetric(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(3);
}

const NavBar: React.FC<NavBarProps> = ({ selectedTool }) => {
  const { dataset, run, restoring, restoreError } = useWorkspace();
  const [health, setHealth] = useState<HealthState>({ status: "checking" });
  const [rechecking, setRechecking] = useState<boolean>(false);

  const checkHealth = useCallback(async (): Promise<HealthState> => {
    try {
      const info = await services.health();
      return { status: "online", info };
    } catch (error) {
      return { status: "offline", reason: errorDetail(error) };
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const next = await checkHealth();
      if (!cancelled) setHealth(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [checkHealth]);

  const handleRecheck = async () => {
    setRechecking(true);
    setHealth({ status: "checking" });
    const next = await checkHealth();
    setHealth(next);
    setRechecking(false);
  };

  const datasetLabel = restoring
    ? "Restoring…"
    : dataset
      ? `${dataset.name} · ${dataset.n_rows.toLocaleString()} rows × ${dataset.n_cols} cols`
      : "No dataset";

  const runLabel = restoring
    ? "Restoring…"
    : run
      ? run.status === "failed"
        ? `${run.algorithm_label} · training failed`
        : `${run.algorithm_label} · ${run.primary_metric_name} ${formatMetric(run.primary_metric)}`
      : "No model trained";

  return (
    <header className="flex-shrink-0 z-10">
      <nav className="w-full min-h-16 bg-gray-800 text-white flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b-2 border-white">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold text-blue-300">Furina ML</h1>
          <span className="font-bold text-blue-400">{selectedTool}</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Workspace status. Light pills keep the status tokens on the
              background contrast they were validated against. */}
          <span
            className="flex items-center gap-2 bg-white rounded-md px-3 py-1 text-sm"
            title={
              dataset
                ? `Active dataset: ${dataset.name} (${dataset.source})`
                : "Upload a CSV on the Data Preprocessing page to get started."
            }
          >
            <span className="text-ink-muted text-xs uppercase tracking-wide">
              Dataset
            </span>
            <span
              className={`font-semibold ${dataset ? "text-ink" : "text-ink-muted"}`}
            >
              {datasetLabel}
            </span>
          </span>

          <span
            className="flex items-center gap-2 bg-white rounded-md px-3 py-1 text-sm"
            title={
              run
                ? `Run "${run.name || run.id}" on target "${run.target_column}"`
                : "Train a model on the Data Training page."
            }
          >
            <span className="text-ink-muted text-xs uppercase tracking-wide">
              Model
            </span>
            <span
              className={`font-semibold ${
                !run
                  ? "text-ink-muted"
                  : run.status === "failed"
                    ? "text-state-bad"
                    : "text-ink"
              }`}
            >
              {runLabel}
            </span>
          </span>

          {/* Backend connectivity. */}
          {health.status === "checking" && (
            <span className="flex items-center gap-2 bg-gray-100 rounded-md px-3 py-1 text-sm font-semibold text-ink-muted">
              <span aria-hidden="true">◌</span> Checking backend…
            </span>
          )}
          {health.status === "online" && (
            <span
              className="flex items-center gap-2 bg-white rounded-md px-3 py-1 text-sm font-semibold text-state-good"
              title={`scikit-learn ${health.info.sklearn} · pandas ${health.info.pandas} · numpy ${health.info.numpy} · ${API_BASE_URL}`}
            >
              <span aria-hidden="true">●</span> Backend connected
              <span className="font-normal text-ink-muted">
                scikit-learn {health.info.sklearn}
              </span>
            </span>
          )}
          {health.status === "offline" && (
            <span className="flex items-center gap-2 bg-white rounded-md px-3 py-1 text-sm font-semibold text-state-bad">
              <span aria-hidden="true">▲</span> Backend disconnected
            </span>
          )}
        </div>
      </nav>

      {/* Unmissable, because nothing in this app works without the server. */}
      {health.status === "offline" && (
        <div
          role="alert"
          className="w-full bg-state-bad text-white px-4 py-3 flex flex-wrap items-center justify-between gap-3"
        >
          <div className="text-sm">
            <p className="font-semibold">
              ▲ Backend disconnected — {health.reason}
            </p>
            <p className="opacity-90">
              Nothing can be uploaded, trained or evaluated until the Django
              server answers at{" "}
              <code className="bg-white/20 px-1 rounded">{API_BASE_URL}</code>.
              Start it with{" "}
              <code className="bg-white/20 px-1 rounded">
                python manage.py runserver
              </code>{" "}
              in the backend folder.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void handleRecheck()}
            disabled={rechecking}
            className="bg-white text-state-bad py-2 px-4 rounded-md text-sm font-semibold hover:bg-gray-200 transition-colors disabled:bg-white/60 disabled:cursor-not-allowed"
          >
            {rechecking ? "Checking…" : "Retry connection"}
          </button>
        </div>
      )}

      {restoreError !== null && (
        <div
          role="alert"
          className="w-full bg-state-warn text-white px-4 py-2 text-sm"
        >
          ▲ Could not restore your previous workspace — {restoreError}
        </div>
      )}
    </header>
  );
};

export default NavBar;
