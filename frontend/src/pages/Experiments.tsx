/**
 * Experiments: create one, train runs inside it, and see the best result.
 *
 * An experiment is the lifecycle's central object — one dataset version, one
 * objective, many runs. This page is the entry point: pick a dataset, name an
 * experiment, choose the objective, then train models against it. Every
 * training call is a job, so the page polls and shows progress rather than
 * blocking on the request.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import services, { toApiError } from "../services/services";
import type {
  Experiment,
  ExperimentDetail,
  RunSummary,
} from "../interface/LifecycleTypes";
import { useWorkspace } from "../context/workspaceContext";
import { useJobPolling } from "../hooks/useJobPolling";
import { JobProgress } from "../components/lifecycle/LifecyclePanels";
import { usePageSeo } from "../hooks/usePageSeo";

const TOOL_TRAINING = "Data Training";

function formatWhen(iso: string): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "—";
  return at.toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatMetric(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : value.toFixed(4);
}

export default function Experiments() {
  const { dataset, run, goToTool } = useWorkspace();
  const [experiments, setExperiments] = useState<Experiment[] | null>(null);
  const [selected, setSelected] = useState<ExperimentDetail | null>(null);
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [objective, setObjective] = useState("");
  const polling = useJobPolling();

  usePageSeo({
    title: "Experiments",
    description:
      "Create reproducible machine-learning experiments: versioned datasets, tracked runs, comparable metrics, and a best model per objective.",
    path: "/app/experiments",
  });

  const loadExperiments = useCallback(async () => {
    try {
      const list = await services.listExperiments();
      setExperiments(list);
      setError(null);
    } catch (caught) {
      setError(toApiError(caught).detail);
      setExperiments([]);
    }
  }, []);

  useEffect(() => {
    void loadExperiments();
  }, [loadExperiments]);

  const loadRuns = useCallback(async (experimentId: string) => {
    try {
      const list = await services.experimentRuns(experimentId);
      setRuns(list);
    } catch (caught) {
      setError(toApiError(caught).detail);
    }
  }, []);

  const openExperiment = useCallback(
    async (id: string) => {
      try {
        const detail = await services.getExperiment(id);
        setSelected(detail);
        setObjective(detail.objective_metric);
        await loadRuns(id);
      } catch (caught) {
        setError(toApiError(caught).detail);
      }
    },
    [loadRuns]
  );

  const handleCreate = async () => {
    if (!dataset) {
      setError("Upload a dataset first — an experiment trains against one.");
      return;
    }
    setCreating(true);
    setError(null);
    try {
      const created = await services.createExperiment({
        dataset_id: dataset.id,
        name: name.trim() || `${dataset.name} experiment`,
        task_type: "auto",
        objective_metric: objective || undefined,
      });
      setName("");
      await loadExperiments();
      await openExperiment(created.id);
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setCreating(false);
    }
  };

  // When a job finishes, refresh the run list and the best-run pointer.
  useEffect(() => {
    if (!selected) return;
    const status = polling.job?.status;
    if (status === "completed" || status === "failed") {
      void loadRuns(selected.id);
      void services
        .getExperiment(selected.id)
        .then(setSelected)
        .catch(() => undefined);
    }
  }, [polling.job?.status, selected, loadRuns]);

  const trainWithWorkspaceRun = async () => {
    if (!selected || !run) return;
    try {
      const job = await services.trainExperiment(selected.id, {
        target_column: run.target_column,
        feature_columns: run.feature_columns,
        algorithm: run.algorithm,
        task_type: run.task_type,
        test_size: run.test_size,
        random_state: run.random_state,
        hyperparameters: run.hyperparameters,
        cv_folds: run.cv_folds,
      });
      polling.watch(job.id);
    } catch (caught) {
      setError(toApiError(caught).detail);
    }
  };

  const bestRun = useMemo(
    () => runs?.find((item) => item.id === selected?.best_run_id) ?? null,
    [runs, selected?.best_run_id]
  );

  return (
    <div className="p-4 md:p-6 space-y-6">
      <header>
        <h1 className="text-heading">Experiments</h1>
        <p className="text-ink-muted mt-1 max-w-3xl">
          An experiment pins one dataset version and one objective, and holds every run
          trained against them — so "which model is best?" is a question the platform can
          answer, and every result can be traced back to its data.
        </p>
      </header>

      {error && (
        <div className="border-l-2 border-state-bad pl-3 text-small" role="alert">
          {error}
        </div>
      )}

      <section className="border border-rule p-4 space-y-3">
        <h2 className="text-subheading">New experiment</h2>
        <p className="text-small text-ink-muted">
          {dataset
            ? `Training against "${dataset.name}" (${dataset.n_rows} rows × ${dataset.n_cols} columns).`
            : "No dataset loaded. Upload one on the Preprocessing page first."}
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-small text-ink-muted">Name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder={dataset ? `${dataset.name} experiment` : "Experiment name"}
              className="border border-rule px-2 py-1.5 bg-paper min-w-[16rem]"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-small text-ink-muted">Objective metric</span>
            <select
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              className="border border-rule px-2 py-1.5 bg-paper min-w-[12rem]"
            >
              <option value="">Choose automatically</option>
              <option value="accuracy">Accuracy</option>
              <option value="f1">F1 (weighted)</option>
              <option value="roc_auc">ROC AUC</option>
              <option value="r2">R²</option>
              <option value="mae">MAE</option>
            </select>
          </label>
          <button
            type="button"
            onClick={handleCreate}
            disabled={creating || !dataset}
            className="border border-primary px-3 py-1.5 hover:bg-paper-raised disabled:opacity-50"
          >
            {creating ? "Creating…" : "Create experiment"}
          </button>
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="text-subheading">Your experiments</h2>
        {experiments === null ? (
          <p className="text-small text-ink-muted" role="status">Loading…</p>
        ) : experiments.length === 0 ? (
          <p className="text-small text-ink-muted">No experiments yet.</p>
        ) : (
          <ul className="divide-y divide-rule border border-rule">
            {experiments.map((experiment) => (
              <li key={experiment.id}>
                <button
                  type="button"
                  onClick={() => void openExperiment(experiment.id)}
                  className={`w-full text-left px-3 py-2 hover:bg-paper-raised ${
                    selected?.id === experiment.id ? "bg-paper-raised" : ""
                  }`}
                >
                  <span className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <span className="font-semibold">{experiment.name}</span>
                    <span className="text-small text-ink-muted">
                      {experiment.task_type} · maximize {experiment.objective_metric}
                    </span>
                    <span className="text-small text-ink-muted">
                      {experiment.n_runs} run{experiment.n_runs === 1 ? "" : "s"}
                    </span>
                    {experiment.best_metric_value !== null && (
                      <span className="text-small">
                        best {formatMetric(experiment.best_metric_value)}
                      </span>
                    )}
                    <span className="text-small text-ink-muted ml-auto">
                      {formatWhen(experiment.created_at)}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {selected && (
        <section className="space-y-4 border-t border-rule pt-4">
          <div>
            <h2 className="text-subheading">{selected.name}</h2>
            <p className="text-small text-ink-muted">
              Dataset version {selected.dataset_version.version} · {selected.dataset_name} ·{" "}
              {selected.dataset_version.row_count.toLocaleString()} rows · objective{" "}
              {selected.objective_direction} {selected.objective_metric}
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={trainWithWorkspaceRun}
              disabled={!run || polling.polling}
              className="border border-primary px-3 py-1.5 hover:bg-paper-raised disabled:opacity-50"
            >
              Train current configuration as a run
            </button>
            <button
              type="button"
              onClick={() => goToTool(TOOL_TRAINING)}
              className="border border-rule px-3 py-1.5 hover:bg-paper-sunken"
            >
              Configure a model
            </button>
          </div>
          {!run && (
            <p className="text-small text-ink-muted">
              Train a model on the Data Training page first, then add it here as a run.
            </p>
          )}

          <JobProgress job={polling.job} onCancel={() => void polling.cancel()} />
          {polling.error && (
            <p className="text-small border-l-2 border-state-bad pl-3">{polling.error}</p>
          )}

          {bestRun && (
            <p className="text-small">
              <strong>Best run:</strong> {bestRun.algorithm_label || bestRun.algorithm} —{" "}
              {bestRun.objective_metric} {formatMetric(bestRun.primary_metric)}
            </p>
          )}

          <div>
            <h3 className="text-small font-semibold mb-2">Runs</h3>
            {runs === null ? (
              <p className="text-small text-ink-muted" role="status">Loading runs…</p>
            ) : runs.length === 0 ? (
              <p className="text-small text-ink-muted">No runs yet.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-small border-collapse">
                  <thead>
                    <tr className="text-left border-b border-rule-strong">
                      <th className="py-1 pr-3 font-semibold">#</th>
                      <th className="py-1 pr-3 font-semibold">Algorithm</th>
                      <th className="py-1 pr-3 font-semibold">Status</th>
                      <th className="py-1 pr-3 font-semibold text-right">{runs[0].objective_metric}</th>
                      <th className="py-1 pr-3 font-semibold text-right">Train time</th>
                      <th className="py-1 font-semibold">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((item) => (
                      <tr key={item.id} className="border-b border-rule">
                        <td className="py-1 pr-3">{item.trial_index !== null ? `t${item.trial_index + 1}` : item.number}</td>
                        <td className="py-1 pr-3">{item.algorithm_label || item.algorithm}</td>
                        <td className="py-1 pr-3">{item.status}</td>
                        <td className="py-1 pr-3 text-right">{formatMetric(item.primary_metric)}</td>
                        <td className="py-1 pr-3 text-right">{(item.duration_ms / 1000).toFixed(2)}s</td>
                        <td className="py-1">{formatWhen(item.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
