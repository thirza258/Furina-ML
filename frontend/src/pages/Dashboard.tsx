/**
 * Every training run this instance has done, ranked and compared.
 *
 * The evaluation page answers "how did this model do?". This one answers the
 * question you can only ask once you have run a few: "which of these is
 * best, and did the last change help?"
 *
 * Two rules shape the whole page:
 *
 *  - Metrics are only comparable within one task type. Accuracy and R² are
 *    different quantities on different scales, so the chart refuses to draw
 *    them on one axis and says why, rather than producing a plausible picture
 *    that means nothing.
 *  - Direction matters. Log loss, MAE and RMSE are better when smaller, so
 *    "best" and the sort order read `higherIsBetter` from the metric catalogue
 *    instead of assuming bigger wins.
 */
import { useEffect, useMemo, useState } from "react";

import services, { toApiError } from "../services/services";
import type { TaskType, TrainingRunSummary } from "../interface/MlTypes";
import {
  CLASSIFICATION_METRICS,
  REGRESSION_METRICS,
  formatCompact,
  readMetric,
  type MetricDef,
} from "../data/metricCatalogue";
import MetricBars from "../components/MetricBars";
import { useWorkspace } from "../context/workspaceContext";
import { WARNING_MARK } from "../explainer/marks";
import { usePageSeo } from "../hooks/usePageSeo";

const TOOL_EVALUATION = "Model Evaluation";
const TOOL_TRAINING = "Data Training";

/** How many runs the comparison chart draws before it starts truncating. */
const MAX_CHART_ROWS = 12;

type TaskFilter = "all" | TaskType;
type SortKey = "metric" | "created_at" | "algorithm" | "duration";

function metricsFor(task: TaskFilter): MetricDef[] {
  if (task === "classification") return CLASSIFICATION_METRICS;
  if (task === "regression") return REGRESSION_METRICS;
  return [];
}

/** `2026-08-31T12:04:00Z` -> `31 Aug 2026, 12:04`. */
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

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

/** A run's label for a chart axis or a table row. */
function runLabel(run: TrainingRunSummary): string {
  return run.name?.trim() || run.algorithm_label || run.algorithm || "Untitled run";
}

export default function Dashboard() {
  const { goToTool, setRun } = useWorkspace();

  usePageSeo({
    title: "Runs Comparison Dashboard",
    description:
      "Compare and rank every training run across classification and regression metrics, fit durations, and holdout test scores.",
    path: "/app/dashboard",
  });

  const [runs, setRuns] = useState<TrainingRunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [taskFilter, setTaskFilter] = useState<TaskFilter>("all");
  const [datasetFilter, setDatasetFilter] = useState<string>("all");
  const [metricKey, setMetricKey] = useState<string>("");
  const [showTrain, setShowTrain] = useState<boolean>(false);
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [openingId, setOpeningId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const list = await services.listRuns();
        if (!cancelled) {
          setRuns(list);
          setError(null);
          // Open on whatever the newest run was working on, so the page is
          // useful without touching a single control.
          const newest = list.find((run) => run.status === "success");
          if (newest) setTaskFilter(newest.task_type);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(toApiError(caught).detail);
          setRuns([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const availableMetrics = useMemo(() => metricsFor(taskFilter), [taskFilter]);

  // Keep the metric selection legal whenever the task changes underneath it.
  useEffect(() => {
    if (availableMetrics.length === 0) {
      setMetricKey("");
      return;
    }
    if (!availableMetrics.some((def) => def.key === metricKey)) {
      setMetricKey(availableMetrics[0].key);
    }
  }, [availableMetrics, metricKey]);

  const datasets = useMemo(() => {
    const seen = new Map<string, string>();
    for (const run of runs ?? []) seen.set(run.dataset_id, run.dataset_name);
    return Array.from(seen, ([id, name]) => ({ id, name }));
  }, [runs]);

  const filtered = useMemo(() => {
    return (runs ?? []).filter((run) => {
      if (taskFilter !== "all" && run.task_type !== taskFilter) return false;
      if (datasetFilter !== "all" && run.dataset_id !== datasetFilter) return false;
      return true;
    });
  }, [runs, taskFilter, datasetFilter]);

  const activeMetric = availableMetrics.find((def) => def.key === metricKey) ?? null;

  const sorted = useMemo(() => {
    const rows = [...filtered];
    rows.sort((a, b) => {
      let compare = 0;
      if (sortKey === "created_at") {
        compare = a.created_at.localeCompare(b.created_at);
      } else if (sortKey === "algorithm") {
        compare = runLabel(a).localeCompare(runLabel(b));
      } else if (sortKey === "duration") {
        compare = (a.duration_ms || 0) - (b.duration_ms || 0);
      } else {
        // Nulls always sink, whichever way the column is pointing: a run with
        // no value for this metric is not "the best" one.
        const left = activeMetric ? readMetric(a.metrics, activeMetric.key) : a.primary_metric;
        const right = activeMetric ? readMetric(b.metrics, activeMetric.key) : b.primary_metric;
        if (left === null && right === null) compare = 0;
        else if (left === null) return 1;
        else if (right === null) return -1;
        else compare = left - right;
      }
      return sortAsc ? compare : -compare;
    });
    return rows;
  }, [filtered, sortKey, sortAsc, activeMetric]);

  /**
   * The leaderboard order: best first, honouring the metric's own direction.
   * This is separate from the table's sort so the chart always ranks, however
   * the table happens to be arranged.
   */
  const ranked = useMemo(() => {
    if (!activeMetric) return [];
    const scored = filtered
      .filter((run) => run.status === "success")
      .map((run) => ({ run, value: readMetric(run.metrics, activeMetric.key) }))
      .filter((row): row is { run: TrainingRunSummary; value: number } => row.value !== null);
    scored.sort((a, b) =>
      activeMetric.higherIsBetter ? b.value - a.value : a.value - b.value
    );
    return scored;
  }, [filtered, activeMetric]);

  const chartRows = ranked.slice(0, MAX_CHART_ROWS);
  const hiddenFromChart = ranked.length - chartRows.length;
  const best = ranked[0] ?? null;

  const successCount = filtered.filter((run) => run.status === "success").length;
  const failedCount = filtered.length - successCount;
  const medianDuration = useMemo(() => {
    const times = filtered
      .map((run) => run.duration_ms)
      .filter((ms) => Number.isFinite(ms) && ms > 0)
      .sort((a, b) => a - b);
    if (times.length === 0) return 0;
    return times[Math.floor(times.length / 2)];
  }, [filtered]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc((previous) => !previous);
    } else {
      setSortKey(key);
      // Dates and scores both read most-useful-first descending.
      setSortAsc(key === "algorithm");
    }
  };

  /**
   * The workspace holds a full run detail, not an id, so opening a row from
   * here means fetching the detail first. If that fails the user stays put
   * with a reason rather than landing on an evaluation page for a run that
   * never loaded.
   */
  const openRun = (run: TrainingRunSummary) => {
    void (async () => {
      setOpeningId(run.id);
      try {
        const detail = await services.getRun(run.id);
        setRun(detail);
        goToTool(TOOL_EVALUATION);
      } catch (caught) {
        setError(toApiError(caught).detail);
      } finally {
        setOpeningId(null);
      }
    })();
  };

  // ---------------------------------------------------------------- states
  if (runs === null) {
    return (
      <div className="bg-paper min-h-full p-8">
        <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">
          Dashboard
        </h1>
        <p className="text-ink-muted">Loading your training runs…</p>
      </div>
    );
  }

  if (error !== null) {
    return (
      <div className="bg-paper min-h-full p-8">
        <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">
          Dashboard
        </h1>
        <div className="callout border-state-bad">
          <p className="font-medium text-state-bad">
            {WARNING_MARK} The run history could not be loaded — {error}
          </p>
        </div>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="bg-paper min-h-full p-8">
        <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">
          Dashboard
        </h1>
        <div className="panel">
          <h2 className="panel-title">Nothing trained yet</h2>
          <p className="prose-page mb-6">
            This page ranks and compares the models you have trained. It fills in
            on its own once there is a run to show — train one and come back.
          </p>
          <button
            type="button"
            onClick={() => goToTool(TOOL_TRAINING)}
            className="btn btn-primary"
          >
            Train a model
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-paper min-h-full p-8">
      <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">
        Dashboard
      </h1>
      <p className="font-serif text-body text-ink-muted max-w-measure mb-8">
        Every model trained on this instance, ranked by the metric you choose.
        Scores come from the held-out test split, which is the only number worth
        comparing between runs.
      </p>

      {/* ---------------------------------------------------------------- */}
      {/* Summary. Plain figures, not charts: they are unrelated quantities. */}
      {/* ---------------------------------------------------------------- */}
      <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4 border-b border-rule pb-6 mb-8">
        <StatTile label="Runs in view" value={String(filtered.length)}
          note={failedCount > 0 ? `${failedCount} failed` : "all completed"} />
        <StatTile
          label={activeMetric ? `Best ${activeMetric.label.toLowerCase()}` : "Best score"}
          value={best && activeMetric ? formatCompact(activeMetric, best.value) : "—"}
          note={best ? runLabel(best.run) : "no scored run in view"}
        />
        <StatTile label="Datasets" value={String(datasets.length)}
          note={datasets.length === 1 ? datasets[0].name : "used across these runs"} />
        <StatTile label="Median fit time" value={formatDuration(medianDuration)}
          note="wall clock, server side" />
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Filters — one row, above the chart they act on.                   */}
      {/* ---------------------------------------------------------------- */}
      <div className="callout mb-8 flex flex-wrap items-end gap-x-6 gap-y-4">
        <Field label="Task">
          <select
            value={taskFilter}
            onChange={(event) => setTaskFilter(event.target.value as TaskFilter)}
            className="border border-rule-strong rounded-control bg-paper-raised px-2 py-1.5 text-small text-ink"
          >
            <option value="all">All tasks</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </Field>

        <Field label="Dataset">
          <select
            value={datasetFilter}
            onChange={(event) => setDatasetFilter(event.target.value)}
            className="border border-rule-strong rounded-control bg-paper-raised px-2 py-1.5 text-small text-ink max-w-[16rem]"
          >
            <option value="all">All datasets</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Metric">
          <select
            value={metricKey}
            onChange={(event) => setMetricKey(event.target.value)}
            disabled={availableMetrics.length === 0}
            className="border border-rule-strong rounded-control bg-paper-raised px-2 py-1.5 text-small text-ink disabled:opacity-50"
          >
            {availableMetrics.length === 0 ? (
              <option value="">Pick a single task first</option>
            ) : (
              availableMetrics.map((def) => (
                <option key={def.key} value={def.key}>
                  {def.label}
                </option>
              ))
            )}
          </select>
        </Field>

        <label className="flex items-center gap-2 text-small text-ink cursor-pointer">
          <input
            type="checkbox"
            checked={showTrain}
            onChange={(event) => setShowTrain(event.target.checked)}
            disabled={!activeMetric}
            className="accent-primary"
          />
          Show training scores
        </label>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Comparison                                                        */}
      {/* ---------------------------------------------------------------- */}
      <section className="panel">
        <h2 className="panel-title">
          {activeMetric ? `Ranked by ${activeMetric.label.toLowerCase()}` : "Comparison"}
        </h2>

        {!activeMetric ? (
          <p className="prose-page text-ink-muted">
            Accuracy and R² are different quantities on different scales, so
            drawing them against one axis would invite a comparison that means
            nothing. Choose either classification or regression above and the
            ranking appears.
          </p>
        ) : chartRows.length === 0 ? (
          <p className="prose-page text-ink-muted">
            No run in this view reported {activeMetric.label.toLowerCase()}.{" "}
            {activeMetric.nullReason}
          </p>
        ) : (
          <MetricBars
            rows={chartRows}
            def={activeMetric}
            showTrain={showTrain}
            onOpen={openRun}
          />
        )}

        {hiddenFromChart > 0 && (
          <p className="mt-4 font-sans text-caption text-ink-muted">
            Showing the top {MAX_CHART_ROWS} of {ranked.length} scored runs.
            The remaining {hiddenFromChart} are in the table below.
          </p>
        )}
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* The table. Also the accessible reading of the chart above.        */}
      {/* ---------------------------------------------------------------- */}
      <section>
        <h2 className="font-serif text-h3 text-ink mb-3">All runs</h2>
        <div className="overflow-x-auto border border-rule">
          <table className="w-full border-collapse font-sans text-small">
            <caption className="sr-only">
              Every training run, with its test-split scores
            </caption>
            <thead>
              <tr className="bg-paper-sunken border-b border-rule">
                <SortHeader label="Run" active={sortKey === "algorithm"} asc={sortAsc}
                  onClick={() => toggleSort("algorithm")} />
                <Th>Dataset</Th>
                <Th>Target</Th>
                <SortHeader
                  label={activeMetric ? activeMetric.label : "Primary metric"}
                  align="right" active={sortKey === "metric"} asc={sortAsc}
                  onClick={() => toggleSort("metric")}
                />
                {showTrain && activeMetric && <Th align="right">On train</Th>}
                <SortHeader label="Fit time" align="right" active={sortKey === "duration"}
                  asc={sortAsc} onClick={() => toggleSort("duration")} />
                <SortHeader label="When" align="right" active={sortKey === "created_at"}
                  asc={sortAsc} onClick={() => toggleSort("created_at")} />
              </tr>
            </thead>
            <tbody>
              {sorted.map((run) => {
                const value = activeMetric
                  ? readMetric(run.metrics, activeMetric.key)
                  : run.primary_metric;
                const trainValue = activeMetric
                  ? readMetric(run.train_metrics, activeMetric.key)
                  : null;
                const isBest = best !== null && best.run.id === run.id;
                return (
                  <tr
                    key={run.id}
                    className="border-b border-rule last:border-b-0 hover:bg-paper-sunken transition-colors"
                  >
                    <td className="px-3 py-2.5 align-top">
                      <button
                        type="button"
                        onClick={() => openRun(run)}
                        disabled={openingId === run.id}
                        className="text-left text-primary hover:underline cursor-pointer disabled:opacity-60"
                      >
                        {runLabel(run)}
                        {openingId === run.id ? " — opening…" : ""}
                      </button>
                      <span className="block text-caption text-ink-muted">
                        {run.algorithm_label}
                        {isBest && activeMetric ? " · best in view" : ""}
                        {run.status === "failed" ? ` · ${WARNING_MARK} failed` : ""}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 align-top text-ink-muted">{run.dataset_name}</td>
                    <td className="px-3 py-2.5 align-top text-ink-muted font-mono text-caption">
                      {run.target_column}
                    </td>
                    <td className="px-3 py-2.5 align-top text-right font-mono tabular">
                      {activeMetric
                        ? formatCompact(activeMetric, value)
                        : value === null
                          ? "n/a"
                          : value.toFixed(3)}
                      {!activeMetric && (
                        <span className="block text-caption text-ink-muted font-sans">
                          {run.primary_metric_name}
                        </span>
                      )}
                    </td>
                    {showTrain && activeMetric && (
                      <td className="px-3 py-2.5 align-top text-right font-mono tabular text-ink-muted">
                        {formatCompact(activeMetric, trainValue)}
                      </td>
                    )}
                    <td className="px-3 py-2.5 align-top text-right tabular text-ink-muted">
                      {formatDuration(run.duration_ms)}
                    </td>
                    <td className="px-3 py-2.5 align-top text-right tabular text-ink-muted whitespace-nowrap">
                      {formatWhen(run.created_at)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {sorted.length === 0 && (
          <p className="mt-4 font-sans text-small text-ink-muted">
            No run matches these filters.
          </p>
        )}
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Pieces                                                              */
/* ------------------------------------------------------------------ */

function StatTile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div>
      <p className="field-label">{label}</p>
      <p className="mt-1 font-serif text-h2 text-ink tabular">{value}</p>
      <p className="mt-0.5 font-sans text-caption text-ink-muted truncate" title={note}>
        {note}
      </p>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

function Th({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }) {
  return (
    <th
      scope="col"
      className={`px-3 py-2 font-medium text-caption uppercase tracking-[0.06em] text-ink-muted ${
        align === "right" ? "text-right" : "text-left"
      }`}
    >
      {children}
    </th>
  );
}

function SortHeader({
  label, active, asc, onClick, align = "left",
}: {
  label: string; active: boolean; asc: boolean;
  onClick: () => void; align?: "left" | "right";
}) {
  return (
    <th
      scope="col"
      aria-sort={active ? (asc ? "ascending" : "descending") : "none"}
      className={`px-3 py-2 font-medium text-caption uppercase tracking-[0.06em] text-ink-muted ${
        align === "right" ? "text-right" : "text-left"
      }`}
    >
      <button
        type="button"
        onClick={onClick}
        className="inline-flex items-center gap-1 uppercase tracking-[0.06em] hover:text-ink cursor-pointer"
      >
        {label}
        <span aria-hidden="true" className={active ? "text-ink" : "text-rule-strong"}>
          {active ? (asc ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </th>
  );
}
