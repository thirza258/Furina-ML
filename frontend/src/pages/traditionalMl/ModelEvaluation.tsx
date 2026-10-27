/**
 * Model Evaluation — the last step of the flow.
 *
 * Everything on this page comes from one real `TrainingRunDetail`: the run the
 * user just trained (handed over through the workspace) or one they pick from
 * the list of past runs, which is what makes the page useful after a reload.
 */
import { useCallback, useEffect, useState } from "react";

import ConfusionMatrixTable from "../../components/ConfusionMatrixTable";
import CourseLink from "../../components/CourseLink";
import FeatureImportanceChart from "../../components/FeatureImportanceChart";
import MetricsPanel from "../../components/MetricsPanel";
import PredictionForm from "../../components/PredictionForm";
import ResidualPlot from "../../components/ResidualPlot";
import { useWorkspace } from "../../context/workspaceContext";
import services from "../../services/services";
import { WARNING_MARK } from "../../explainer/marks";
import { usePageSeo } from "../../hooks/usePageSeo";
import type {
  TrainingRunDetail,
  TrainingRunSummary,
} from "../../interface/MlTypes";

/**
 * `services` throws the already-normalized `{ detail }` shape, so this narrows
 * it rather than re-running it through `toApiError`, which would stringify the
 * object into "[object Object]".
 */
function apiErrorDetail(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Something went wrong.";
}

function formatDuration(ms: number): string {
  if (ms >= 60000) {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.round((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms} ms`;
}

function formatTimestamp(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? iso : parsed.toLocaleString();
}

function formatScore(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  if (magnitude < 0.001) return value.toExponential(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function stringifyParam(value: unknown): string {
  if (value === null || value === undefined) return "default";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function runLabel(run: TrainingRunDetail | TrainingRunSummary): string {
  const trimmed = run.name.trim();
  if (trimmed !== "") return trimmed;
  return `${run.algorithm_label} on ${run.dataset_name}`;
}

interface ErrorBoxProps {
  title: string;
  message: string;
  onDismiss?: () => void;
}

function ErrorBox({ title, message, onDismiss }: ErrorBoxProps) {
  return (
    <div className="mb-6 flex items-start justify-between gap-4 rounded-control border border-state-bad/40 bg-state-bad/10 p-4">
      <p className="text-sm text-state-bad">
        <span className="font-semibold">✕ {title} </span>
        {message}
      </p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 text-sm font-semibold text-state-bad hover:underline"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}

interface FactProps {
  label: string;
  value: string;
  hint?: string;
}

function Fact({ label, value, hint }: FactProps) {
  return (
    <div className="bg-paper-sunken p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
        {label}
      </p>
      <p className="mt-1 break-words text-sm font-medium text-ink" title={value}>
        {value}
      </p>
      {hint && <p className="mt-0.5 text-xs text-ink-muted">{hint}</p>}
    </div>
  );
}

/** Fold scores can be negative (sklearn's `neg_*` scorers), so bars are scaled
 *  by the largest magnitude and always drawn from a shared baseline. */
function CrossValidationBlock({ run }: { run: TrainingRunDetail }) {
  const cv = run.cv;
  if (!cv) return null;

  const magnitudes = cv.scores.map((score) => Math.abs(score));
  const largest = magnitudes.length > 0 ? Math.max(...magnitudes) : 0;
  const hasNegative = cv.scores.some((score) => score < 0);

  return (
    <section>
      <h2 className="font-serif text-h3 text-ink mb-2">
        Cross-validation
      </h2>
      <p className="mb-4 text-sm text-ink-muted">
        The data was split {cv.folds} ways and the model retrained {cv.folds}{" "}
        times, scoring on a different slice each time. Scores that sit close
        together mean the result is stable rather than a lucky split.
        {hasNegative &&
          " Some scikit-learn scorers are negated, so a negative score here is normal — closer to zero is better."}
      </p>

      <ul className="space-y-2">
        {cv.scores.map((score, index) => {
          const width =
            largest > 0 ? Math.max(1, (Math.abs(score) / largest) * 100) : 0;
          return (
            <li key={`fold-${index}`} className="flex items-center gap-3">
              <span className="w-20 shrink-0 text-sm text-ink">
                Fold {index + 1}
              </span>
              <span className="h-3 flex-1 rounded-control bg-paper-sunken">
                <span
                  className="block h-3 rounded-control bg-series-1 transition-opacity hover:opacity-80"
                  style={{ width: `${width}%` }}
                  title={`Fold ${index + 1} (${cv.scoring}): ${formatScore(score)}`}
                />
              </span>
              <span className="w-24 shrink-0 text-right font-mono text-xs text-ink">
                {formatScore(score)}
              </span>
            </li>
          );
        })}
      </ul>

      <p className="mt-3 text-sm text-ink">
        <span className="font-semibold">Mean {cv.scoring}: </span>
        <span className="font-mono">{formatScore(cv.mean)}</span>
        <span className="font-mono"> ± {formatScore(cv.std)}</span>
        <span className="text-ink-muted"> (standard deviation across folds)</span>
      </p>
    </section>
  );
}

export default function EvaluationReport() {
  const { dataset, run, setRun, goToTool, restoring, restoreError } =
    useWorkspace();

  usePageSeo({
    title: "Model Evaluation — Metrics, Diagnostics & Predictions",
    description:
      "Review fitted model evaluations: confusion matrices, per-class classification metrics, regression residuals, feature importance, and real-time predictions.",
    path: "/app/evaluation",
  });

  const [pastRuns, setPastRuns] = useState<TrainingRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [listError, setListError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [listing, setListing] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);

  const loadRunList = useCallback(async () => {
    setListing(true);
    setListError(null);
    try {
      setPastRuns(await services.listRuns());
    } catch (caught) {
      setListError(apiErrorDetail(caught));
    } finally {
      setListing(false);
    }
  }, []);

  useEffect(() => {
    void loadRunList();
  }, [loadRunList]);

  const handleLoadRun = async () => {
    if (selectedRunId === "") return;
    setLoadingRun(true);
    setLoadError(null);
    try {
      setRun(await services.getRun(selectedRunId));
    } catch (caught) {
      setLoadError(apiErrorDetail(caught));
    } finally {
      setLoadingRun(false);
    }
  };

  const runPicker = (
    <div className="mt-4">
      <h3 className="mb-2 text-sm font-semibold text-ink">
        Open a previous training run
      </h3>
      {listError && (
        <ErrorBox
          title="Could not list past runs."
          message={listError}
          onDismiss={() => setListError(null)}
        />
      )}
      {listing ? (
        <p className="text-sm text-ink-muted">Loading past runs…</p>
      ) : pastRuns.length === 0 ? (
        <p className="text-sm text-ink-muted">
          No training runs have been saved yet.
        </p>
      ) : (
        <div className="flex flex-col gap-3 sm:flex-row">
          <select
            value={selectedRunId}
            onChange={(event) => setSelectedRunId(event.target.value)}
            aria-label="Previous training run"
            className="w-full rounded-control border border-rule bg-paper-sunken p-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="">Choose a run…</option>
            {pastRuns.map((summary) => (
              <option key={summary.id} value={summary.id}>
                {runLabel(summary)} — {summary.primary_metric_name}{" "}
                {summary.primary_metric === null
                  ? "n/a"
                  : summary.primary_metric.toFixed(3)}{" "}
                ({formatTimestamp(summary.created_at)})
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void handleLoadRun()}
            disabled={selectedRunId === "" || loadingRun}
            className="btn btn-primary text-body px-6 py-3 shrink-0"
          >
            {loadingRun ? "Loading…" : "Load run"}
          </button>
        </div>
      )}
      {loadError && (
        <div className="mt-3">
          <ErrorBox
            title="Could not open that run."
            message={loadError}
            onDismiss={() => setLoadError(null)}
          />
        </div>
      )}
    </div>
  );

  if (restoring) {
    return (
      <div className="bg-paper min-h-full p-8">
        <div className="panel">
          <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">
            Evaluation Report
          </h1>
          <p className="text-sm text-ink-muted">
            Restoring your last training run…
          </p>
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="bg-paper min-h-full p-8">
        <div className="panel">
          <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">
            Evaluation Report
          </h1>
          <CourseLink courseId="evaluation" />
          {restoreError && (
            <ErrorBox title="Could not reach the API." message={restoreError} />
          )}
          <p className="mb-6 text-ink-muted">
            There is no trained model to evaluate yet. Train one on the Data
            Training page — pick a target column, choose an algorithm, and this
            page will fill with its real metrics, confusion matrix or residual
            plot, feature importances, and a live prediction form.
          </p>
          <button
            type="button"
            onClick={() => goToTool("Data Training")}
            className="btn btn-primary text-body py-3 w-full"
          >
            Go to Data Training
          </button>
          <hr className="my-6 border-t border-rule" />
          {runPicker}
        </div>
      </div>
    );
  }

  const isClassification = run.task_type === "classification";
  const failed = run.status === "failed";

  // Applying a cleaning recipe, undoing one, or loading a previous dataset
  // swaps the active dataset but deliberately keeps the trained run. When that
  // happens the numbers below describe a dataset the user has moved off, so
  // say so instead of letting them read as the current data.
  const trainedOnOtherDataset =
    dataset !== null && run.dataset_id !== dataset.id;

  return (
    <div className="bg-paper min-h-full p-8">
      <div className="panel">
        <h1 className="font-serif text-h1 font-semibold text-ink mb-2">
          Evaluation Report
        </h1>
        <p className="mb-8 text-lg text-ink">{runLabel(run)}</p>
        <CourseLink courseId="evaluation" />

        {trainedOnOtherDataset && dataset !== null && (
          <div className="mb-6 rounded-control border border-state-warn/40 bg-state-warn/10 p-4">
            <p className="text-sm font-semibold text-state-warn">
              {WARNING_MARK} This report is not about your active dataset
            </p>
            <p className="mt-1 text-sm text-ink">
              These results come from{" "}
              <span className="font-semibold">{run.dataset_name}</span>, but the
              dataset now loaded is{" "}
              <span className="font-semibold">{dataset.name}</span>. Train again
              to evaluate the data you are working on.
            </p>
            <button
              type="button"
              onClick={() => goToTool("Data Training")}
              className="mt-3 rounded-control bg-rule px-4 py-2 text-sm font-semibold text-ink transition-colors hover:bg-rule-strong"
            >
              Go to Data Training
            </button>
          </div>
        )}

        {restoreError && (
          <ErrorBox title="Could not reach the API." message={restoreError} />
        )}

        {failed && run.error && (
          <ErrorBox title="This training run failed." message={run.error} />
        )}
        {!failed && run.error && (
          <ErrorBox title="The backend reported a problem." message={run.error} />
        )}

        {run.warnings.length > 0 && (
          <div className="mb-6 rounded-control border border-state-warn/40 bg-state-warn/10 p-4">
            <p className="mb-2 text-sm font-semibold text-state-warn">
              {WARNING_MARK} {run.warnings.length} warning
              {run.warnings.length === 1 ? "" : "s"} from training
            </p>
            <ul className="list-disc space-y-1 pl-5 text-sm text-state-warn">
              {run.warnings.map((warning, index) => (
                <li key={`warning-${index}`}>{warning}</li>
              ))}
            </ul>
          </div>
        )}

        <section className="mb-8">
          <h2 className="font-serif text-h3 text-ink mb-4">
            About this run
          </h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Fact label="Algorithm" value={run.algorithm_label} />
            <Fact
              label="Task"
              value={isClassification ? "Classification" : "Regression"}
              hint={
                isClassification
                  ? "predicting a category"
                  : "predicting a number"
              }
            />
            <Fact label="Target column" value={run.target_column} />
            <Fact label="Dataset" value={run.dataset_name} />
            <Fact
              label="Rows used"
              value={`${run.n_train} train / ${run.n_test} test`}
              hint={`test size ${(run.test_size * 100).toFixed(0)}%`}
            />
            <Fact
              label="Features"
              value={`${run.feature_columns.length} column${
                run.feature_columns.length === 1 ? "" : "s"
              }`}
              hint={run.feature_columns.join(", ")}
            />
            <Fact
              label="Training time"
              value={formatDuration(run.duration_ms)}
            />
            <Fact label="Trained at" value={formatTimestamp(run.created_at)} />
            <Fact
              label="Random seed"
              value={
                run.random_state === null
                  ? "not set"
                  : String(run.random_state)
              }
              hint={
                run.random_state === null
                  ? "results vary between runs"
                  : "same seed reproduces this run"
              }
            />
            <Fact
              label="Feature scaling"
              value={run.scale_features ? "on" : "off"}
            />
            <Fact
              label="Stratified split"
              value={
                isClassification ? (run.stratify ? "on" : "off") : "n/a"
              }
              hint={
                isClassification
                  ? "keeps class balance in both splits"
                  : "classification only"
              }
            />
            <Fact
              label="Status"
              value={failed ? "failed" : "success"}
              hint={run.id}
            />
          </div>

          {Object.keys(run.effective_hyperparameters).length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-semibold text-ink">
                Hyperparameters actually used
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <tbody>
                    {Object.entries(run.effective_hyperparameters).map(
                      ([name, value]) => (
                        <tr key={name} className="border-b border-rule">
                          <td className="py-1 pr-6 text-ink-muted">{name}</td>
                          <td className="py-1 font-mono text-ink">
                            {stringifyParam(value)}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>

        {failed ? (
          <>
            <p className="mb-6 text-sm text-ink-muted">
              This run did not finish, so there are no metrics to show. Fix the
              problem reported above and train again, or open a run that
              succeeded.
            </p>
            <button
              type="button"
              onClick={() => goToTool("Data Training")}
              className="btn btn-primary text-body py-3 w-full"
            >
              Back to Data Training
            </button>
            <hr className="my-6 border-t border-rule" />
            {runPicker}
          </>
        ) : (
          <>
            <MetricsPanel
              taskType={run.task_type}
              metrics={run.metrics}
              trainMetrics={run.train_metrics}
            />

            {run.cv && (
              <>
                <hr className="my-8 border-t border-rule" />
                <CrossValidationBlock run={run} />
              </>
            )}

            <hr className="my-8 border-t border-rule" />

            {isClassification ? (
              run.confusion_matrix ? (
                <ConfusionMatrixTable
                  data={run.confusion_matrix}
                  perClass={run.per_class}
                />
              ) : (
                <section>
                  <h2 className="font-serif text-h3 text-ink mb-4">
                    Confusion matrix
                  </h2>
                  <p className="text-small text-ink-muted max-w-measure">
                    This run did not return a confusion matrix.
                  </p>
                </section>
              )
            ) : (
              <ResidualPlot
                points={run.residual_sample}
                targetColumn={run.target_column}
              />
            )}

            <p className="mt-3 text-xs text-ink-muted">
              {isClassification
                ? "A residual plot is not shown because it compares predicted numbers against actual numbers, which only makes sense for regression."
                : "A confusion matrix is not shown because it counts predicted classes against actual classes, which only makes sense for classification."}
            </p>

            <hr className="my-8 border-t border-rule" />

            <FeatureImportanceChart
              key={run.id}
              importance={run.feature_importance}
              algorithmLabel={run.algorithm_label}
            />

            <hr className="my-8 border-t border-rule" />

            <section>
              <h2 className="font-serif text-h3 text-ink mb-4">
                Take it with you
              </h2>
              <div className="space-y-4">
                <a
                  href={services.reportDownloadUrl(run.id, "md")}
                  download
                  className="btn btn-primary text-body py-3 w-full"
                >
                  Download the evaluation report (Markdown)
                </a>
                <a
                  href={services.modelDownloadUrl(run.id)}
                  download
                  className="block w-full rounded-control bg-rule px-4 py-3 text-center text-lg font-semibold text-ink transition-colors hover:bg-rule-strong"
                >
                  Download the trained model (.joblib)
                </a>
              </div>
              <p className="mt-2 text-xs text-ink-muted">
                The .joblib file is the fitted scikit-learn pipeline, including
                the imputers, encoders and scaler — load it with{" "}
                <code>joblib.load</code> and call <code>predict</code> directly.
              </p>
            </section>

            <hr className="my-8 border-t border-rule" />

            {/* Keyed by run id so switching runs cannot leave the previous
                run's prediction sitting under the new run's target name. */}
            <PredictionForm
              key={run.id}
              runId={run.id}
              schema={run.input_schema}
              taskType={run.task_type}
              targetColumn={run.target_column}
            />

            <hr className="my-8 border-t border-rule" />

            {runPicker}
          </>
        )}
      </div>
    </div>
  );
}
