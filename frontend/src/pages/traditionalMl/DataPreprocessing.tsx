import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import DatasetProfile from "../../components/DatasetProfile";
import InsertDataFile from "../../components/InsertFile";
import PreprocessingCheckbox from "../../components/PreprocessingCheckbox";
import TableViewer from "../../components/TableViewer";
import { useWorkspace } from "../../context/workspaceContext";
import services from "../../services/services";
import { WARNING_MARK } from "../../explainer/marks";
import type {
  DatasetDetail,
  DatasetPreviewResult,
  DatasetSummary,
  PreprocessReport,
  PreprocessStep,
} from "../../interface/MlTypes";
import {
  opLabel,
  pruneRecipe,
  validateRecipe,
} from "../../interface/PreprocessingTypes";

/** Service methods reject with a normalized ApiError; render its `detail`. */
function errorDetail(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "string" && detail.length > 0) return detail;
  }
  if (error instanceof Error) return error.message;
  return "Something went wrong talking to the API.";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? iso : parsed.toLocaleString();
}

interface StatTileProps {
  label: string;
  value: string;
  sublabel?: string;
}

function StatTile({ label, value, sublabel }: StatTileProps) {
  return (
    <div className="rounded-control border border-rule bg-paper px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-ink-muted">
        {label}
      </div>
      <div className="text-2xl font-bold text-ink break-words">{value}</div>
      {sublabel && <div className="text-xs text-ink-muted">{sublabel}</div>}
    </div>
  );
}

interface ErrorBoxProps {
  message: string;
  onDismiss?: () => void;
}

function ErrorBox({ message, onDismiss }: ErrorBoxProps) {
  return (
    <div
      role="alert"
      className="mb-6 rounded-control border border-state-bad/40 bg-state-bad/10 p-4"
    >
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm text-state-bad">
          <span className="font-semibold">Error — </span>
          {message}
        </p>
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-sm font-semibold text-state-bad hover:underline"
          >
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}

function ReportPanel({ report }: { report: PreprocessReport }) {
  const skipped = report.steps.filter((step) => step.skipped);

  return (
    <div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
        <StatTile
          label="Rows before"
          value={report.rows_before.toLocaleString()}
        />
        <StatTile
          label="Rows after"
          value={report.rows_after.toLocaleString()}
          sublabel={`${report.rows_after - report.rows_before >= 0 ? "+" : ""}${(
            report.rows_after - report.rows_before
          ).toLocaleString()}`}
        />
        <StatTile
          label="Columns before"
          value={report.cols_before.toLocaleString()}
        />
        <StatTile
          label="Columns after"
          value={report.cols_after.toLocaleString()}
          sublabel={`${report.cols_after - report.cols_before >= 0 ? "+" : ""}${(
            report.cols_after - report.cols_before
          ).toLocaleString()}`}
        />
      </div>

      {skipped.length > 0 && (
        <p className="mb-3 text-sm font-semibold text-state-warn">
          <span aria-hidden="true">{WARNING_MARK}</span> {skipped.length} of{" "}
          {report.steps.length} steps were skipped — see the list below.
        </p>
      )}

      <ol className="space-y-2">
        {report.steps.map((step, index) => (
          <li
            key={`${step.op}-${index}`}
            className={`rounded-control border p-3 ${
              step.skipped
                ? "border-state-warn/40 bg-state-warn/10"
                : "border-rule bg-paper-raised"
            }`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-primary text-white text-xs font-bold">
                {index + 1}
              </span>
              <span className="font-semibold text-ink">{opLabel(step.op)}</span>
              {step.skipped && (
                <span className="rounded-control px-2 py-0.5 text-xs font-semibold text-state-warn border border-state-warn/40">
                  <span aria-hidden="true">{WARNING_MARK}</span> Skipped
                </span>
              )}
              <span className="ml-auto text-xs text-ink-muted">
                {step.rows_after.toLocaleString()} rows ·{" "}
                {step.cols_after.toLocaleString()} cols
              </span>
            </div>
            <p className="mt-1 text-sm text-ink-muted">{step.message}</p>
            {step.columns.length > 0 && (
              <p className="mt-1 text-xs text-ink-muted">
                Columns: {step.columns.join(", ")}
              </p>
            )}
          </li>
        ))}
      </ol>

      {report.warnings.length > 0 && (
        <ul className="mt-4 space-y-1">
          {report.warnings.map((warning) => (
            <li key={warning} className="text-sm font-semibold text-state-warn">
              <span aria-hidden="true">{WARNING_MARK}</span> {warning}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function DataPreprocessing() {
  const { dataset, setDataset, goToTool, restoring, restoreError } =
    useWorkspace();

  const [steps, setSteps] = useState<PreprocessStep[]>([]);
  const [report, setReport] = useState<PreprocessReport | null>(null);
  const [previewResult, setPreviewResult] =
    useState<DatasetPreviewResult | null>(null);
  const [busy, setBusy] = useState<"preview" | "apply" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [appliedFrom, setAppliedFrom] = useState<{
    name: string;
    parentId: string;
  } | null>(null);
  const [undoing, setUndoing] = useState<boolean>(false);

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [listLoading, setListLoading] = useState<boolean>(false);
  const [listError, setListError] = useState<string | null>(null);
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const datasetId = dataset?.id ?? null;
  const datasetRef = useRef<DatasetDetail | null>(dataset);
  datasetRef.current = dataset;

  const refreshList = useCallback(async () => {
    setListLoading(true);
    setListError(null);
    try {
      setDatasets(await services.listDatasets());
    } catch (listFailure) {
      setListError(errorDetail(listFailure));
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  // A recipe built against the previous dataset can point at columns that no
  // longer exist. Prune rather than discard, so the user keeps their work.
  useEffect(() => {
    const columns = datasetRef.current?.columns ?? [];
    setSteps((current) =>
      current.length === 0 ? current : pruneRecipe(current, columns)
    );
  }, [datasetId]);

  const problems = useMemo(
    () => validateRecipe(steps, dataset?.columns ?? []),
    [steps, dataset]
  );

  const canRun =
    dataset !== null && steps.length > 0 && problems.length === 0 && busy === null;

  const resetResults = () => {
    setReport(null);
    setPreviewResult(null);
  };

  const handleUploadSuccess = (uploaded: DatasetDetail) => {
    setDataset(uploaded);
    setSteps([]);
    setAppliedFrom(null);
    resetResults();
    setError(null);
    void refreshList();
  };

  const runPreprocess = async (dryRun: boolean) => {
    if (!dataset) return;
    setBusy(dryRun ? "preview" : "apply");
    setError(null);

    try {
      const response = await services.preprocess(dataset.id, {
        steps,
        dry_run: dryRun,
      });
      setReport(response.report);

      if (dryRun) {
        setPreviewResult(response.dataset);
        return;
      }

      const applied = response.dataset;
      if (!applied || applied.id === null) {
        setPreviewResult(null);
        setError(
          "The recipe ran but the API did not return a saved dataset. Reload it from the list below."
        );
        void refreshList();
        return;
      }

      const cleaned: DatasetDetail = { ...applied, id: applied.id };
      setPreviewResult(null);
      setAppliedFrom({
        name: dataset.name,
        parentId: cleaned.parent ?? dataset.id,
      });
      setDataset(cleaned);
      void refreshList();
    } catch (runFailure) {
      setError(errorDetail(runFailure));
    } finally {
      setBusy(null);
    }
  };

  const undoApply = async () => {
    if (!appliedFrom) return;
    setUndoing(true);
    setError(null);
    try {
      const previous = await services.getDataset(appliedFrom.parentId, 10);
      setDataset(previous);
      setAppliedFrom(null);
      resetResults();
    } catch (undoFailure) {
      setError(errorDetail(undoFailure));
    } finally {
      setUndoing(false);
    }
  };

  const loadDataset = async (id: string) => {
    setPendingId(id);
    setError(null);
    try {
      const detail = await services.getDataset(id, 10);
      setDataset(detail);
      setAppliedFrom(null);
      resetResults();
    } catch (loadFailure) {
      setError(errorDetail(loadFailure));
    } finally {
      setPendingId(null);
    }
  };

  const removeDataset = async (id: string) => {
    setPendingId(id);
    setError(null);
    try {
      await services.deleteDataset(id);
      if (datasetRef.current?.id === id) {
        setDataset(null);
        setAppliedFrom(null);
        resetResults();
      }
      setConfirmDeleteId(null);
      await refreshList();
    } catch (deleteFailure) {
      setError(errorDetail(deleteFailure));
    } finally {
      setPendingId(null);
    }
  };

  const previousDatasets = (
    <div className="w-full panel p-8">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <h2 className="text-xl font-semibold text-ink">
            Load a previous dataset
          </h2>
          <p className="text-sm text-ink-muted">
            Everything you have uploaded or cleaned, newest first.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refreshList()}
          disabled={listLoading}
          className="whitespace-nowrap rounded-control border border-rule px-3 py-2 text-sm font-semibold text-ink hover:bg-paper-sunken disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {listLoading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {listError && <ErrorBox message={listError} onDismiss={() => setListError(null)} />}

      {listLoading && datasets.length === 0 ? (
        <p className="text-sm text-ink-muted">Loading datasets…</p>
      ) : datasets.length === 0 ? (
        <p className="text-sm text-ink-muted">
          Nothing saved yet — upload a CSV above and it will appear here.
        </p>
      ) : (
        <ul className="space-y-2">
          {datasets.map((summary) => {
            const active = summary.id === datasetId;
            const pending = pendingId === summary.id;
            return (
              <li
                key={summary.id}
                className={`rounded-control border p-3 ${
                  active ? "border-primary bg-primary/5" : "border-rule"
                }`}
              >
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex-grow min-w-[12rem]">
                    <div className="font-semibold text-ink break-words">
                      {summary.name}
                      {active && (
                        <span className="ml-2 rounded-control bg-primary px-2 py-0.5 text-xs font-semibold text-white">
                          Active
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-ink-muted">
                      {summary.source === "preprocessed"
                        ? "Cleaned"
                        : "Uploaded"}{" "}
                      · {summary.n_rows.toLocaleString()} rows ·{" "}
                      {summary.n_cols.toLocaleString()} cols ·{" "}
                      {formatDate(summary.created_at)}
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => void loadDataset(summary.id)}
                    disabled={active || pending}
                    className="rounded-control bg-primary px-3 py-2 text-sm font-semibold text-white hover:bg-[#471724] transition-colors disabled:bg-primary/50 disabled:cursor-not-allowed"
                  >
                    {pending ? "Working…" : active ? "In use" : "Load"}
                  </button>

                  {confirmDeleteId === summary.id ? (
                    <>
                      <span className="text-sm font-semibold text-state-bad">
                        Delete permanently?
                      </span>
                      <button
                        type="button"
                        onClick={() => void removeDataset(summary.id)}
                        disabled={pending}
                        className="rounded-control border border-state-bad px-3 py-2 text-sm font-semibold text-state-bad hover:bg-state-bad/10 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {pending ? "Deleting…" : "Yes, delete"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmDeleteId(null)}
                        className="rounded-control border border-rule px-3 py-2 text-sm font-semibold text-ink hover:bg-paper-sunken"
                      >
                        Keep
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setConfirmDeleteId(summary.id)}
                      disabled={pending}
                      className="rounded-control border border-rule px-3 py-2 text-sm font-semibold text-ink hover:bg-paper-sunken disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );

  if (restoring) {
    return (
      <div className="bg-paper min-h-full p-8">
        <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">
          Data Preprocessing
        </h1>
        <div className="w-full panel p-8">
          <p className="text-ink-muted">Restoring your last dataset…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-paper min-h-full p-8">
      <h1 className="font-serif text-h1 font-semibold text-ink mb-2">
        Data Preprocessing
      </h1>
      <p className="font-serif text-body text-ink-muted max-w-measure mb-8">
        Step 1 of 3 — upload a CSV, look at what is actually in it, then build a
        cleaning recipe. Training and evaluation both run on whichever dataset
        is active here.
      </p>

      {restoreError && <ErrorBox message={restoreError} />}
      {error && <ErrorBox message={error} onDismiss={() => setError(null)} />}

      {!dataset ? (
        <div className="space-y-8">
          <InsertDataFile onUploadSuccess={handleUploadSuccess} />
          {previousDatasets}
        </div>
      ) : (
        <div className="space-y-8">
          <div className="w-full panel p-8">
            <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
              <div>
                <h2 className="text-xl font-semibold text-ink break-words">
                  {dataset.name}
                </h2>
                <p className="text-sm text-ink-muted">
                  {dataset.source === "preprocessed"
                    ? "Cleaned dataset"
                    : "Uploaded dataset"}{" "}
                  · added {formatDate(dataset.created_at)}
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setDataset(null);
                  setSteps([]);
                  setAppliedFrom(null);
                  resetResults();
                }}
                className="rounded-control border border-rule px-3 py-2 text-sm font-semibold text-ink hover:bg-paper-sunken"
              >
                Upload a different file
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <StatTile label="Rows" value={dataset.n_rows.toLocaleString()} />
              <StatTile
                label="Columns"
                value={dataset.n_cols.toLocaleString()}
              />
              <StatTile label="Size" value={formatBytes(dataset.size_bytes)} />
              <StatTile
                label="Missing cells"
                value={dataset.columns
                  .reduce((total, column) => total + column.missing, 0)
                  .toLocaleString()}
                sublabel={`${dataset.columns.filter((c) => c.missing > 0).length} columns affected`}
              />
            </div>

            {appliedFrom && (
              <div className="mt-4 rounded-control border border-state-good/40 bg-state-good/10 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-sm text-state-good">
                    <span aria-hidden="true">✓</span>{" "}
                    <span className="font-semibold">
                      Switched to the cleaned dataset.
                    </span>{" "}
                    Training and evaluation will now use it instead of{" "}
                    {appliedFrom.name}.
                  </p>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => void undoApply()}
                      disabled={undoing}
                      className="rounded-control border border-state-good px-3 py-2 text-sm font-semibold text-state-good hover:bg-state-good/10 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {undoing ? "Switching back…" : "Undo — go back"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setAppliedFrom(null)}
                      className="rounded-control border border-rule px-3 py-2 text-sm font-semibold text-ink hover:bg-paper-sunken"
                    >
                      Keep it
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="w-full panel p-8">
            <TableViewer
              preview={dataset.preview}
              title="Preview"
              totalRows={dataset.n_rows}
            />
          </div>

          <div className="w-full panel p-8">
            <DatasetProfile columns={dataset.columns} />
          </div>

          <div className="w-full panel p-8">
            <PreprocessingCheckbox
              columns={dataset.columns}
              steps={steps}
              onStepsChange={setSteps}
              disabled={busy !== null}
            />

            <hr className="my-6 border-t border-rule" />

            {problems.length > 0 && (
              <ul className="mb-4 space-y-1">
                {problems.map((problem) => (
                  <li
                    key={problem}
                    className="text-sm font-semibold text-state-warn"
                  >
                    <span aria-hidden="true">{WARNING_MARK}</span> {problem}
                  </li>
                ))}
              </ul>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <button
                type="button"
                onClick={() => void runPreprocess(true)}
                disabled={!canRun}
                className="w-full bg-rule text-ink py-3 px-4 rounded-control text-lg font-semibold hover:bg-rule-strong transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {busy === "preview" ? "Previewing…" : "Preview changes"}
              </button>
              <button
                type="button"
                onClick={() => void runPreprocess(false)}
                disabled={!canRun}
                className="w-full bg-primary text-white py-3 px-4 rounded-control text-lg font-semibold hover:bg-[#471724] transition-colors disabled:bg-primary/50 disabled:cursor-not-allowed"
              >
                {busy === "apply" ? "Applying…" : "Apply preprocessing"}
              </button>
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              Preview leaves your data untouched. Apply saves a new cleaned
              dataset and makes it the active one.
            </p>
          </div>

          {report && (
            <div className="w-full panel p-8">
              <div className="mb-4">
                <h2 className="text-xl font-semibold text-ink">
                  {previewResult ? "Preview report" : "What was applied"}
                </h2>
                <p className="text-sm text-ink-muted">
                  {previewResult
                    ? "Nothing has been saved — this is what the recipe would do."
                    : "This recipe has been saved as the active dataset."}
                </p>
              </div>
              <ReportPanel report={report} />
            </div>
          )}

          {previewResult && (
            <div className="w-full panel p-8">
              <TableViewer
                preview={previewResult.preview}
                title="Preview of the cleaned data"
                totalRows={previewResult.n_rows}
              />
              <hr className="my-6 border-t border-rule" />
              <DatasetProfile columns={previewResult.columns} />
            </div>
          )}

          <div className="w-full panel p-8">
            <h2 className="text-xl font-semibold text-ink mb-4">
              Next steps
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <a
                href={services.datasetDownloadUrl(dataset.id)}
                download
                className="block w-full text-center bg-rule text-ink py-3 px-4 rounded-control text-lg font-semibold hover:bg-rule-strong transition-colors"
              >
                Download data
              </a>
              <button
                type="button"
                onClick={() => goToTool("Data Training")}
                className="w-full bg-primary text-white py-3 px-4 rounded-control text-lg font-semibold hover:bg-[#471724] transition-colors disabled:bg-primary/50 disabled:cursor-not-allowed"
              >
                Continue to Training
              </button>
            </div>
          </div>

          {previousDatasets}
        </div>
      )}
    </div>
  );
}

export default DataPreprocessing;
