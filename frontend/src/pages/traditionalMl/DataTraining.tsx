/**
 * Data Training — fit a real scikit-learn model on the uploaded CSV.
 *
 * Everything on this page comes from the API: the columns are the profiled
 * columns of the active dataset, the algorithm list is the backend registry,
 * and the hyperparameter form is generated from whatever that registry says
 * the chosen estimator accepts.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  FeatureColumnSelector,
  TargetColumnSelector,
} from "../../components/ColumnSelector";
import HyperparameterForm from "../../components/HyperparameterForm";
import { useWorkspace } from "../../context/workspaceContext";
import services from "../../services/services";
import { WARNING_MARK } from "../../explainer/marks";
import type {
  AlgorithmSpec,
  ApiError,
  ColumnProfile,
  RequestedTaskType,
  TaskType,
  TrainRequest,
  TrainingRunDetail,
  TrainingRunSummary,
} from "../../interface/MlTypes";

/**
 * The real training time is unknowable from the client, so the progress bar is
 * indeterminate. Tailwind 3 ships no indeterminate keyframe, and the config is
 * not ours to edit, so the animation is plain CSS.
 */
const INDETERMINATE_CSS = `
@keyframes furina-indeterminate {
  0%   { transform: translateX(-110%); }
  100% { transform: translateX(430%); }
}
.furina-indeterminate-bar {
  width: 24%;
  animation: furina-indeterminate 1.4s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  .furina-indeterminate-bar { animation-duration: 3.2s; animation-timing-function: linear; }
}
`;

const TASK_CHOICES: { value: RequestedTaskType; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "classification", label: "Classification" },
  { value: "regression", label: "Regression" },
];

const CV_FOLD_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10];

/** The service layer throws a flat `ApiError`; anything else gets wrapped. */
function asApiError(error: unknown): ApiError {
  if (error !== null && typeof error === "object" && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "string") return error as ApiError;
  }
  return {
    detail:
      error instanceof Error
        ? error.message
        : "Training failed for an unknown reason.",
  };
}

function formatMetric(metric: number | null): string {
  if (metric === null || !Number.isFinite(metric)) return "—";
  const magnitude = Math.abs(metric);
  if (magnitude !== 0 && (magnitude < 0.0001 || magnitude >= 100000)) {
    return metric.toExponential(2);
  }
  let text = metric.toFixed(4);
  if (text.includes(".")) text = text.replace(/0+$/, "").replace(/\.$/, "");
  return text;
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) return `${durationMs} ms`;
  return `${(durationMs / 1000).toFixed(2)} s`;
}

function metricLabel(name: string): string {
  return name === "r2" ? "R²" : name.replace(/_/g, " ");
}

/** Spell out what "auto" resolved to, and why, so it can be overridden knowingly. */
function explainAutoTask(profile: ColumnProfile): string {
  if (profile.suggested_task === null) {
    return `Auto cannot read a task type from “${profile.name}” — choose Classification or Regression yourself.`;
  }

  const resolved =
    profile.suggested_task === "classification" ? "Classification" : "Regression";

  let because: string;
  switch (profile.kind) {
    case "boolean":
      because = "it holds only two true/false values";
      break;
    case "numeric":
      because =
        profile.suggested_task === "classification"
          ? `${profile.unique} distinct whole numbers is few enough to treat as classes`
          : `it holds ${profile.unique} distinct numbers spread across a continuous range`;
      break;
    case "datetime":
      because = "it is a date/time column";
      break;
    case "text":
      because = `it holds ${profile.unique} distinct free-text values`;
      break;
    default:
      because = `it holds ${profile.unique} distinct labels`;
      break;
  }

  return `Auto → ${resolved}, because ${profile.name} — ${because}.`;
}

/** The split the API accepts: never less than 5% and never more than 95%. */
const clampPct = (percent: number) =>
  Math.min(95, Math.max(5, Math.round(percent)));

/**
 * What is being typed into one of the two split boxes right now.
 *
 * The boxes cannot be plain controlled inputs over `trainPct`: clamping every
 * keystroke rewrites the box mid-word, so typing "30" into the testing box
 * lands on 5 after the "3" and finishes at 50. Holding the raw text of the one
 * focused box lets a two-digit number be typed in full; `trainPct` is still the
 * single source of truth, and the clamp runs once, when the value is committed.
 */
interface SplitDraft {
  field: "train" | "test";
  text: string;
}

/** The training percentage a draft means, or null while it is not a number. */
function draftTrainPct(draft: SplitDraft): number | null {
  if (draft.text.trim() === "") return null;
  const parsed = Number(draft.text);
  if (!Number.isFinite(parsed)) return null;
  return clampPct(draft.field === "train" ? parsed : 100 - parsed);
}

const DataTraining = () => {
  const { dataset, setRun, goToTool, restoring, restoreError } = useWorkspace();

  /* ---------------- algorithm registry ---------------- */
  const [algorithms, setAlgorithms] = useState<AlgorithmSpec[]>([]);
  const [algorithmsLoading, setAlgorithmsLoading] = useState<boolean>(true);
  const [algorithmsError, setAlgorithmsError] = useState<string | null>(null);

  /* ---------------- form ---------------- */
  const [trainPct, setTrainPct] = useState<number>(80);
  const [splitDraft, setSplitDraft] = useState<SplitDraft | null>(null);
  const [target, setTarget] = useState<string>("");
  const [features, setFeatures] = useState<string[]>([]);
  const [requestedTask, setRequestedTask] = useState<RequestedTaskType>("auto");
  const [seed, setSeed] = useState<string>("42");
  const [algorithm, setAlgorithm] = useState<string>("");
  const [hyperparams, setHyperparams] = useState<Record<string, unknown>>({});
  const [stratify, setStratify] = useState<boolean>(true);
  const [scaleFeatures, setScaleFeatures] = useState<boolean>(true);
  const [cvFolds, setCvFolds] = useState<number | null>(null);
  const [runName, setRunName] = useState<string>("");

  /* ---------------- training ---------------- */
  const [training, setTraining] = useState<boolean>(false);
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [trainError, setTrainError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<TrainingRunDetail | null>(null);

  /* ---------------- recent runs ---------------- */
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState<boolean>(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [openingRunId, setOpeningRunId] = useState<string | null>(null);

  const startedAtRef = useRef<number>(0);

  const columns = useMemo<ColumnProfile[]>(() => dataset?.columns ?? [], [dataset]);
  const nRows = dataset?.n_rows ?? 0;

  /* ---------------- load the registry once ---------------- */
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setAlgorithmsLoading(true);
      try {
        const specs = await services.listAlgorithms();
        if (cancelled) return;
        setAlgorithms(specs);
        setAlgorithmsError(null);
      } catch (error) {
        if (!cancelled) setAlgorithmsError(asApiError(error).detail);
      } finally {
        if (!cancelled) setAlgorithmsLoading(false);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  /* ---------------- seed the form from the real dataset ---------------- */
  useEffect(() => {
    if (!dataset) {
      setTarget("");
      setFeatures([]);
      return;
    }
    // Conventionally the label is the last column, so prefer the last one the
    // backend is willing to accept as a target.
    const usable = dataset.columns.filter((column) => column.suitable_as_target);
    const initialTarget =
      usable.length > 0 ? usable[usable.length - 1].name : "";

    setTarget(initialTarget);
    setFeatures(
      dataset.columns
        .filter((column) => column.name !== initialTarget)
        .map((column) => column.name)
    );
    setResult(null);
    setTrainError(null);
  }, [dataset]);

  /* ---------------- recent runs for this dataset ---------------- */
  const loadRuns = useCallback(async (datasetId: string) => {
    setRunsLoading(true);
    try {
      const list = await services.listRuns(datasetId);
      setRuns(list);
      setRunsError(null);
    } catch (error) {
      setRunsError(asApiError(error).detail);
    } finally {
      setRunsLoading(false);
    }
  }, []);

  const datasetId = dataset?.id ?? null;
  useEffect(() => {
    if (!datasetId) {
      setRuns([]);
      return;
    }
    void loadRuns(datasetId);
  }, [datasetId, loadRuns]);

  /* ---------------- derived: task, algorithms, split ---------------- */
  const targetProfile = useMemo(
    () => columns.find((column) => column.name === target) ?? null,
    [columns, target]
  );

  const resolvedTask: TaskType | null =
    requestedTask === "auto"
      ? targetProfile?.suggested_task ?? null
      : requestedTask;

  const availableAlgorithms = useMemo(
    () =>
      resolvedTask === null
        ? algorithms
        : algorithms.filter((spec) => spec.tasks.includes(resolvedTask)),
    [algorithms, resolvedTask]
  );

  // Keep the dropdown honest: an algorithm that no longer fits the task must
  // not stay selected behind the user's back.
  useEffect(() => {
    if (availableAlgorithms.length === 0) return;
    if (availableAlgorithms.some((spec) => spec.key === algorithm)) return;
    setAlgorithm(availableAlgorithms[0].key);
    setHyperparams({});
  }, [availableAlgorithms, algorithm]);

  const selectedSpec =
    availableAlgorithms.find((spec) => spec.key === algorithm) ?? null;

  const familyGroups = useMemo(() => {
    const grouped = new Map<string, AlgorithmSpec[]>();
    for (const spec of availableAlgorithms) {
      const bucket = grouped.get(spec.family);
      if (bucket) bucket.push(spec);
      else grouped.set(spec.family, [spec]);
    }
    return Array.from(grouped.entries());
  }, [availableAlgorithms]);

  // While a box is being typed into, the split the user is asking for is what
  // that box says — so the summary sentence and the request agree with what is
  // on screen, and there is no half-typed value left uncommitted at training
  // time.
  const effectiveTrainPct =
    splitDraft === null ? trainPct : (draftTrainPct(splitDraft) ?? trainPct);
  const testSize = Number(((100 - effectiveTrainPct) / 100).toFixed(4));
  const estimatedTest =
    nRows > 1 ? Math.min(Math.max(Math.ceil(nRows * testSize), 1), nRows - 1) : 0;
  const estimatedTrain = nRows > 1 ? nRows - estimatedTest : 0;

  const effectiveFeatures = useMemo(
    () =>
      features.filter(
        (name) =>
          name !== target && columns.some((column) => column.name === name)
      ),
    [features, target, columns]
  );

  const validationError = useMemo<string | null>(() => {
    if (!dataset) return "Upload a dataset first.";
    if (target === "") return "Choose the column you want to predict.";
    if (effectiveFeatures.length === 0)
      return "Select at least one feature column.";
    if (selectedSpec === null) return "Choose an algorithm.";
    if (nRows < 4) return "This dataset has too few rows to split and train.";
    return null;
  }, [dataset, target, effectiveFeatures.length, selectedSpec, nRows]);

  /* ---------------- elapsed-seconds counter ---------------- */
  useEffect(() => {
    if (!training) return;
    startedAtRef.current = Date.now();
    setElapsedMs(0);
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAtRef.current);
    }, 100);
    return () => window.clearInterval(timer);
  }, [training]);

  /* ---------------- handlers ---------------- */
  /** Turn what was typed into the committed `trainPct`, clamped exactly once. */
  const commitSplitDraft = () => {
    if (splitDraft === null) return;
    const committed = draftTrainPct(splitDraft);
    if (committed !== null) setTrainPct(committed);
    setSplitDraft(null);
  };

  const handleTargetChange = (nextTarget: string) => {
    setTarget(nextTarget);
    // Nothing else moves: the new target simply leaves the feature list.
    setFeatures((current) => current.filter((name) => name !== nextTarget));
  };

  const handleAlgorithmChange = (nextKey: string) => {
    setAlgorithm(nextKey);
    setHyperparams({});
  };

  const handleTrain = async () => {
    if (!dataset || selectedSpec === null || validationError !== null) return;

    setTraining(true);
    setTrainError(null);
    setResult(null);

    try {
      // Only settings this algorithm actually declares may travel.
      const allowed = new Set(
        selectedSpec.hyperparameters.map((spec) => spec.name)
      );
      const hyperparameters: Record<string, unknown> = Object.fromEntries(
        Object.entries(hyperparams).filter(([key]) => allowed.has(key))
      );

      const parsedSeed = seed.trim() === "" ? Number.NaN : Number(seed);
      const request: TrainRequest = {
        dataset_id: dataset.id,
        target_column: target,
        feature_columns: effectiveFeatures,
        algorithm: selectedSpec.key,
        task_type: requestedTask,
        test_size: testSize,
        random_state: Number.isFinite(parsedSeed) ? Math.trunc(parsedSeed) : null,
        hyperparameters,
        stratify,
        scale_features: scaleFeatures,
        cv_folds: cvFolds,
        name: runName.trim() === "" ? null : runName.trim(),
      };

      const detail = await services.train(request);
      setResult(detail);
      setRun(detail);
      void loadRuns(dataset.id);
    } catch (error) {
      setTrainError(asApiError(error));
    } finally {
      setTraining(false);
    }
  };

  const handleOpenRun = async (runId: string) => {
    setOpeningRunId(runId);
    setRunsError(null);
    try {
      const detail = await services.getRun(runId);
      setRun(detail);
      goToTool("Model Evaluation");
    } catch (error) {
      setRunsError(asApiError(error).detail);
    } finally {
      setOpeningRunId(null);
    }
  };

  /* ---------------- restoring / empty states ---------------- */
  if (restoring) {
    return (
      <div className="bg-paper min-h-full p-8 w-full">
        <div className="panel">
          <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">Data Training</h1>
          <p className="text-ink-muted">Loading your workspace…</p>
        </div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="bg-paper min-h-full p-8 w-full">
        <div className="panel">
          <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">Data Training</h1>

          {restoreError && (
            <div className="mb-6 rounded-control border-l-4 border-state-bad bg-red-50 p-4">
              <p className="font-semibold text-state-bad">
                ✕ Could not restore your last dataset
              </p>
              <p className="mt-1 text-sm text-ink">{restoreError}</p>
            </div>
          )}

          <div className="py-10 text-center">
            <h2 className="font-serif text-h3 text-ink mb-4">
              No dataset loaded yet
            </h2>
            <p className="mx-auto mb-6 max-w-xl text-ink-muted">
              Training needs a real CSV to learn from. Upload one on the Data
              Preprocessing page — Furina profiles every column there, and this
              page then builds the target, feature and algorithm choices out of
              that profile.
            </p>
            <button
              type="button"
              onClick={() => goToTool("Data Preprocessing")}
              className="btn btn-primary text-body py-3 mx-auto w-full max-w-sm"
            >
              Go to Data Preprocessing
            </button>
          </div>
        </div>
      </div>
    );
  }

  const formDisabled = training;

  return (
    <div className="bg-paper min-h-full p-8 w-full">
      <div className="panel">
        <style>{INDETERMINATE_CSS}</style>

        <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">Data Training</h1>

        <div className="mb-8 bg-paper-sunken p-4">
          <p className="text-sm text-ink-muted">Training on</p>
          <p className="text-lg font-semibold text-ink">{dataset.name}</p>
          <p className="text-sm text-ink-muted">
            {dataset.n_rows.toLocaleString()} rows · {dataset.n_cols} columns ·
            source: {dataset.source}
          </p>
        </div>

        {/* ---------------- split ---------------- */}
        <div className="mb-10">
          <h2 className="font-serif text-h3 text-ink mb-4">
            Split the data
          </h2>
          <div className="relative flex items-center">
            <input
              type="range"
              min={5}
              max={95}
              step={1}
              value={effectiveTrainPct}
              disabled={formDisabled}
              onChange={(event) => {
                setTrainPct(clampPct(Number(event.target.value)));
                setSplitDraft(null);
              }}
              className="w-full h-2 bg-paper-sunken rounded-control appearance-none cursor-pointer accent-primary disabled:cursor-not-allowed"
            />
          </div>
          <div className="flex justify-between items-center mt-4">
            <div className="flex items-center space-x-2">
              <input
                type="number"
                name="training"
                min={5}
                max={95}
                value={
                  splitDraft?.field === "train"
                    ? splitDraft.text
                    : String(effectiveTrainPct)
                }
                disabled={formDisabled}
                onChange={(event) =>
                  setSplitDraft({ field: "train", text: event.target.value })
                }
                onBlur={commitSplitDraft}
                onKeyDown={(event) => {
                  if (event.key === "Enter") commitSplitDraft();
                }}
                className="w-20 p-2 border border-rule rounded-control text-center bg-paper-sunken disabled:cursor-not-allowed disabled:opacity-60"
              />
              <span className="font-medium text-ink">% Training</span>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="number"
                name="testing"
                min={5}
                max={95}
                value={
                  splitDraft?.field === "test"
                    ? splitDraft.text
                    : String(100 - effectiveTrainPct)
                }
                disabled={formDisabled}
                onChange={(event) =>
                  setSplitDraft({ field: "test", text: event.target.value })
                }
                onBlur={commitSplitDraft}
                onKeyDown={(event) => {
                  if (event.key === "Enter") commitSplitDraft();
                }}
                className="w-20 p-2 border border-rule rounded-control text-center bg-paper-sunken disabled:cursor-not-allowed disabled:opacity-60"
              />
              <span className="font-medium text-ink">% Testing</span>
            </div>
          </div>
          <p className="mt-2 text-sm text-ink-muted">
            test_size = {testSize} — roughly{" "}
            <span className="font-semibold text-ink">
              {estimatedTrain.toLocaleString()}
            </span>{" "}
            rows to train on and{" "}
            <span className="font-semibold text-ink">
              {estimatedTest.toLocaleString()}
            </span>{" "}
            rows held back for testing. The split must leave 5%–95% on each side.
          </p>
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- target ---------------- */}
        <div className="mb-10">
          <h2 className="font-serif text-h3 text-ink mb-4">
            Choose the target column
          </h2>
          <TargetColumnSelector
            columns={columns}
            nRows={nRows}
            value={target}
            onChange={handleTargetChange}
            disabled={formDisabled}
          />
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- features ---------------- */}
        <div className="mb-10">
          <h2 className="font-serif text-h3 text-ink mb-4">
            Choose the features
          </h2>
          <FeatureColumnSelector
            columns={columns}
            nRows={nRows}
            target={target}
            selected={features}
            onChange={setFeatures}
            disabled={formDisabled}
          />
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- task type ---------------- */}
        <div className="mb-10">
          <h2 className="font-serif text-h3 text-ink mb-4">Task type</h2>
          <div className="flex flex-wrap gap-2">
            {TASK_CHOICES.map((choice) => (
              <button
                key={choice.value}
                type="button"
                disabled={formDisabled}
                onClick={() => setRequestedTask(choice.value)}
                className={`rounded-control px-5 py-3 text-base font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                  requestedTask === choice.value
                    ? "bg-primary text-white"
                    : "bg-rule text-ink hover:bg-rule-strong"
                }`}
              >
                {choice.label}
              </button>
            ))}
          </div>

          {requestedTask === "auto" ? (
            <p className="mt-3 text-sm text-ink-muted">
              {targetProfile === null ? (
                <>Pick a target column and Furina will infer the task from it.</>
              ) : (
                <span className="font-semibold text-ink">
                  {explainAutoTask(targetProfile)}
                </span>
              )}
            </p>
          ) : (
            <p className="mt-3 text-sm font-semibold text-ink">
              Forcing{" "}
              {requestedTask === "classification"
                ? "Classification"
                : "Regression"}
              . Your choice is never overridden.
            </p>
          )}

          <p className="mt-2 text-sm text-ink-muted">
            Auto reads the target's values, so an integer column like 0/1/2 is
            the case it most often gets wrong: it can look like three classes or
            like a small number. If Auto guessed wrong, say so explicitly here —
            the backend always honours an explicit choice.
          </p>
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- seed ---------------- */}
        <div className="mb-10">
          <h2 className="font-serif text-h3 text-ink mb-4">
            Set random seed
          </h2>
          <div className="flex items-center space-x-4">
            <input
              type="number"
              id="random-seed"
              value={seed}
              disabled={formDisabled}
              placeholder="Leave empty for a different split every run"
              onChange={(event) => setSeed(event.target.value)}
              className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-60"
            />
            <button
              type="button"
              disabled={formDisabled}
              className="px-4 py-3 bg-rule hover:bg-rule-strong rounded-control text-ink font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() =>
                setSeed(String(Math.floor(Math.random() * 100000)))
              }
            >
              Random
            </button>
          </div>
          <p className="text-sm text-ink-muted mt-2">
            {seed.trim() === ""
              ? "No seed — the split and any randomised estimator will differ on every run."
              : `Seed ${Math.trunc(Number(seed)) || 0} — the same seed reproduces exactly the same run.`}
          </p>
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- algorithm ---------------- */}
        <div className="mb-10">
          <label
            htmlFor="model-select"
            className="block font-serif text-h3 text-ink mb-4"
          >
            Choose model
          </label>

          {algorithmsLoading && (
            <p className="text-sm text-ink-muted">Loading the algorithm list…</p>
          )}

          {algorithmsError !== null && (
            <div className="rounded-control border-l-4 border-state-bad bg-red-50 p-4">
              <p className="font-semibold text-state-bad">
                ✕ Could not load the algorithms
              </p>
              <p className="mt-1 text-sm text-ink">{algorithmsError}</p>
            </div>
          )}

          {!algorithmsLoading && algorithmsError === null && (
            <>
              <select
                id="model-select"
                value={availableAlgorithms.length === 0 ? "" : algorithm}
                disabled={formDisabled || availableAlgorithms.length === 0}
                onChange={(event) => handleAlgorithmChange(event.target.value)}
                className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-60"
              >
                {availableAlgorithms.length === 0 && (
                  <option value="">No algorithm fits this task</option>
                )}
                {familyGroups.map(([family, specs]) => (
                  <optgroup key={family} label={family}>
                    {specs.map((spec) => (
                      <option key={spec.key} value={spec.key}>
                        {spec.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>

              <p className="mt-2 text-sm text-ink-muted">
                {resolvedTask === null ? (
                  <>Showing all {algorithms.length} algorithms — pick a target to narrow this list.</>
                ) : (
                  <>
                    Showing the {availableAlgorithms.length} of{" "}
                    {algorithms.length} algorithms that support{" "}
                    {resolvedTask === "classification"
                      ? "classification"
                      : "regression"}
                    .
                  </>
                )}
              </p>

              {selectedSpec !== null && (
                <div className="mt-4">
                  <p className="text-sm text-ink">{selectedSpec.description}</p>
                  <p className="mt-2 text-sm text-ink-muted">
                    {selectedSpec.supports_feature_importance
                      ? "Reports feature importances"
                      : "No feature importances"}{" "}
                    ·{" "}
                    {selectedSpec.supports_proba
                      ? "Reports class probabilities"
                      : "No class probabilities"}
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* ---------------- hyperparameters ---------------- */}
        <div className="mb-10">
          <h2 className="font-serif text-h3 text-ink mb-4">
            Hyperparameters
          </h2>
          {selectedSpec === null ? (
            <p className="text-small text-ink-muted max-w-measure">
              Choose an algorithm to see the settings it accepts.
            </p>
          ) : (
            <HyperparameterForm
              key={selectedSpec.key}
              algorithmKey={selectedSpec.key}
              specs={selectedSpec.hyperparameters}
              value={hyperparams}
              onChange={setHyperparams}
              disabled={formDisabled}
            />
          )}
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- options ---------------- */}
        <div className="mb-10">
          <h2 className="font-serif text-h3 text-ink mb-4">
            Training options
          </h2>

          <div className="space-y-4">
            <label
              className={`flex items-start gap-3 rounded-control border border-rule bg-paper-sunken p-4 ${
                resolvedTask === "classification" && !formDisabled
                  ? "cursor-pointer"
                  : "cursor-not-allowed opacity-70"
              }`}
            >
              <input
                type="checkbox"
                checked={stratify}
                disabled={formDisabled || resolvedTask !== "classification"}
                onChange={(event) => setStratify(event.target.checked)}
                className="mt-0.5 h-5 w-5 flex-shrink-0 rounded-control border-rule text-accent focus:ring-accent"
              />
              <span>
                <span className="block font-medium text-ink">
                  Stratify the split
                </span>
                <span className="block text-sm text-ink-muted">
                  {resolvedTask === "classification"
                    ? "Keeps each class in the same proportion in the training and test halves. Turned off automatically if a class has fewer than two rows."
                    : "Only applies to classification — a regression target has no classes to keep in proportion, so the backend ignores this."}
                </span>
              </span>
            </label>

            <label
              className={`flex items-start gap-3 rounded-control border border-rule bg-paper-sunken p-4 ${
                formDisabled ? "cursor-not-allowed opacity-70" : "cursor-pointer"
              }`}
            >
              <input
                type="checkbox"
                checked={scaleFeatures}
                disabled={formDisabled}
                onChange={(event) => setScaleFeatures(event.target.checked)}
                className="mt-0.5 h-5 w-5 flex-shrink-0 rounded-control border-rule text-accent focus:ring-accent"
              />
              <span>
                <span className="block font-medium text-ink">
                  Standardise numeric features
                </span>
                <span className="block text-sm text-ink-muted">
                  Centres and scales the numeric columns inside the pipeline.
                  Distance- and gradient-based models (SVM, KNN, MLP, logistic
                  regression) need this; trees and forests do not care.
                </span>
              </span>
            </label>

            <div>
              <label
                htmlFor="cv-folds"
                className="block font-medium text-ink mb-1"
              >
                Cross-validation
              </label>
              <select
                id="cv-folds"
                value={cvFolds === null ? "" : String(cvFolds)}
                disabled={formDisabled}
                onChange={(event) =>
                  setCvFolds(
                    event.target.value === "" ? null : Number(event.target.value)
                  )
                }
                className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-60"
              >
                <option value="">Off — a single train/test split</option>
                {CV_FOLD_CHOICES.map((folds) => (
                  <option key={folds} value={String(folds)}>
                    {folds}-fold cross-validation
                  </option>
                ))}
              </select>
              <p className="mt-1 text-sm text-ink-muted">
                {cvFolds === null
                  ? "One split gives one score. Cross-validation refits the model on several splits so you can see how stable that score is."
                  : `Refits the model ${cvFolds} times on ${cvFolds} different splits and reports the mean and spread — slower, but far more trustworthy.`}
              </p>
            </div>

            <div>
              <label
                htmlFor="run-name"
                className="block font-medium text-ink mb-1"
              >
                Run name <span className="text-ink-muted">(optional)</span>
              </label>
              <input
                id="run-name"
                type="text"
                value={runName}
                disabled={formDisabled}
                placeholder="e.g. forest, 300 trees, no ID columns"
                onChange={(event) => setRunName(event.target.value)}
                className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-60"
              />
              <p className="mt-1 text-sm text-ink-muted">
                Names make the Recent runs list below readable when you compare
                several models on this dataset.
              </p>
            </div>
          </div>
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- train ---------------- */}
        <div className="space-y-4">
          {validationError !== null && !training && (
            <p className="text-sm font-medium text-state-warn">
              {WARNING_MARK} {validationError}
            </p>
          )}

          <button
            type="button"
            onClick={() => void handleTrain()}
            disabled={training || validationError !== null}
            className="btn btn-primary text-body py-3 w-full"
          >
            {training ? "Training…" : "Train the model"}
          </button>

          {training && (
            <div>
              <div
                className="w-full bg-paper-sunken rounded-full h-8 overflow-hidden"
                role="progressbar"
                aria-label="Training in progress"
              >
                <div className="furina-indeterminate-bar h-8 rounded-full bg-accent" />
              </div>
              <p className="mt-2 text-center text-sm text-ink-muted">
                Fitting {selectedSpec?.label ?? "the model"} on{" "}
                {estimatedTrain.toLocaleString()} rows —{" "}
                <span className="font-semibold text-ink">
                  {(elapsedMs / 1000).toFixed(1)}s
                </span>{" "}
                elapsed. The real duration depends on the data and the
                algorithm, so there is no percentage to show.
              </p>
            </div>
          )}

          {trainError !== null && (
            <div className="rounded-control border-l-4 border-state-bad bg-red-50 p-4">
              <div className="flex items-start justify-between gap-4">
                <p className="font-semibold text-state-bad">✕ Training failed</p>
                <button
                  type="button"
                  onClick={() => setTrainError(null)}
                  className="text-sm font-medium text-ink-muted hover:text-ink"
                >
                  Dismiss
                </button>
              </div>
              <p className="mt-1 text-sm text-ink">{trainError.detail}</p>
              {trainError.errors && (
                <ul className="mt-2 space-y-1">
                  {Object.entries(trainError.errors).map(([field, messages]) => (
                    <li key={field} className="text-sm text-ink">
                      <span className="font-semibold">{field}</span>:{" "}
                      {messages.join(" ")}
                    </li>
                  ))}
                </ul>
              )}
              <p className="mt-2 text-sm text-ink-muted">
                Your settings above are untouched — adjust them and train again.
              </p>
            </div>
          )}

          {result !== null && (
            <div className="mt-4">
              {result.status === "failed" ? (
                <div className="rounded-control border-l-4 border-state-bad bg-red-50 p-4">
                  <p className="font-semibold text-state-bad">
                    ✕ The run finished with an error
                  </p>
                  <p className="mt-1 text-sm text-ink">
                    {result.error ?? "No detail was returned."}
                  </p>
                </div>
              ) : (
                <>
                  <p className="font-semibold text-state-good">
                    ✓ Trained {result.algorithm_label} on {result.target_column}
                  </p>

                  <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-4">
                    <div className="rounded-control bg-paper-raised p-4">
                      <p className="font-serif text-h2 text-ink tabular">
                        {formatMetric(result.primary_metric)}
                      </p>
                      <p className="text-sm text-ink-muted">
                        {metricLabel(result.primary_metric_name)} (test)
                      </p>
                    </div>
                    <div className="rounded-control bg-paper-raised p-4">
                      <p className="font-serif text-h2 text-ink tabular">
                        {formatDuration(result.duration_ms)}
                      </p>
                      <p className="text-sm text-ink-muted">fit + evaluate</p>
                    </div>
                    <div className="rounded-control bg-paper-raised p-4">
                      <p className="font-serif text-h2 text-ink tabular">
                        {result.n_train.toLocaleString()}
                      </p>
                      <p className="text-sm text-ink-muted">rows trained on</p>
                    </div>
                    <div className="rounded-control bg-paper-raised p-4">
                      <p className="font-serif text-h2 text-ink tabular">
                        {result.n_test.toLocaleString()}
                      </p>
                      <p className="text-sm text-ink-muted">rows tested on</p>
                    </div>
                  </div>

                  {result.cv && (
                    <p className="mt-4 text-sm text-ink-muted">
                      {result.cv.folds}-fold cross-validation (
                      {result.cv.scoring}):{" "}
                      <span className="font-semibold text-ink">
                        {formatMetric(result.cv.mean)}
                      </span>{" "}
                      ± {formatMetric(result.cv.std)} across{" "}
                      {result.cv.scores.length} folds.
                    </p>
                  )}

                  {result.warnings.length > 0 && (
                    <div className="mt-4 rounded-control border-l-4 border-state-warn bg-amber-50 p-4">
                      <p className="font-semibold text-state-warn">
                        {WARNING_MARK} {result.warnings.length === 1 ? "Warning" : "Warnings"}
                      </p>
                      <ul className="mt-1 list-disc space-y-1 pl-5">
                        {result.warnings.map((warning) => (
                          <li key={warning} className="text-sm text-ink">
                            {warning}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => goToTool("Model Evaluation")}
                    className="btn btn-primary text-body py-3 w-full mt-6"
                  >
                    View full evaluation
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        <hr className="my-6 border-t border-rule" />

        {/* ---------------- recent runs ---------------- */}
        <div>
          <h2 className="font-serif text-h3 text-ink mb-4">
            Recent runs on this dataset
          </h2>

          {runsError !== null && (
            <div className="mb-4 rounded-control border-l-4 border-state-bad bg-red-50 p-4">
              <div className="flex items-start justify-between gap-4">
                <p className="font-semibold text-state-bad">
                  ✕ Could not load the run history
                </p>
                <button
                  type="button"
                  onClick={() => setRunsError(null)}
                  className="text-sm font-medium text-ink-muted hover:text-ink"
                >
                  Dismiss
                </button>
              </div>
              <p className="mt-1 text-sm text-ink">{runsError}</p>
            </div>
          )}

          {runsLoading && runs.length === 0 ? (
            <p className="text-sm text-ink-muted">Loading previous runs…</p>
          ) : runs.length === 0 ? (
            <p className="text-small text-ink-muted max-w-measure">
              No models have been trained on {dataset.name} yet. Train one above
              and it will appear here, so you can compare algorithms on the same
              data.
            </p>
          ) : (
            <ul className="space-y-2">
              {runs.map((summary) => (
                <li key={summary.id}>
                  <button
                    type="button"
                    disabled={openingRunId !== null || training}
                    onClick={() => void handleOpenRun(summary.id)}
                    className="flex w-full flex-wrap items-center gap-x-4 gap-y-1 rounded-control border border-rule bg-paper-sunken p-4 text-left transition-colors hover:bg-paper-sunken disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <span className="font-semibold text-ink">
                      {summary.name.trim() === ""
                        ? summary.algorithm_label
                        : summary.name}
                    </span>
                    <span className="text-sm text-ink-muted">
                      {summary.task_type === "classification"
                        ? "Classification"
                        : "Regression"}{" "}
                      · target {summary.target_column}
                    </span>
                    <span
                      className={`text-sm font-semibold ${
                        summary.status === "success"
                          ? "text-state-good"
                          : "text-state-bad"
                      }`}
                    >
                      {summary.status === "success" ? "✓ success" : "✕ failed"}
                    </span>
                    <span className="text-sm text-ink">
                      {metricLabel(summary.primary_metric_name)}{" "}
                      <span className="font-semibold">
                        {formatMetric(summary.primary_metric)}
                      </span>
                    </span>
                    <span className="ml-auto text-sm text-ink-muted">
                      {new Date(summary.created_at).toLocaleString()}
                    </span>
                    <span className="text-sm font-semibold text-secondary">
                      {openingRunId === summary.id ? "Opening…" : "Open →"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default DataTraining;
