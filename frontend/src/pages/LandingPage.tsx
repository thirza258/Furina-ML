/**
 * The front door.
 *
 * Someone opening this app has a CSV and a question, and no idea what the
 * three tools in the sidebar have to do with each other. This page explains
 * the pipeline in the order they will walk it, says plainly what the backend
 * does with their data, and hands them the one button that starts it. When
 * they already have work in progress it gets out of the way and offers to
 * take them back to it.
 */
import { useEffect, useState } from "react";

import services from "../services/services";
import type { AlgorithmSpec, ApiError } from "../interface/MlTypes";
import { useWorkspace } from "../context/workspaceContext";

/** Must match the strings App.tsx switches on. */
const TOOL_PREPROCESSING = "Data Preprocessing";
const TOOL_TRAINING = "Data Training";
const TOOL_EVALUATION = "Model Evaluation";
const TOOL_ABOUT = "About Us";

/**
 * Service methods throw an already-normalized ApiError, so read `detail`
 * straight off the caught value rather than re-normalizing it.
 */
function errorDetail(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as ApiError).detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Something went wrong.";
}

interface StepCard {
  index: number;
  tool: string;
  title: string;
  body: string;
  action: string;
}

const STEPS: StepCard[] = [
  {
    index: 1,
    tool: TOOL_PREPROCESSING,
    title: "Data Preprocessing",
    body:
      "Upload a CSV and the backend reads it with pandas and profiles every " +
      "column for real: its type, how many values are missing, how many are " +
      "distinct, its range, and whether it would make a sensible target. From " +
      "there you can clean it — drop duplicates and dead columns, fill gaps, " +
      "tidy text, trim outliers, rescale or encode — and each step reports " +
      "exactly how many rows and columns it changed.",
    action: "Upload a CSV",
  },
  {
    index: 2,
    tool: TOOL_TRAINING,
    title: "Data Training",
    body:
      "Choose the column you want to predict, the columns to predict it from, " +
      "and one of the scikit-learn algorithms. Tune the hyperparameters that " +
      "matter, set the train/test split and the random seed, and optionally " +
      "add cross-validation. Training happens on the server against real " +
      "scikit-learn — there is no simulation anywhere in this app.",
    action: "Train a model",
  },
  {
    index: 3,
    tool: TOOL_EVALUATION,
    title: "Model Evaluation",
    body:
      "See how the model did on the rows it was never trained on: the " +
      "headline metrics, the full confusion matrix or actual-versus-predicted " +
      "scatter, per-class precision and recall, and which features carried " +
      "the prediction. Then type in a new row and get a live prediction back " +
      "from the fitted model.",
    action: "Evaluate the model",
  },
];

/**
 * Shown only when the algorithms endpoint cannot be reached, and labelled as
 * such in the UI. This is the documented catalogue from the API contract, not
 * stand-in data: the live list always wins when the server answers.
 */
const OFFLINE_FAMILIES: { family: string; labels: string[] }[] = [
  {
    family: "Linear models",
    labels: [
      "Logistic Regression",
      "Ridge Classifier",
      "Linear Regression",
      "Ridge Regression",
      "Lasso Regression",
      "Elastic Net",
      "Stochastic Gradient Descent",
    ],
  },
  {
    family: "Support vector machines",
    labels: ["Support Vector Machine (SVC / SVR)"],
  },
  {
    family: "Trees and ensembles",
    labels: [
      "Decision Tree",
      "Random Forest",
      "Extra Trees",
      "Gradient Boosting",
      "Histogram Gradient Boosting",
      "AdaBoost",
    ],
  },
  { family: "Nearest neighbours", labels: ["K-Nearest Neighbours"] },
  { family: "Naive Bayes", labels: ["Gaussian Naive Bayes"] },
  { family: "Neural networks", labels: ["Neural Network (MLP)"] },
];

function groupByFamily(
  algorithms: AlgorithmSpec[]
): { family: string; labels: string[] }[] {
  const families = new Map<string, string[]>();
  for (const algorithm of algorithms) {
    const bucket = families.get(algorithm.family);
    if (bucket) bucket.push(algorithm.label);
    else families.set(algorithm.family, [algorithm.label]);
  }
  return Array.from(families, ([family, labels]) => ({ family, labels }));
}

const LandingPage = () => {
  const { dataset, run, restoring, goToTool } = useWorkspace();

  const [algorithms, setAlgorithms] = useState<AlgorithmSpec[] | null>(null);
  const [loadingAlgorithms, setLoadingAlgorithms] = useState<boolean>(true);
  const [algorithmError, setAlgorithmError] = useState<string | null>(null);
  const [errorDismissed, setErrorDismissed] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const list = await services.listAlgorithms();
        if (!cancelled) {
          setAlgorithms(list);
          setAlgorithmError(null);
        }
      } catch (error) {
        if (!cancelled) setAlgorithmError(errorDetail(error));
      } finally {
        if (!cancelled) setLoadingAlgorithms(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // An empty successful response and an unreachable server are different
  // failures and get different states: the first is a backend problem worth
  // saying out loud, the second falls back to the documented catalogue.
  const liveFamilies =
    algorithms && algorithms.length > 0 ? groupByFamily(algorithms) : null;
  const serverReturnedNone = algorithms !== null && algorithms.length === 0;
  const families = liveFamilies ?? OFFLINE_FAMILIES;
  const usingOfflineList = liveFamilies === null && !serverReturnedNone;
  const hasWork = dataset !== null || run !== null;

  return (
    <div className="bg-gray-100 min-h-screen p-8">
      <h1 className="text-4xl font-bold text-secondary mb-8">Furina ML</h1>

      {/* --------------------------------------------------------------- */}
      {/* Intro, or a way back into work already in progress.              */}
      {/* --------------------------------------------------------------- */}
      {restoring ? (
        <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
          <p className="text-ink-muted">Restoring your last workspace…</p>
        </div>
      ) : hasWork ? (
        <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Resume where you left off
          </h2>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="border border-gray-300 rounded-md p-4">
              <p className="text-xs uppercase tracking-wide text-ink-muted mb-1">
                Active dataset
              </p>
              {dataset ? (
                <>
                  <p className="text-lg font-semibold text-ink">
                    {dataset.name}
                  </p>
                  <p className="text-sm text-ink-muted">
                    {dataset.n_rows.toLocaleString()} rows ×{" "}
                    {dataset.n_cols} columns · {dataset.source}
                  </p>
                </>
              ) : (
                <p className="text-sm text-ink-muted">
                  No dataset loaded. Upload a CSV to begin.
                </p>
              )}
            </div>

            <div className="border border-gray-300 rounded-md p-4">
              <p className="text-xs uppercase tracking-wide text-ink-muted mb-1">
                Last training run
              </p>
              {run ? (
                <>
                  <p className="text-lg font-semibold text-ink">
                    {run.algorithm_label}
                  </p>
                  {run.status === "failed" ? (
                    <p className="text-sm font-semibold text-state-bad">
                      ▲ Failed — {run.error ?? "no reason reported"}
                    </p>
                  ) : (
                    <p className="text-sm text-ink-muted">
                      {run.task_type} on “{run.target_column}” ·{" "}
                      {run.primary_metric_name}{" "}
                      {run.primary_metric === null
                        ? "—"
                        : run.primary_metric.toFixed(3)}
                    </p>
                  )}
                </>
              ) : (
                <p className="text-sm text-ink-muted">
                  Nothing trained yet on this dataset.
                </p>
              )}
            </div>
          </div>

          <hr className="my-6 border-t border-gray-300" />

          <div className="grid gap-4 md:grid-cols-3">
            <button
              type="button"
              onClick={() => goToTool(TOOL_PREPROCESSING)}
              className="w-full bg-gray-300 text-gray-800 py-3 px-4 rounded-md text-lg font-semibold hover:bg-gray-400 transition-colors"
            >
              Back to the data
            </button>
            <button
              type="button"
              onClick={() => goToTool(TOOL_TRAINING)}
              className={
                run
                  ? "w-full bg-gray-300 text-gray-800 py-3 px-4 rounded-md text-lg font-semibold hover:bg-gray-400 transition-colors"
                  : "w-full bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
              }
            >
              {run ? "Train another model" : "Train a model"}
            </button>
            <button
              type="button"
              onClick={() => goToTool(TOOL_EVALUATION)}
              className={
                run
                  ? "w-full bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
                  : "w-full bg-gray-300 text-gray-800 py-3 px-4 rounded-md text-lg font-semibold hover:bg-gray-400 transition-colors"
              }
            >
              See the evaluation
            </button>
          </div>
        </div>
      ) : (
        <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Train a real scikit-learn model on your own CSV
          </h2>
          <p className="text-ink mb-3">
            Furina ML turns a spreadsheet into a working machine learning model
            without you writing any Python. You upload a CSV, look at what is
            actually in it, clean up what needs cleaning, pick the column you
            want to predict and an algorithm to predict it with, and get back
            an honest evaluation of how well that model does on data it has
            never seen.
          </p>
          <p className="text-ink mb-6">
            Everything here runs against a real Django and scikit-learn backend.
            The profiles, the metrics, the confusion matrix and the predictions
            are all computed from your data — nothing on these pages is a
            sample or a simulation.
          </p>

          <button
            type="button"
            onClick={() => goToTool(TOOL_PREPROCESSING)}
            className="w-full bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
          >
            Start by uploading a CSV
          </button>
        </div>
      )}

      {/* --------------------------------------------------------------- */}
      {/* The three-step flow.                                             */}
      {/* --------------------------------------------------------------- */}
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        Three steps, in order
      </h2>
      <div className="grid gap-6 lg:grid-cols-3 mb-6">
        {STEPS.map((step) => (
          <div
            key={step.tool}
            className="w-full bg-white p-8 rounded-lg shadow-lg flex flex-col"
          >
            <div className="flex items-center gap-3 mb-4">
              <span
                aria-hidden="true"
                className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary text-white flex items-center justify-center font-bold"
              >
                {step.index}
              </span>
              <h3 className="text-lg font-semibold text-gray-800">
                {step.title}
              </h3>
            </div>
            <p className="text-sm text-ink mb-6 flex-grow">{step.body}</p>
            <button
              type="button"
              onClick={() => goToTool(step.tool)}
              className="w-full bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
            >
              {step.action}
            </button>
          </div>
        ))}
      </div>

      {/* --------------------------------------------------------------- */}
      {/* What the backend actually does.                                  */}
      {/* --------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          What happens under the hood
        </h2>
        <p className="text-ink mb-3">
          When you press Train, the backend assembles a single scikit-learn{" "}
          <code className="bg-gray-200 px-1 rounded">Pipeline</code>. Numeric
          columns get their gaps filled with the column median and, if you leave
          scaling on, are standardised. Categorical, text, boolean and date
          columns get their gaps filled with the most frequent value and are
          then one-hot encoded, with unknown categories ignored so a value the
          model has never seen cannot crash a prediction. The estimator you
          chose is the last step of that same pipeline.
        </p>
        <p className="text-ink">
          Building it as one pipeline is what keeps the evaluation honest: every
          median, every scale factor and every category list is learned from the
          training rows alone and then applied to the test rows, so nothing
          about the test set leaks into the model that is about to be scored on
          it.
        </p>
        <hr className="my-6 border-t border-gray-300" />
        <button
          type="button"
          onClick={() => goToTool(TOOL_ABOUT)}
          className="w-full bg-gray-300 text-gray-800 py-3 px-4 rounded-md text-lg font-semibold hover:bg-gray-400 transition-colors"
        >
          Read the full pipeline and API reference
        </button>
      </div>

      {/* --------------------------------------------------------------- */}
      {/* Algorithm families.                                              */}
      {/* --------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Algorithms you can train
        </h2>

        {loadingAlgorithms && (
          <p className="text-ink-muted mb-4">Loading the algorithm catalogue…</p>
        )}

        {algorithmError !== null && !errorDismissed && (
          <div
            role="alert"
            className="mb-4 border-l-4 border-state-warn bg-gray-100 p-4 rounded-md"
          >
            <div className="flex items-start justify-between gap-4">
              <p className="font-semibold text-state-warn">
                ▲ Could not load the live algorithm list
              </p>
              <button
                type="button"
                onClick={() => setErrorDismissed(true)}
                aria-label="Dismiss this message"
                className="flex-shrink-0 font-bold text-ink-muted hover:text-ink px-2"
              >
                ✕
              </button>
            </div>
            <p className="text-sm text-ink mt-1">{algorithmError}</p>
            <p className="text-sm text-ink-muted mt-1">
              The families below are the documented catalogue for offline
              reference. Start the API to see the list this server actually
              offers, along with each algorithm&rsquo;s hyperparameters.
            </p>
          </div>
        )}

        {!loadingAlgorithms && serverReturnedNone && (
          <p className="text-ink-muted">
            The server answered, but listed no algorithms at all. That points at
            a backend registry problem rather than anything you did — check the
            Django logs, then reload this page.
          </p>
        )}

        {!loadingAlgorithms && !serverReturnedNone && (
          <>
            <p className="text-ink mb-4">
              {usingOfflineList
                ? "Offline reference — the training page reads the live list from the server."
                : `${algorithms?.length ?? 0} algorithms available, grouped by family. The training page shows each one's hyperparameters and which task types it supports.`}
            </p>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {families.map((group) => (
                <div
                  key={group.family}
                  className="border border-gray-300 rounded-md p-4"
                >
                  <h3 className="font-semibold text-ink mb-2">
                    {group.family}
                  </h3>
                  <ul className="text-sm text-ink-muted list-disc pl-5">
                    {group.labels.map((label) => (
                      <li key={label}>{label}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default LandingPage;
