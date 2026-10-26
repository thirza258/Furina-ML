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
import { Link } from "react-router-dom";

import services from "../services/services";
import type { AlgorithmSpec, ApiError } from "../interface/MlTypes";
import { useWorkspace } from "../context/workspaceContext";
import {
  DOCUMENTED_FAMILIES,
  groupByFamily,
} from "../data/algorithmCatalogue";
import { usePageSeo } from "../hooks/usePageSeo";

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

const LandingPage = () => {
  const { dataset, run, restoring, goToTool } = useWorkspace();

  usePageSeo({
    title: "Workbench Overview — Machine Learning Pipeline & Algorithms",
    description:
      "Interactive workbench overview: data preprocessing, model training, evaluation, and scikit-learn algorithm catalogue.",
    path: "/app",
  });

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
  const families = liveFamilies ?? DOCUMENTED_FAMILIES;
  const usingOfflineList = liveFamilies === null && !serverReturnedNone;
  const hasWork = dataset !== null || run !== null;

  return (
    <div className="bg-paper min-h-full p-8">
      <h1 className="font-serif text-h1 font-semibold text-ink border-b border-rule pb-3 mb-6">Overview</h1>

      {/* --------------------------------------------------------------- */}
      {/* Intro, or a way back into work already in progress.              */}
      {/* --------------------------------------------------------------- */}
      {restoring ? (
        <div className="panel">
          <p className="text-ink-muted">Restoring your last workspace…</p>
        </div>
      ) : hasWork ? (
        <div className="panel">
          <h2 className="font-serif text-h3 text-ink mb-4">
            Resume where you left off
          </h2>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <p className="text-xs uppercase tracking-wide text-ink-muted mb-1">
                Active dataset
              </p>
              {dataset ? (
                <>
                  <p className="font-serif text-h3 text-ink">
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

            <div>
              <p className="text-xs uppercase tracking-wide text-ink-muted mb-1">
                Last training run
              </p>
              {run ? (
                <>
                  <p className="font-serif text-h3 text-ink">
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

          <hr className="my-6 border-t border-rule" />

          <div className="grid gap-4 md:grid-cols-3">
            <button
              type="button"
              onClick={() => goToTool(TOOL_PREPROCESSING)}
              className="btn btn-secondary w-full"
            >
              Back to the data
            </button>
            <button
              type="button"
              onClick={() => goToTool(TOOL_TRAINING)}
              className={
                run
                  ? "btn btn-secondary text-body py-3 w-full"
                  : "btn btn-primary text-body py-3 w-full"
              }
            >
              {run ? "Train another model" : "Train a model"}
            </button>
            <button
              type="button"
              onClick={() => goToTool(TOOL_EVALUATION)}
              className={
                run
                  ? "btn btn-primary text-body py-3 w-full"
                  : "btn btn-secondary text-body py-3 w-full"
              }
            >
              See the evaluation
            </button>
          </div>
        </div>
      ) : (
        <div className="panel">
          <h2 className="panel-title">Nothing loaded yet</h2>
          <p className="prose-page mb-6">
            The workbench needs a dataset before it can do anything. Upload a
            CSV on the preprocessing page and the rest of the pipeline opens up
            behind it. There is a fuller description of what each step does on
            the <Link to="/">front page</Link>.
          </p>

          <button
            type="button"
            onClick={() => goToTool(TOOL_PREPROCESSING)}
            className="btn btn-primary text-body px-6 py-3"
          >
            Upload a CSV
          </button>
        </div>
      )}

      {/* --------------------------------------------------------------- */}
      {/* Courses are available before a dataset or a trained model exists. */}
      <section className="border-l-2 border-primary pl-5 mb-10">
        <h2 className="font-serif text-h3 mb-2">Learn to train and evaluate ML &amp; AI models</h2>
        <p className="text-ink-muted max-w-measure mb-4">
          Follow courses from tabular ML and base neural models to transformers,
          LLMs, vision-language models, voice and ML system design. Work through guided lessons,
          download Python labs, and review capstone projects while keeping your
          knowledge-check progress in this browser.
        </p>
        <Link to="/app/courses" className="btn btn-secondary">Browse courses</Link>
      </section>

      {/* The three-step flow.                                             */}
      {/* --------------------------------------------------------------- */}
      <h2 className="font-serif text-h3 text-ink mb-4">
        Three steps, in order
      </h2>
      <div className="grid gap-6 lg:grid-cols-3 mb-6">
        {STEPS.map((step) => (
          <div
            key={step.tool}
            className="panel flex flex-col"
          >
            <div className="flex items-center gap-3 mb-4">
              <span
                aria-hidden="true"
                className="flex-shrink-0 w-7 h-7 border border-rule-strong bg-paper-sunken text-ink flex items-center justify-center font-serif text-small font-semibold"
              >
                {step.index}
              </span>
              <h3 className="font-serif text-h3 text-ink">
                {step.title}
              </h3>
            </div>
            <p className="text-sm text-ink mb-6 flex-grow">{step.body}</p>
            <button
              type="button"
              onClick={() => goToTool(step.tool)}
              className="btn btn-primary w-full"
            >
              {step.action}
            </button>
          </div>
        ))}
      </div>

      {/* --------------------------------------------------------------- */}
      {/* What the backend actually does.                                  */}
      {/* --------------------------------------------------------------- */}
      <div className="panel">
        <h2 className="font-serif text-h3 text-ink mb-4">
          What happens under the hood
        </h2>
        <p className="text-ink mb-3">
          When you press Train, the backend assembles a single scikit-learn{" "}
          <code className="bg-paper-sunken px-1 rounded-control">Pipeline</code>. Numeric
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
        <hr className="my-6 border-t border-rule" />
        <button
          type="button"
          onClick={() => goToTool(TOOL_ABOUT)}
          className="btn btn-secondary w-full"
        >
          Read the full pipeline and API reference
        </button>
      </div>

      {/* --------------------------------------------------------------- */}
      {/* Algorithm families.                                              */}
      {/* --------------------------------------------------------------- */}
      <div className="panel">
        <h2 className="font-serif text-h3 text-ink mb-4">
          Algorithms you can train
        </h2>

        {loadingAlgorithms && (
          <p className="text-ink-muted mb-4">Loading the algorithm catalogue…</p>
        )}

        {algorithmError !== null && !errorDismissed && (
          <div
            role="alert"
            className="mb-4 border-l-4 border-state-warn bg-paper p-4 rounded-control"
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
