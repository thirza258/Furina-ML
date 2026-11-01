/**
 * The public front door, served at `/`.
 *
 * This is the one page someone reads before they have uploaded anything, so it
 * is written as a page rather than a pitch: what the app does, in what order,
 * with what library, and what it will not pretend to do. It deliberately sits
 * outside the workbench chrome — no navbar, no sidebar — because none of that
 * furniture means anything until there is a dataset behind it.
 *
 * The estimator table paints immediately from the documented catalogue and is
 * quietly replaced if `/api/algorithms/` answers. A landing page should not
 * show a spinner, and it should not go blank when the backend is down.
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import services from "../services/services";
import { usePageSeo } from "../hooks/usePageSeo";
import {
  DOCUMENTED_COUNT,
  DOCUMENTED_FAMILIES,
  groupByFamily,
  type AlgorithmFamily,
} from "../data/algorithmCatalogue";

interface Stage {
  n: string;
  title: string;
  lead: string;
  points: string[];
}

const STAGES: Stage[] = [
  {
    n: "1",
    title: "Preprocessing",
    lead:
      "A CSV goes in and a description of it comes back. Not a preview — a profile: " +
      "the inferred kind of every column, its dtype, how many values are missing, " +
      "how many are distinct, its range, and whether it could serve as a target.",
    points: [
      "Delimiter detected automatically; UTF-8, UTF-8-BOM and Latin-1 all read",
      "A cleaning recipe composed step by step, with a dry run before anything is written",
      "Deduplicate, fill gaps, tidy text, drop outliers, rescale, encode, PCA, LDA, SelectKBest",
      "Cleaned data is saved as a derived dataset, so the original file is never overwritten",
    ],
  },
  {
    n: "2",
    title: "Training",
    lead:
      "Choose the column to predict and the columns to predict it from. The task type " +
      "is inferred and can be overridden, which matters for integer-coded targets that " +
      "look numeric but are really classes.",
    points: [
      "Hyperparameter forms generated from each estimator's own parameter schema",
      "Adjustable train/test split, random seed, stratification and feature scaling",
      "Optional k-fold cross-validation",
      "Parameters you do not touch are left at scikit-learn's defaults",
    ],
  },
  {
    n: "3",
    title: "Evaluation",
    lead:
      "Every figure is computed on the rows the model was never fitted on, and the " +
      "training-set figures are printed beside them, so overfitting is visible rather " +
      "than implied.",
    points: [
      "Classification: accuracy, balanced accuracy, precision, recall, F1, F2, ROC AUC, log loss, Matthews correlation",
      "Regression: R², MAE, MSE, RMSE, MAPE, explained variance, max error",
      "Full N×N confusion matrix, per-class support, and feature importances over expanded feature names",
      "A prediction form built from the model's own input schema, and the fitted pipeline as a .joblib download",
    ],
  },
  {
    n: "4",
    title: "Dashboard",
    lead:
      "Once there is more than one run, the question stops being \u201chow did this model " +
      "do?\u201d and becomes \u201cwhich of these is best, and did the last change help?\u201d " +
      "The dashboard ranks every run you have trained.",
    points: [
      "Ranked by whichever metric you choose, on the held-out test split",
      "Ranking follows the metric's own direction, so for log loss or MAE the shortest bar leads",
      "Training scores beside test scores, so the overfitting gap is visible across runs, not one at a time",
      "Metrics are only compared within one task type: accuracy and R\u00b2 are not on the same scale, and the chart says so rather than drawing them together",
    ],
  },
];

const PIPELINE = `Pipeline
└── ColumnTransformer
    ├── numeric      SimpleImputer(median) [→ StandardScaler]
    └── categorical  SimpleImputer(most_frequent) → OneHotEncoder
└── estimator        the algorithm you chose`;

const HomePage = () => {
  usePageSeo({
    title: "Furina ML — Train scikit-learn models on your own CSV",
    description:
      "Upload a CSV, clean it, choose a target column and algorithm, and train a real scikit-learn model in your browser. Explore interactive courses from tabular ML to LLMs.",
    path: "/",
    jsonLd: {
      "@context": "https://schema.org",
      "@type": "WebSite",
      name: "Furina ML",
      url: "https://furina.nevatal.id/",
      description:
        "Train scikit-learn models on your own CSV, with interactive courses from tabular ML to LLMs.",
    },
  });

  // Paint the documented catalogue first, then upgrade in place if the server
  // answers. `live` staying null is not an error state worth reporting here.
  const [families, setFamilies] = useState<AlgorithmFamily[]>(DOCUMENTED_FAMILIES);
  const [count, setCount] = useState<number>(DOCUMENTED_COUNT);
  const [isLive, setIsLive] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const list = await services.listAlgorithms();
        if (!cancelled && list.length > 0) {
          setFamilies(groupByFamily(list));
          setCount(list.length);
          setIsLive(true);
        }
      } catch {
        // The documented catalogue is already on screen and is correct.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-dvh bg-paper text-ink">
      <div className="mx-auto max-w-4xl px-6 sm:px-8">
        {/* ---------------------------------------------------------- */}
        {/* Masthead                                                    */}
        {/* ---------------------------------------------------------- */}
        <header className="pt-14 pb-10 border-b border-rule">
          <div className="flex items-center gap-4 mb-3">
            <img
              src="/logo_furina.png"
              alt="Furina ML Logo"
              className="w-14 h-14 sm:w-16 sm:h-16 object-contain flex-shrink-0"
            />
            <div>
              <p className="field-label">Machine learning workbench</p>
              <h1 className="mt-1 font-serif text-h1 sm:text-display font-semibold">
                Furina ML
              </h1>
            </div>
          </div>
          <p className="mt-4 prose-page text-lead">
            Supervised learning on tabular data, from a CSV file to a fitted
            model, without writing any Python.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3">
            <Link
              to="/app"
              className="btn btn-primary no-underline text-body px-6 py-3"
            >
              Enter the workbench
            </Link>
            <Link to="/app/courses" className="font-sans text-small">
              ML &amp; AI courses
            </Link>
            <Link to="/about" className="font-sans text-small">
              How it works
            </Link>
            <a href="/api/docs/" className="font-sans text-small">
              API documentation
            </a>
          </div>
        </header>

        {/* ---------------------------------------------------------- */}
        {/* Abstract                                                    */}
        {/* ---------------------------------------------------------- */}
        <section className="py-10 border-b border-rule">
          <h2 className="field-label mb-3">Abstract</h2>
          <div className="prose-page">
            <p>
              Furina ML trains real scikit-learn estimators on data you provide.
              You upload a CSV, read a profile of what is actually in it, clean
              what needs cleaning, choose a target column and an algorithm, and
              fit the model on the server. What comes back is an evaluation
              measured on rows the model never saw.
            </p>
            <p>
              Every workspace metric, confusion matrix and prediction is
              computed from the file you uploaded by the estimator you chose.
              Where the application cannot do something, it says so rather than
              filling the gap with a plausible number.
              The courses use clearly labelled teaching examples to explain
              how to train and evaluate your own models.
            </p>
          </div>
        </section>

        {/* ---------------------------------------------------------- */}
        {/* The three stages                                            */}
        {/* ---------------------------------------------------------- */}
        <section className="py-10 border-b border-rule">
          <h2 className="font-serif text-h2 mb-2">The pipeline</h2>
          <p className="prose-page text-ink-muted mb-8">
            Three tools, in the order you walk them &mdash; each needs the
            previous one&rsquo;s output &mdash; and a fourth that looks back
            across everything you have trained.
          </p>

          <ol className="space-y-10">
            {STAGES.map((stage) => (
              <li key={stage.n} className="grid gap-x-6 sm:grid-cols-[3rem_1fr]">
                <div
                  aria-hidden="true"
                  className="font-serif text-h2 text-rule-strong leading-none pt-1"
                >
                  {stage.n}
                </div>
                <div>
                  <h3 className="font-serif text-h3">
                    {stage.n}. {stage.title}
                  </h3>
                  <p className="mt-2 prose-page">{stage.lead}</p>
                  <ul className="mt-4 ml-4 list-disc space-y-1.5 font-sans text-small text-ink-muted marker:text-rule-strong">
                    {stage.points.map((point) => (
                      <li key={point} className="leading-relaxed">
                        {point}
                      </li>
                    ))}
                  </ul>
                </div>
              </li>
            ))}
          </ol>
        </section>

        {/* ---------------------------------------------------------- */}
        {/* Method                                                      */}
        {/* ---------------------------------------------------------- */}
        <section className="py-10 border-b border-rule">
          <h2 className="font-serif text-h2 mb-2">Method</h2>
          <div className="prose-page mb-6">
            <p>
              Every run assembles a single scikit-learn <code>Pipeline</code>.
              Imputation, scaling and encoding all live <em>inside</em> it, so
              they are fitted on the training split alone and the test split
              never leaks into them. The same pipeline is what gets pickled,
              which is why a downloaded model already knows how to preprocess a
              raw input row.
            </p>
          </div>

          <figure className="callout bg-paper-sunken overflow-x-auto">
            <pre className="font-mono text-small text-ink leading-relaxed">
              {PIPELINE}
            </pre>
          </figure>
          <figcaption className="mt-2 font-sans text-caption text-ink-muted">
            Figure 1. The fitted object. Preprocessing is part of the model, not
            a step performed beforehand.
          </figcaption>
        </section>

        {/* ---------------------------------------------------------- */}
        {/* Estimators                                                  */}
        {/* ---------------------------------------------------------- */}
        <section className="py-10 border-b border-rule">
          <h2 className="font-serif text-h2 mb-2">
            Estimators
          </h2>
          <p className="prose-page text-ink-muted mb-6">
            {count} algorithms across classification and regression, including
            XGBoost and LightGBM alongside scikit-learn&rsquo;s own.{" "}
            {isLive
              ? "This list was read from the server just now."
              : "This is the documented catalogue; the server publishes the authoritative list at /api/algorithms/."}
          </p>

          <table className="w-full font-sans text-small border-t border-rule">
            <caption className="sr-only">
              Available estimators, grouped by family
            </caption>
            <tbody>
              {families.map((family) => (
                <tr
                  key={family.family}
                  className="border-b border-rule align-baseline"
                >
                  <th
                    scope="row"
                    className="py-3 pr-6 text-left font-medium text-ink whitespace-nowrap w-1/3"
                  >
                    {family.family}
                  </th>
                  <td className="py-3 text-ink-muted">
                    {family.labels.join(" · ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {/* ---------------------------------------------------------- */}
        {/* Colophon                                                    */}
        {/* ---------------------------------------------------------- */}
        <footer className="py-10">
          <div className="grid gap-8 sm:grid-cols-3">
            <div>
              <h2 className="field-label mb-3">Built on</h2>
              <p className="font-sans text-small text-ink-muted leading-relaxed mb-4">
                Django and Django REST Framework on the server, scikit-learn and
                pandas for the modelling, React and Vite in the browser. The
                neural-network option is scikit-learn&rsquo;s multi-layer
                perceptron, so the whole backend rests on one ML library.
              </p>
              <ul className="font-sans text-small space-y-1.5">
                <li>
                  <Link to="/about">Architecture &amp; pipeline details</Link>
                </li>
                <li>
                  <a href="/api/docs/">Interactive API documentation</a>
                </li>
                <li>
                  <a href="/api/health/">Service health</a>
                </li>
              </ul>
            </div>
            <div>
              <h2 className="field-label mb-3">Workbench Tools</h2>
              <ul className="font-sans text-small space-y-1.5">
                <li>
                  <Link to="/app">Workbench Overview</Link>
                </li>
                <li>
                  <Link to="/app/preprocessing">Data Preprocessing &amp; Profiling</Link>
                </li>
                <li>
                  <Link to="/app/training">Model Training &amp; Tuning</Link>
                </li>
                <li>
                  <Link to="/app/evaluation">Model Evaluation &amp; Metrics</Link>
                </li>
                <li>
                  <Link to="/app/dashboard">Training Runs Dashboard</Link>
                </li>
              </ul>
            </div>
            <div>
              <h2 className="field-label mb-3">Interactive Courses</h2>
              <ul className="font-sans text-small space-y-1.5">
                <li>
                  <Link to="/app/courses">Course Library (All 9 Tracks)</Link>
                </li>
                <li>
                  <Link to="/app/courses/training">Model Training Course</Link>
                </li>
                <li>
                  <Link to="/app/courses/evaluation">Model Evaluation Course</Link>
                </li>
                <li>
                  <Link to="/app/courses/base-models">Base Deep Learning Models</Link>
                </li>
                <li>
                  <Link to="/app/courses/transformers">Transformer Architecture</Link>
                </li>
                <li>
                  <Link to="/app/courses/llms">Large Language Models (LLMs)</Link>
                </li>
                <li>
                  <Link to="/app/courses/vision-language">Vision-Language Models</Link>
                </li>
                <li>
                  <Link to="/app/courses/voice">Voice &amp; Speech Models</Link>
                </li>
                <li>
                  <Link to="/app/courses/ml-system-design">ML System Design</Link>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-10 pt-6 border-t border-rule flex flex-col sm:flex-row items-center justify-between gap-4 font-sans text-caption text-ink-muted">
            <div className="flex items-center gap-2.5">
              <img
                src="/logo_furina.png"
                alt="Furina ML Logo"
                className="w-5 h-5 object-contain"
              />
              <span>Furina ML — MIT licensed. Self-hosted Machine Learning Workbench.</span>
            </div>
            <div className="flex items-center gap-4">
              <Link to="/privacy" className="hover:text-ink transition-colors underline">Privacy Policy</Link>
              <span>•</span>
              <Link to="/terms" className="hover:text-ink transition-colors underline">Terms of Service</Link>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default HomePage;
