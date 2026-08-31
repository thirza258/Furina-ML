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
];

const PIPELINE = `Pipeline
└── ColumnTransformer
    ├── numeric      SimpleImputer(median) [→ StandardScaler]
    └── categorical  SimpleImputer(most_frequent) → OneHotEncoder
└── estimator        the algorithm you chose`;

const HomePage = () => {
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
      {/* The rule across the top is the one piece of ornament on the page. */}
      <div className="border-t-4 border-primary" />

      <div className="mx-auto max-w-4xl px-6 sm:px-8">
        {/* ---------------------------------------------------------- */}
        {/* Masthead                                                    */}
        {/* ---------------------------------------------------------- */}
        <header className="pt-14 pb-10 border-b border-rule">
          <p className="field-label">Machine learning workbench</p>
          <h1 className="mt-3 font-serif text-h1 sm:text-display font-semibold">
            Furina ML
          </h1>
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
              Nothing on these pages is a sample, a mock-up or a simulation.
              Every metric, every confusion matrix and every prediction is
              computed from the file you uploaded by the estimator you chose.
              Where the application cannot do something, it says so rather than
              filling the gap with a plausible number.
            </p>
          </div>
        </section>

        {/* ---------------------------------------------------------- */}
        {/* The three stages                                            */}
        {/* ---------------------------------------------------------- */}
        <section className="py-10 border-b border-rule">
          <h2 className="font-serif text-h2 mb-2">The pipeline</h2>
          <p className="prose-page text-ink-muted mb-8">
            Three tools, in the order you walk them. Each one needs the previous
            one&rsquo;s output.
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
                  <ul className="mt-4 space-y-1.5 font-sans text-small text-ink-muted">
                    {stage.points.map((point) => (
                      <li
                        key={point}
                        className="pl-4 border-l border-rule leading-relaxed"
                      >
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

          <figure className="panel bg-paper-sunken p-5 overflow-x-auto">
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
            {count} algorithms across classification and regression.{" "}
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
          <div className="grid gap-8 sm:grid-cols-2">
            <div>
              <h2 className="field-label mb-3">Built on</h2>
              <p className="font-sans text-small text-ink-muted leading-relaxed">
                Django and Django REST Framework on the server, scikit-learn and
                pandas for the modelling, React and Vite in the browser. The
                neural-network option is scikit-learn&rsquo;s multi-layer
                perceptron, so the whole backend rests on one ML library.
              </p>
            </div>
            <div>
              <h2 className="field-label mb-3">Elsewhere</h2>
              <ul className="font-sans text-small space-y-1.5">
                <li>
                  <Link to="/app">The workbench</Link>
                </li>
                <li>
                  <Link to="/about">How it works, in detail</Link>
                </li>
                <li>
                  <a href="/api/docs/">Interactive API documentation</a>
                </li>
                <li>
                  <a href="/api/health/">Service health</a>
                </li>
              </ul>
            </div>
          </div>

          <p className="mt-10 pt-6 border-t border-rule font-sans text-caption text-ink-muted">
            Furina ML — MIT licensed.
          </p>
        </footer>
      </div>
    </div>
  );
};

export default HomePage;
