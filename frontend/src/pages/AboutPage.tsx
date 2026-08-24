/**
 * The reference page.
 *
 * Everything a user needs in order to trust — or argue with — the numbers the
 * app shows them: exactly what the backend builds, how it decides whether your
 * problem is classification or regression, what each metric on the evaluation
 * page means, and the full HTTP surface if they would rather drive it
 * themselves. The metric glossary is rendered from the same help catalogue the
 * tooltips use, so the two can never drift apart.
 */
import { API_BASE_URL } from "../services/services";
import { helpFor } from "../explainer/help";
import type { HelpEntry } from "../explainer/help";
import { useWorkspace } from "../context/workspaceContext";

const TOOL_PREPROCESSING = "Data Preprocessing";

/** The base already ends in `/api`; Swagger lives one segment further down. */
const DOCS_URL = `${API_BASE_URL.replace(/\/+$/, "")}/docs/`;

interface Endpoint {
  n: number;
  method: string;
  path: string;
  body: string;
  response: string;
}

/** Section 2 of the API contract, in full. */
const ENDPOINTS: Endpoint[] = [
  {
    n: 1,
    method: "GET",
    path: "/api/health/",
    body: "—",
    response: "{status, sklearn, pandas, numpy}",
  },
  {
    n: 2,
    method: "POST",
    path: "/api/datasets/upload/",
    body: "multipart: file (.csv/.tsv/.txt), name?, delimiter?",
    response: "201 DatasetDetail",
  },
  {
    n: 3,
    method: "GET",
    path: "/api/datasets/",
    body: "—",
    response: "200 [DatasetSummary], newest first",
  },
  {
    n: 4,
    method: "GET",
    path: "/api/datasets/<uuid>/",
    body: "?preview_rows=N (default 10, max 200)",
    response: "200 DatasetDetail",
  },
  {
    n: 5,
    method: "DELETE",
    path: "/api/datasets/<uuid>/",
    body: "—",
    response: "204, no body",
  },
  {
    n: 6,
    method: "GET",
    path: "/api/datasets/<uuid>/download/",
    body: "—",
    response: "200 text/csv attachment",
  },
  {
    n: 7,
    method: "POST",
    path: "/api/datasets/<uuid>/preprocess/",
    body: "PreprocessRequest {steps[], dry_run?, name?}",
    response: "200 {dataset, report}",
  },
  {
    n: 8,
    method: "GET",
    path: "/api/algorithms/",
    body: "?task=classification|regression (optional)",
    response: "200 [AlgorithmSpec]",
  },
  {
    n: 9,
    method: "POST",
    path: "/api/train/",
    body: "TrainRequest",
    response: "201 TrainingRunDetail",
  },
  {
    n: 10,
    method: "GET",
    path: "/api/runs/",
    body: "?dataset_id=<uuid> (optional)",
    response: "200 [TrainingRunSummary], newest first",
  },
  {
    n: 11,
    method: "GET",
    path: "/api/runs/<uuid>/",
    body: "—",
    response: "200 TrainingRunDetail",
  },
  {
    n: 12,
    method: "DELETE",
    path: "/api/runs/<uuid>/",
    body: "—",
    response: "204, no body",
  },
  {
    n: 13,
    method: "POST",
    path: "/api/runs/<uuid>/predict/",
    body: "{rows: [{column: value}]}",
    response: "200 {predictions: [PredictionResult]}",
  },
  {
    n: 14,
    method: "GET",
    path: "/api/runs/<uuid>/download_model/",
    body: "—",
    response: "200 .joblib attachment",
  },
  {
    n: 15,
    method: "GET",
    path: "/api/runs/<uuid>/report/",
    body: "?format=md|json (default md)",
    response: "200 text/markdown attachment, or JSON",
  },
];

const CLASSIFICATION_METRICS = [
  "accuracy",
  "balanced_accuracy",
  "precision",
  "recall",
  "f1",
  "f2",
  "roc_auc",
  "log_loss",
  "matthews_corrcoef",
];

const REGRESSION_METRICS = [
  "r2",
  "mae",
  "mse",
  "rmse",
  "mape",
  "explained_variance",
  "max_error",
];

const STACK_FRONTEND: { name: string; version: string; role: string }[] = [
  { name: "React", version: "19", role: "UI, with hooks-only components" },
  { name: "Vite", version: "7", role: "Dev server and production bundler" },
  { name: "TypeScript", version: "5.9", role: "Strict mode, no implicit any" },
  { name: "Tailwind CSS", version: "3", role: "All styling on this page" },
  { name: "MUI", version: "7", role: "Data grid for large table previews" },
  { name: "axios", version: "1.12", role: "The single HTTP layer" },
  { name: "React Router", version: "7", role: "Shell routing" },
];

const STACK_BACKEND: { name: string; version: string; role: string }[] = [
  { name: "Python", version: "3.14", role: "Runtime" },
  { name: "Django", version: "5.2.6", role: "App server, ORM, media storage" },
  {
    name: "Django REST Framework",
    version: "3.16.1",
    role: "The JSON API surface",
  },
  {
    name: "scikit-learn",
    version: "1.9.0",
    role: "Pipelines, estimators, metrics",
  },
  { name: "pandas", version: "3.0.5", role: "CSV reading and profiling" },
  { name: "numpy", version: "2.3.4", role: "Numeric arrays under everything" },
  { name: "scipy", version: "1.18.1", role: "Statistics behind the metrics" },
  { name: "joblib", version: "1.5.3", role: "Serialising the fitted pipeline" },
];

function MetricGlossary({ keys }: { keys: string[] }) {
  return (
    <dl className="divide-y divide-gray-300">
      {keys.map((key) => {
        const entry: HelpEntry | undefined = helpFor(key);
        if (!entry) return null;
        return (
          <div key={key} className="py-3">
            <dt className="font-semibold text-ink">
              {entry.title}{" "}
              <code className="bg-gray-200 px-1 rounded text-sm font-normal text-ink-muted">
                {key}
              </code>
            </dt>
            <dd className="text-sm text-ink mt-1">{entry.short}</dd>
            <dd className="text-sm text-ink-muted mt-1">{entry.long}</dd>
          </div>
        );
      })}
    </dl>
  );
}

const AboutPage = () => {
  const { goToTool } = useWorkspace();

  return (
    <div className="bg-gray-100 min-h-screen p-8">
      <h1 className="text-4xl font-bold text-secondary mb-8">
        About Furina ML
      </h1>

      {/* ---------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          What this project is for
        </h2>
        <p className="text-ink mb-3">
          Furina ML exists to close the gap between having a spreadsheet and
          having a model. The usual first machine learning project stalls long
          before any learning happens — on environments, on dataframe syntax, on
          the difference between fitting and transforming. This app removes all
          of that and leaves the decisions that actually matter: what am I
          predicting, from what, with which algorithm, and is the result any
          good?
        </p>
        <p className="text-ink">
          It is deliberately not a black box. Every number on the evaluation
          page comes from scikit-learn running on your rows on a real server,
          the exact pipeline is described below, and every knob in the interface
          carries an explanation of what it does and when you would want it.
          The aim is that you leave understanding the model, not just possessing
          one.
        </p>
      </div>

      {/* ---------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          The pipeline the backend builds
        </h2>
        <p className="text-ink mb-4">
          Every training run assembles one scikit-learn{" "}
          <code className="bg-gray-200 px-1 rounded">Pipeline</code> with two
          stages: a <code className="bg-gray-200 px-1 rounded">
            ColumnTransformer
          </code>{" "}
          that prepares your columns, and the estimator you selected. Your
          feature columns are sorted into two branches.
        </p>

        <div className="grid gap-4 md:grid-cols-2 mb-4">
          <div className="border border-gray-300 rounded-md p-4">
            <h3 className="font-semibold text-ink mb-2">Numeric columns</h3>
            <ol className="text-sm text-ink-muted list-decimal pl-5 space-y-1">
              <li>
                <code className="bg-gray-200 px-1 rounded">
                  SimpleImputer(strategy=&quot;median&quot;)
                </code>{" "}
                — missing values are filled with the column median, which
                outliers cannot drag around.
              </li>
              <li>
                <code className="bg-gray-200 px-1 rounded">
                  StandardScaler()
                </code>{" "}
                — applied only when &ldquo;scale features&rdquo; is on;
                recentres each column to mean 0, standard deviation 1.
              </li>
            </ol>
          </div>
          <div className="border border-gray-300 rounded-md p-4">
            <h3 className="font-semibold text-ink mb-2">
              Categorical, text, boolean and date columns
            </h3>
            <ol className="text-sm text-ink-muted list-decimal pl-5 space-y-1">
              <li>
                Values are cast to strings, then{" "}
                <code className="bg-gray-200 px-1 rounded">
                  SimpleImputer(strategy=&quot;most_frequent&quot;)
                </code>{" "}
                fills the gaps.
              </li>
              <li>
                <code className="bg-gray-200 px-1 rounded">
                  OneHotEncoder(handle_unknown=&quot;ignore&quot;)
                </code>{" "}
                — one 0/1 column per category. A category the model never saw in
                training becomes all zeros instead of an error at prediction
                time.
              </li>
            </ol>
          </div>
        </div>

        <p className="text-ink mb-3">
          Columns you did not select are dropped (
          <code className="bg-gray-200 px-1 rounded">remainder=&quot;drop&quot;</code>
          ), and the branch for a kind of column you have none of is simply
          absent — an all-numeric dataset builds a one-branch transformer.
        </p>
        <p className="text-ink">
          <strong>Why one pipeline matters.</strong> The train/test split happens
          before anything is fitted, and the whole pipeline is then fitted on
          the training rows only. Every median, every mean and standard
          deviation, and every category list is learned from those rows and
          merely applied to the test rows. Impute or scale the full table by
          hand beforehand and information from the test set silently reaches the
          model, which then scores better than it deserves. Keeping the
          preparation inside the pipeline makes that leak structurally
          impossible — and it means the saved{" "}
          <code className="bg-gray-200 px-1 rounded">.joblib</code> artifact
          contains the preparation too, so predictions on new raw rows go
          through exactly the same transformations.
        </p>
      </div>

      {/* ---------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          How the task type is decided
        </h2>
        <p className="text-ink mb-3">
          Classification predicts a category; regression predicts a number.
          Leave the task type on <strong>Auto</strong> and the backend inspects
          your target column and applies these rules in order:
        </p>
        <ul className="text-sm text-ink-muted list-disc pl-5 space-y-1 mb-4">
          <li>
            A boolean, string or categorical target →{" "}
            <strong className="text-ink">classification</strong>.
          </li>
          <li>
            A datetime target → <strong className="text-ink">regression</strong>.
          </li>
          <li>
            A numeric target whose values are all whole numbers, with at most 50
            distinct values and no more than the larger of 20 or 5% of your row
            count → <strong className="text-ink">classification</strong>.
          </li>
          <li>
            A numeric target with two or fewer distinct values →{" "}
            <strong className="text-ink">classification</strong>.
          </li>
          <li>
            Any other numeric target →{" "}
            <strong className="text-ink">regression</strong>.
          </li>
        </ul>
        <p className="text-ink">
          <strong>Overriding it.</strong> Choose classification or regression
          explicitly on the training page and your choice always wins — the
          inference never overrides you. Force classification onto a
          high-cardinality numeric target and the run still goes ahead, with a
          warning attached to the result. The one refusal is forcing regression
          onto a non-numeric target, which cannot mean anything and returns a
          400 with an explanation.
        </p>
      </div>

      {/* ---------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          What each metric means
        </h2>
        <h3 className="font-semibold text-ink mb-2">Classification</h3>
        <p className="text-sm text-ink-muted mb-2">
          Precision, recall, F1 and F2 are averaged across classes weighted by
          how many rows each class has. ROC AUC and log loss need class
          probabilities, so they are blank for algorithms that cannot produce
          them and when a class is missing from the test split.
        </p>
        <MetricGlossary keys={CLASSIFICATION_METRICS} />

        <hr className="my-6 border-t border-gray-300" />

        <h3 className="font-semibold text-ink mb-2">Regression</h3>
        <p className="text-sm text-ink-muted mb-2">
          MAE, RMSE and max error are in your target&rsquo;s own units, so read
          them against the size of the thing you are predicting. MAPE is blank
          whenever any true value is zero, because a percentage of zero is
          undefined.
        </p>
        <MetricGlossary keys={REGRESSION_METRICS} />
      </div>

      {/* ---------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Tech stack</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <div>
            <h3 className="font-semibold text-ink mb-2">Frontend</h3>
            <ul className="text-sm divide-y divide-gray-300">
              {STACK_FRONTEND.map((item) => (
                <li key={item.name} className="py-2">
                  <span className="font-semibold text-ink">{item.name}</span>{" "}
                  <span className="text-ink-muted">{item.version}</span>
                  <span className="block text-ink-muted">{item.role}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-ink mb-2">
              Backend (verified environment)
            </h3>
            <ul className="text-sm divide-y divide-gray-300">
              {STACK_BACKEND.map((item) => (
                <li key={item.name} className="py-2">
                  <span className="font-semibold text-ink">{item.name}</span>{" "}
                  <span className="text-ink-muted">{item.version}</span>
                  <span className="block text-ink-muted">{item.role}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <p className="text-sm text-ink-muted mt-4">
          These are the versions this build was developed and verified against.
          The library versions your server is actually running are reported live
          by the health endpoint and shown in the top bar when the backend is
          connected. No charting library is installed — every chart in this app
          is hand-written SVG.
        </p>
      </div>

      {/* ---------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          API reference
        </h2>
        <p className="text-ink mb-4">
          Everything this interface does, it does over these fifteen endpoints —
          so anything you can do by clicking, you can also do with{" "}
          <code className="bg-gray-200 px-1 rounded">curl</code> or from a
          notebook. List responses are bare JSON arrays with no pagination
          envelope. Errors come back as{" "}
          <code className="bg-gray-200 px-1 rounded">
            {'{"detail": "…", "errors": {…}}'}
          </code>{" "}
          and every value in a response is finite JSON — NaN and infinity are
          serialised as null.
        </p>
        <p className="text-sm text-ink-muted mb-4">
          Base URL:{" "}
          <code className="bg-gray-200 px-1 rounded">{API_BASE_URL}</code>{" "}
          (set <code className="bg-gray-200 px-1 rounded">VITE_API_BASE_URL</code>{" "}
          to point elsewhere).
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left border-collapse">
            <caption className="sr-only">
              Furina ML HTTP API endpoints, with request body and success
              response for each.
            </caption>
            <thead>
              <tr className="border-b border-gray-300 text-ink-muted">
                <th scope="col" className="py-2 pr-4 font-semibold">
                  #
                </th>
                <th scope="col" className="py-2 pr-4 font-semibold">
                  Method
                </th>
                <th scope="col" className="py-2 pr-4 font-semibold">
                  Path
                </th>
                <th scope="col" className="py-2 pr-4 font-semibold">
                  Request
                </th>
                <th scope="col" className="py-2 font-semibold">
                  Success response
                </th>
              </tr>
            </thead>
            <tbody>
              {ENDPOINTS.map((endpoint) => (
                <tr
                  key={endpoint.n}
                  className="border-b border-gray-200 hover:bg-gray-100"
                >
                  <td className="py-2 pr-4 text-ink-muted">{endpoint.n}</td>
                  <td className="py-2 pr-4">
                    <span className="font-semibold text-ink">
                      {endpoint.method}
                    </span>
                  </td>
                  <td className="py-2 pr-4 whitespace-nowrap">
                    <code className="text-ink">{endpoint.path}</code>
                  </td>
                  <td className="py-2 pr-4 text-ink-muted">{endpoint.body}</td>
                  <td className="py-2 text-ink-muted">{endpoint.response}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <hr className="my-6 border-t border-gray-300" />

        <p className="text-ink mb-4">
          The backend also publishes an OpenAPI schema with a live Swagger UI —
          you can read every field and fire real requests from the browser.
        </p>
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noreferrer"
          className="block w-full text-center bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
        >
          Open the live API docs at {DOCS_URL}
        </a>
        <p className="text-sm text-ink-muted mt-2">
          This link only works while the Django server is running.
        </p>
      </div>

      {/* ---------------------------------------------------------------- */}
      <div className="w-full bg-white p-8 rounded-lg shadow-lg">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Ready to try it?
        </h2>
        <p className="text-ink mb-6">
          Any CSV with a column worth predicting will do. A few hundred rows is
          plenty to see the whole flow work end to end.
        </p>
        <button
          type="button"
          onClick={() => goToTool(TOOL_PREPROCESSING)}
          className="w-full bg-primary text-white py-3 px-4 rounded-md text-lg font-semibold hover:bg-opacity-90 transition-colors"
        >
          Upload a CSV
        </button>
      </div>
    </div>
  );
};

export default AboutPage;
