/**
 * Deployments: live inference APIs and their health.
 *
 * Two jobs on one page. First, calling a deployed model — a form built from
 * its signature, generated client snippets, and a batch CSV path. Second,
 * watching it: request counts, error rate, and distribution drift against the
 * training data, because a model that was good last month is not
 * automatically good now.
 */
import { useCallback, useEffect, useState } from "react";

import services, { toApiError } from "../services/services";
import type {
  ClientSnippets,
  Deployment,
  DeploymentPrediction,
  DriftReport,
  MonitoringSummary,
} from "../interface/LifecycleTypes";
import { usePageSeo } from "../hooks/usePageSeo";

const DRIFT_TONE: Record<string, string> = {
  stable: "border-state-good",
  moderate: "border-state-warn",
  significant: "border-state-bad",
  unknown: "border-rule",
};

export default function Deployments() {
  const [summaries, setSummaries] = useState<MonitoringSummary[] | null>(null);
  const [selected, setSelected] = useState<Deployment | null>(null);
  const [snippets, setSnippets] = useState<ClientSnippets | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<DeploymentPrediction | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [busy, setBusy] = useState(false);

  usePageSeo({
    title: "Deployments",
    description:
      "Call deployed Furina ML models as APIs, generate client code, batch-score CSVs, and watch request and drift monitoring.",
    path: "/app/deployments",
  });

  const load = useCallback(async () => {
    try {
      const list = await services.monitoringOverview();
      setSummaries(list);
      setError(null);
    } catch (caught) {
      setError(toApiError(caught).detail);
      setSummaries([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const open = async (deployment: Deployment) => {
    setSelected(deployment);
    setPrediction(null);
    setDrift(null);
    setSnippets(null);
    // Prefill the form from the signature's example request.
    const example = deployment.signature?.example_request ?? {};
    setValues(
      Object.fromEntries(
        Object.entries(example).map(([key, value]) => [key, value === null ? "" : String(value)])
      )
    );
    try {
      setSnippets(await services.deploymentSnippets(deployment.slug));
    } catch {
      setSnippets(null);
    }
  };

  const predict = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      const payload: Record<string, unknown> = {};
      for (const [key, raw] of Object.entries(values)) {
        payload[key] = raw === "" ? null : raw;
      }
      setPrediction(await services.predictDeployment(selected.slug, payload));
      await load();
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setBusy(false);
    }
  };

  const runDrift = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      // Compare a batch built from the current form: a single row is enough
      // for a smoke check, and the backend compares it to training stats.
      const rows = [values];
      setDrift(await services.checkDrift(selected.slug, rows));
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setBusy(false);
    }
  };

  const batchUpload = async (file: File) => {
    if (!selected) return;
    setBusy(true);
    try {
      const blob = await services.batchPredictDeployment(selected.slug, file);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${selected.slug}-predictions.csv`;
      link.click();
      window.URL.revokeObjectURL(url);
      await load();
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      <header>
        <h1 className="text-heading">Deployments</h1>
        <p className="text-ink-muted mt-1 max-w-3xl">
          Every deployed model is a live prediction API with a published signature, plus
          monitoring that compares what it sees now against what it was trained on.
        </p>
      </header>

      {error && (
        <div className="border-l-2 border-state-bad pl-3 text-small" role="alert">
          {error}
        </div>
      )}

      <section className="space-y-2">
        <h2 className="text-subheading">Deployed models</h2>
        {summaries === null ? (
          <p className="text-small text-ink-muted" role="status">Loading…</p>
        ) : summaries.length === 0 ? (
          <p className="text-small text-ink-muted">
            Nothing deployed yet. Promote a model to staging or production and deploy it
            from the Model Registry page.
          </p>
        ) : (
          <ul className="divide-y divide-rule border border-rule">
            {summaries.map(({ deployment, total_requests, error_rate }) => (
              <li key={deployment.id}>
                <button
                  type="button"
                  onClick={() => void open(deployment)}
                  className={`w-full text-left px-3 py-2 hover:bg-paper-raised ${
                    selected?.id === deployment.id ? "bg-paper-raised" : ""
                  }`}
                >
                  <span className="flex flex-wrap items-baseline gap-x-3">
                    <span className="font-semibold">{deployment.model_name}</span>
                    <span className="text-small text-ink-muted">v{deployment.model_version}</span>
                    <span className="text-small text-ink-muted">{deployment.slug}</span>
                    <span className="text-small">
                      {total_requests} request{total_requests === 1 ? "" : "s"}
                    </span>
                    {error_rate > 0 && (
                      <span className="text-small text-state-bad">
                        {(error_rate * 100).toFixed(1)}% errors
                      </span>
                    )}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {selected && (
        <section className="border-t border-rule pt-4 space-y-5">
          <h2 className="text-subheading">
            {selected.model_name} v{selected.model_version}
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h3 className="text-small font-semibold">Try a prediction</h3>
              <div className="space-y-2">
                {selected.signature?.features?.map((field) => (
                  <label key={field.name} className="flex flex-col gap-1">
                    <span className="text-small text-ink-muted">
                      {field.name}
                      {field.type === "numeric" && field.minimum !== null && field.maximum !== null
                        ? ` (${field.minimum}–${field.maximum})`
                        : ""}
                      {field.required ? "" : " — optional"}
                    </span>
                    {field.categories ? (
                      <select
                        value={values[field.name] ?? ""}
                        onChange={(event) =>
                          setValues((prev) => ({ ...prev, [field.name]: event.target.value }))
                        }
                        className="border border-rule px-2 py-1.5 bg-paper"
                      >
                        <option value="">(blank)</option>
                        {field.categories.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        value={values[field.name] ?? ""}
                        onChange={(event) =>
                          setValues((prev) => ({ ...prev, [field.name]: event.target.value }))
                        }
                        className="border border-rule px-2 py-1.5 bg-paper"
                      />
                    )}
                  </label>
                ))}
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={predict}
                  disabled={busy}
                  className="border border-primary px-3 py-1.5 hover:bg-paper-raised disabled:opacity-50"
                >
                  Predict
                </button>
                <label className="border border-rule px-3 py-1.5 hover:bg-paper-sunken cursor-pointer">
                  Batch CSV…
                  <input
                    type="file"
                    accept=".csv,.tsv,.txt"
                    className="hidden"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void batchUpload(file);
                    }}
                  />
                </label>
              </div>

              {prediction && (
                <div className="border-l-2 border-state-good pl-3 text-small">
                  <p>
                    <strong>Prediction:</strong> {String(prediction.prediction)} (model v
                    {prediction.model_version})
                  </p>
                  {prediction.probabilities && (
                    <ul className="mt-1">
                      {prediction.probabilities.map((entry) => (
                        <li key={entry.label}>
                          {entry.label}: {(entry.probability * 100).toFixed(1)}%
                        </li>
                      ))}
                    </ul>
                  )}
                  {prediction.notes?.length ? (
                    <ul className="mt-1 text-ink-muted">
                      {prediction.notes.map((note, index) => (
                        <li key={index}>{note}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              )}
            </div>

            <div className="space-y-3">
              <h3 className="text-small font-semibold">Client code</h3>
              {snippets ? (
                <div className="space-y-2">
                  {(["curl", "python", "javascript"] as const).map((language) => (
                    <details key={language} className="border border-rule">
                      <summary className="px-2 py-1 text-small cursor-pointer capitalize">
                        {language}
                      </summary>
                      <pre className="text-small overflow-x-auto p-2 bg-paper-sunken">
                        {snippets.snippets[language]}
                      </pre>
                    </details>
                  ))}
                </div>
              ) : (
                <p className="text-small text-ink-muted">Loading snippets…</p>
              )}
              <a
                href={services.deploymentOpenApiUrl(selected.slug)}
                className="text-small underline"
                target="_blank"
                rel="noreferrer"
              >
                Open the OpenAPI contract
              </a>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-3">
              <h3 className="text-small font-semibold">Drift check</h3>
              <button
                type="button"
                onClick={runDrift}
                disabled={busy}
                className="border border-rule px-3 py-1 text-small hover:bg-paper-sunken disabled:opacity-50"
              >
                Compare current row to training
              </button>
            </div>
            {drift ? (
              <div className={`border-l-2 pl-3 text-small ${DRIFT_TONE[drift.overall] ?? "border-rule"}`}>
                <p>
                  <strong>Overall:</strong> {drift.overall} (max PSI {drift.max_psi.toFixed(3)})
                </p>
                <ul className="mt-1 space-y-1">
                  {drift.features.map((feature) => (
                    <li key={feature.feature}>
                      <span className="font-medium">{feature.feature}</span>
                      {feature.psi !== null && ` — PSI ${feature.psi.toFixed(3)}`} ({feature.status}){" "}
                      <span className="text-ink-muted">{feature.message}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="text-small text-ink-muted">
                PSI compares the distribution of live inputs against the training data.
                Below 0.1 is stable; above 0.25 is a significant shift.
              </p>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
