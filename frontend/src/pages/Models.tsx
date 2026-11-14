/**
 * Model Registry: the "is this model ready for use?" page.
 *
 * A trained run is not a shippable model. This page is where a run becomes a
 * versioned, registerable artefact with a lifecycle (candidate → validated →
 * staging → production → archived), an inference signature, and a model card
 * that states its limits. Promoting to production is a deliberate, confirmed
 * action because it changes what live traffic gets.
 */
import { useCallback, useEffect, useState } from "react";

import services, { toApiError } from "../services/services";
import type {
  Deployment,
  ModelDetail,
  ModelState,
  ModelSummary,
  ModelVersion,
} from "../interface/LifecycleTypes";
import { usePageSeo } from "../hooks/usePageSeo";

const STATE_TONE: Record<ModelState, string> = {
  candidate: "text-ink-muted",
  validated: "text-ink",
  staging: "text-ink",
  production: "text-state-good font-semibold",
  archived: "text-ink-muted line-through",
};

/** The next forward state a version may move to, or null at the end. */
function nextState(status: ModelState): ModelState | null {
  const order: ModelState[] = ["candidate", "validated", "staging", "production"];
  const index = order.indexOf(status);
  if (index === -1) return null; // archived
  return index < order.length - 1 ? order[index + 1] : null;
}

export default function Models() {
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [selected, setSelected] = useState<ModelDetail | null>(null);
  const [version, setVersion] = useState<ModelVersion | null>(null);
  const [deployment, setDeployment] = useState<Deployment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [runs, setRuns] = useState<string[]>([]);

  usePageSeo({
    title: "Model Registry",
    description:
      "Register trained models as versioned artefacts, promote them through candidate, staging and production, and read their model cards.",
    path: "/app/models",
  });

  const loadModels = useCallback(async () => {
    try {
      setModels(await services.listModels());
      setError(null);
    } catch (caught) {
      setError(toApiError(caught).detail);
      setModels([]);
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const openModel = useCallback(async (id: string) => {
    try {
      const detail = await services.getModel(id);
      setSelected(detail);
      setDeployment(null);
      const latest = detail.versions[0] ?? null;
      setVersion(latest);
    } catch (caught) {
      setError(toApiError(caught).detail);
    }
  }, []);

  const selectVersion = async (modelVersion: ModelVersion) => {
    setVersion(modelVersion);
    try {
      const deployments = await services.listDeployments();
      setDeployment(
        deployments.find((item) => item.model_version_id === modelVersion.id) ?? null
      );
    } catch {
      setDeployment(null);
    }
  };

  const promote = async (target: ModelState) => {
    if (!selected || !version) return;
    if (target === "production") {
      const confirmed = window.confirm(
        `Promote ${selected.name} v${version.version} to production? ` +
          "This becomes the live model for its deployments."
      );
      if (!confirmed) return;
    }
    setBusy(true);
    try {
      const updated = await services.promoteModelVersion(
        selected.id,
        version.version,
        target,
        target === "production"
      );
      setVersion(updated);
      await openModel(selected.id);
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setBusy(false);
    }
  };

  const archive = async () => {
    if (!selected || !version) return;
    setBusy(true);
    try {
      const updated = await services.promoteModelVersion(
        selected.id,
        version.version,
        "archived"
      );
      setVersion(updated);
      await openModel(selected.id);
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setBusy(false);
    }
  };

  const deploy = async () => {
    if (!selected || !version) return;
    setBusy(true);
    try {
      const created = await services.deployModelVersion(selected.id, version.version);
      setDeployment(created);
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setBusy(false);
    }
  };

  const registerRun = async () => {
    const runId = runs[0]?.trim();
    if (!runId) {
      setError("Paste a run id to register — you can copy it from the Experiments page.");
      return;
    }
    setRegistering(true);
    try {
      const created = await services.registerModel(runId);
      await loadModels();
      await openModel(created.model_id);
    } catch (caught) {
      setError(toApiError(caught).detail);
    } finally {
      setRegistering(false);
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      <header>
        <h1 className="text-heading">Model Registry</h1>
        <p className="text-ink-muted mt-1 max-w-3xl">
          A versioned record of every model you consider shippable, with the run it came
          from, the dataset behind that run, its metrics, and an inference signature that
          says exactly what input it accepts.
        </p>
      </header>

      {error && (
        <div className="border-l-2 border-state-bad pl-3 text-small" role="alert">
          {error}
        </div>
      )}

      <section className="border border-rule p-4 space-y-2">
        <h2 className="text-subheading">Register a run</h2>
        <p className="text-small text-ink-muted">
          Register a successful run by its id (shown on the Experiments page).
        </p>
        <div className="flex flex-wrap gap-2">
          <input
            value={runs[0] ?? ""}
            onChange={(event) => setRuns([event.target.value])}
            placeholder="run id"
            className="border border-rule px-2 py-1.5 bg-paper min-w-[24rem]"
          />
          <button
            type="button"
            onClick={registerRun}
            disabled={registering}
            className="border border-primary px-3 py-1.5 hover:bg-paper-raised disabled:opacity-50"
          >
            {registering ? "Registering…" : "Register"}
          </button>
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="text-subheading">Models</h2>
        {models === null ? (
          <p className="text-small text-ink-muted" role="status">Loading…</p>
        ) : models.length === 0 ? (
          <p className="text-small text-ink-muted">Nothing registered yet.</p>
        ) : (
          <ul className="divide-y divide-rule border border-rule">
            {models.map((model) => (
              <li key={model.id}>
                <button
                  type="button"
                  onClick={() => void openModel(model.id)}
                  className={`w-full text-left px-3 py-2 hover:bg-paper-raised ${
                    selected?.id === model.id ? "bg-paper-raised" : ""
                  }`}
                >
                  <span className="flex flex-wrap items-baseline gap-x-3">
                    <span className="font-semibold">{model.name}</span>
                    <span className="text-small text-ink-muted">{model.task_type}</span>
                    <span className="text-small text-ink-muted">
                      {model.n_versions} version{model.n_versions === 1 ? "" : "s"}
                    </span>
                    {model.production_version !== null && (
                      <span className="text-small text-state-good">
                        v{model.production_version} in production
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
        <section className="border-t border-rule pt-4 space-y-4">
          <h2 className="text-subheading">{selected.name} versions</h2>
          <ul className="divide-y divide-rule border border-rule">
            {selected.versions.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => void selectVersion(item)}
                  className={`w-full text-left px-3 py-2 flex flex-wrap items-baseline gap-x-3 hover:bg-paper-raised ${
                    version?.id === item.id ? "bg-paper-raised" : ""
                  }`}
                >
                  <span className="font-semibold">v{item.version}</span>
                  <span className={`text-small ${STATE_TONE[item.status]}`}>{item.status_label}</span>
                  <span className="text-small text-ink-muted">
                    {item.model_card?.algorithm ?? "—"}
                  </span>
                </button>
              </li>
            ))}
          </ul>

          {version && (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {(() => {
                  const forward = nextState(version.status);
                  return forward ? (
                    <button
                      type="button"
                      onClick={() => void promote(forward)}
                      disabled={busy}
                      className="border border-primary px-3 py-1.5 hover:bg-paper-raised disabled:opacity-50"
                    >
                      Promote to {forward}
                    </button>
                  ) : null;
                })()}
                {version.status !== "archived" && (
                  <button
                    type="button"
                    onClick={archive}
                    disabled={busy}
                    className="border border-rule px-3 py-1.5 hover:bg-paper-sunken disabled:opacity-50"
                  >
                    Archive
                  </button>
                )}
                <button
                  type="button"
                  onClick={deploy}
                  disabled={busy || deployment !== null}
                  className="border border-rule px-3 py-1.5 hover:bg-paper-sunken disabled:opacity-50"
                >
                  {deployment ? "Already deployed" : "Deploy as API"}
                </button>
              </div>

              {deployment && (
                <p className="text-small border-l-2 border-state-good pl-3">
                  Deployed at <code>{deployment.predict_url}</code> — see the Deployments
                  page for snippets and monitoring.
                </p>
              )}

              <div>
                <h3 className="text-small font-semibold mb-1">Model card</h3>
                <dl className="text-small grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1">
                  <div>
                    <dt className="text-ink-muted inline">Task: </dt>
                    <dd className="inline">{version.model_card?.task}</dd>
                  </div>
                  <div>
                    <dt className="text-ink-muted inline">Validation: </dt>
                    <dd className="inline">{version.model_card?.validation_strategy}</dd>
                  </div>
                  <div>
                    <dt className="text-ink-muted inline">Algorithm: </dt>
                    <dd className="inline">{version.model_card?.algorithm}</dd>
                  </div>
                  <div>
                    <dt className="text-ink-muted inline">Dataset: </dt>
                    <dd className="inline">
                      {version.model_card?.dataset?.dataset_version} (
                      {version.model_card?.dataset?.rows?.toLocaleString()} rows)
                    </dd>
                  </div>
                </dl>
                {version.model_card?.intended_use && (
                  <p className="text-small mt-2">
                    <strong>Intended use:</strong> {version.model_card.intended_use}
                  </p>
                )}
                {version.model_card?.out_of_scope_use?.length > 0 && (
                  <div className="text-small mt-1">
                    <strong>Out of scope:</strong>
                    <ul className="list-disc pl-5">
                      {version.model_card.out_of_scope_use.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {version.model_card?.known_limitations?.length > 0 && (
                  <div className="text-small mt-1">
                    <strong>Known limitations:</strong>
                    <ul className="list-disc pl-5">
                      {version.model_card.known_limitations.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {version.model_card?.data_quality_note && (
                  <p className="text-small border-l-2 border-state-warn pl-3 mt-2">
                    {version.model_card.data_quality_note}
                  </p>
                )}
              </div>

              <div>
                <h3 className="text-small font-semibold mb-1">Inference signature</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-small border-collapse">
                    <thead>
                      <tr className="text-left border-b border-rule-strong">
                        <th className="py-1 pr-3 font-semibold">Feature</th>
                        <th className="py-1 pr-3 font-semibold">Type</th>
                        <th className="py-1 pr-3 font-semibold">Required</th>
                        <th className="py-1 font-semibold">Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {version.signature?.features?.map((field) => (
                        <tr key={field.name} className="border-b border-rule">
                          <td className="py-1 pr-3">{field.name}</td>
                          <td className="py-1 pr-3">{field.type}</td>
                          <td className="py-1 pr-3">{field.required ? "yes" : "no"}</td>
                          <td className="py-1 text-ink-muted">
                            {field.categories
                              ? `${field.categories.length} categories`
                              : field.minimum !== null && field.maximum !== null
                                ? `range ${field.minimum}–${field.maximum}`
                                : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
