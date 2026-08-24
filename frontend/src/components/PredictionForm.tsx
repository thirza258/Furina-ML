/**
 * "Try it yourself" — a live prediction against the trained model.
 *
 * Every input here is generated from `run.input_schema`, which the backend
 * built from the columns the model was actually fitted on. Nothing about the
 * form is hardcoded, so it is correct for a two-column toy dataset and for a
 * forty-column one without a code change.
 */
import { useState } from "react";
import type { FormEvent } from "react";

import services from "../services/services";
import type {
  PredictField,
  PredictionResult,
  TaskType,
} from "../interface/MlTypes";

/**
 * `services` already throws the normalized `{ detail }` shape, so this only has
 * to narrow it — running it back through `toApiError` would stringify the
 * object and render "[object Object]".
 */
function apiErrorDetail(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error
    ? error.message
    : "The prediction failed for an unknown reason.";
}

function emptyValues(schema: PredictField[]): Record<string, string> {
  const next: Record<string, string> = {};
  for (const field of schema) next[field.name] = "";
  return next;
}

/**
 * The backend rejects a row in which nothing was supplied
 * (`predict.py::_prediction_frame` raises "Row N has no values" when its filled
 * count is 0), so an all-blank row must never leave the browser. Trimming here
 * matches `handleSubmit`, which trims before deciding what to send.
 */
function hasAnyValue(
  schema: PredictField[],
  values: Record<string, string>
): boolean {
  return schema.some((field) => (values[field.name] ?? "").trim() !== "");
}

function exampleValues(schema: PredictField[]): Record<string, string> {
  const next: Record<string, string> = {};
  for (const field of schema) {
    next[field.name] = field.default === null ? "" : String(field.default);
  }
  return next;
}

function rangeHint(field: PredictField): string | null {
  if (field.min === null && field.max === null) return null;
  if (field.min !== null && field.max !== null) {
    return `Training data ranged ${field.min} to ${field.max}.`;
  }
  if (field.min !== null) return `Training data started at ${field.min}.`;
  return `Training data topped out at ${field.max}.`;
}

function formatPrediction(value: string | number): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "n/a";
    const magnitude = Math.abs(value);
    if (magnitude >= 10000) {
      return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
    if (magnitude >= 1) return value.toFixed(3);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return value;
}

interface PredictionFormProps {
  runId: string;
  schema: PredictField[];
  taskType: TaskType;
  targetColumn: string;
}

export default function PredictionForm({
  runId,
  schema,
  taskType,
  targetColumn,
}: PredictionFormProps) {
  const [values, setValues] = useState<Record<string, string>>(() =>
    emptyValues(schema)
  );
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const filledIn = hasAnyValue(schema, values);

  const setField = (name: string, value: string) => {
    setValues((previous) => ({ ...previous, [name]: value }));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!filledIn) {
      setError(
        "Fill in at least one field — the model needs something to go on. " +
          "\u201cFill with an example row\u201d gives you a valid starting point."
      );
      setResult(null);
      return;
    }
    setSubmitting(true);
    setError(null);

    // Blank stays blank on the wire: the backend turns it into NaN and the
    // pipeline's imputer fills it with a typical value.
    const row: Record<string, unknown> = {};
    for (const field of schema) {
      const raw = (values[field.name] ?? "").trim();
      if (raw === "") {
        row[field.name] = null;
        continue;
      }
      if (field.kind === "numeric" && Number.isFinite(Number(raw))) {
        row[field.name] = Number(raw);
      } else {
        row[field.name] = raw;
      }
    }

    try {
      const response = await services.predict(runId, [row]);
      const first = response.predictions.length > 0 ? response.predictions[0] : null;
      if (!first) {
        setError("The API returned no prediction for that row.");
        setResult(null);
      } else {
        setResult(first);
      }
    } catch (caught) {
      setError(apiErrorDetail(caught));
      setResult(null);
    } finally {
      setSubmitting(false);
    }
  };

  if (schema.length === 0) {
    return (
      <section>
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Try it yourself
        </h2>
        <p className="rounded-md border border-gray-300 bg-gray-100 p-4 text-sm text-ink-muted">
          This run did not record an input schema, so there is no form to build.
          Retrain the model to make live predictions available.
        </p>
      </section>
    );
  }

  const sortedProbabilities = result?.probabilities
    ? [...result.probabilities].sort((a, b) => b.probability - a.probability)
    : null;

  return (
    <section>
      <h2 className="text-xl font-semibold text-gray-800 mb-2">
        Try it yourself
      </h2>
      <p className="mb-4 text-sm text-ink-muted">
        Fill in a row and the trained model will predict{" "}
        <code>{targetColumn}</code>. At least one field is needed; every field
        you leave blank is filled with a typical value from the training data,
        exactly as it was for missing values during training.
      </p>

      <div className="mb-4 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={() => setValues(exampleValues(schema))}
          className="rounded-md bg-gray-300 px-4 py-2 text-sm font-semibold text-gray-800 transition-colors hover:bg-gray-400"
        >
          Fill with an example row
        </button>
        <button
          type="button"
          onClick={() => {
            setValues(emptyValues(schema));
            setResult(null);
            setError(null);
          }}
          className="rounded-md bg-gray-300 px-4 py-2 text-sm font-semibold text-gray-800 transition-colors hover:bg-gray-400"
        >
          Clear the form
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {schema.map((field) => {
            const inputId = `predict-${field.name}`;
            const hint = rangeHint(field);
            const placeholder =
              field.default === null
                ? "leave blank to impute"
                : `e.g. ${field.default}`;

            return (
              <div key={field.name}>
                <label
                  htmlFor={inputId}
                  className="mb-1 block text-sm font-medium text-gray-700"
                  title={field.name}
                >
                  <span className="align-middle">{field.name}</span>
                  <span className="ml-2 align-middle text-xs font-normal text-ink-muted">
                    {field.kind}
                    {field.required ? "" : " · optional"}
                  </span>
                </label>

                {field.options && field.options.length > 0 ? (
                  <select
                    id={inputId}
                    value={values[field.name] ?? ""}
                    onChange={(event) =>
                      setField(field.name, event.target.value)
                    }
                    className="w-full rounded-md border border-gray-300 bg-gray-200 p-3 text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary"
                  >
                    <option value="">(leave blank)</option>
                    {field.options.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : field.kind === "numeric" ? (
                  <input
                    id={inputId}
                    type="number"
                    step="any"
                    value={values[field.name] ?? ""}
                    placeholder={placeholder}
                    onChange={(event) =>
                      setField(field.name, event.target.value)
                    }
                    className="w-full rounded-md border border-gray-300 bg-gray-200 p-3 text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                ) : (
                  <input
                    id={inputId}
                    type="text"
                    value={values[field.name] ?? ""}
                    placeholder={placeholder}
                    onChange={(event) =>
                      setField(field.name, event.target.value)
                    }
                    className="w-full rounded-md border border-gray-300 bg-gray-200 p-3 text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                )}

                <p className="mt-1 text-xs text-ink-muted">
                  {hint ? `${hint} ` : ""}
                  Leave it blank and the model fills it in.
                </p>
              </div>
            );
          })}
        </div>

        <button
          type="submit"
          disabled={submitting || !filledIn}
          className="mt-6 w-full rounded-md bg-primary px-4 py-3 text-lg font-semibold text-white transition-colors hover:bg-opacity-90 disabled:cursor-not-allowed disabled:bg-primary/50"
        >
          {submitting ? "Predicting…" : "Predict"}
        </button>

        {!filledIn && (
          <p className="mt-2 text-center text-xs text-ink-muted">
            Fill in at least one field to predict, or use “Fill with an example
            row” above.
          </p>
        )}
      </form>

      {error && (
        <div className="mt-4 flex items-start justify-between gap-4 rounded-md border border-state-bad/40 bg-state-bad/10 p-4">
          <p className="text-sm text-state-bad">
            <span className="font-semibold">✕ Prediction failed. </span>
            {error}
          </p>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-sm font-semibold text-state-bad hover:underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {result && (
        <div className="mt-6 rounded-lg border border-gray-200 bg-gray-50 p-6">
          <p className="text-sm font-semibold uppercase tracking-wide text-ink-muted">
            Predicted {targetColumn}
          </p>
          <p className="mt-1 break-words font-mono text-4xl font-bold text-ink">
            {formatPrediction(result.prediction)}
          </p>
          <p className="mt-2 text-xs text-ink-muted">
            {taskType === "classification"
              ? "The class the model believes this row belongs to."
              : "The value the model expects for this row, in the target's own units."}
          </p>

          {sortedProbabilities && sortedProbabilities.length > 0 ? (
            <div className="mt-5">
              <h3 className="mb-2 text-sm font-semibold text-gray-800">
                How confident the model is
              </h3>
              <ul className="space-y-2">
                {sortedProbabilities.map((entry) => {
                  const pct = Math.max(
                    0,
                    Math.min(100, entry.probability * 100)
                  );
                  return (
                    <li key={entry.label} className="flex items-center gap-3">
                      <span
                        className="w-1/4 max-w-[12rem] shrink-0 truncate text-sm text-ink"
                        title={entry.label}
                      >
                        {entry.label}
                      </span>
                      <span className="h-3 flex-1 rounded-[4px] bg-gray-200">
                        <span
                          className="block h-3 rounded-[4px] bg-series-1 transition-opacity hover:opacity-80"
                          style={{ width: `${pct}%` }}
                          title={`${entry.label}: ${pct.toFixed(1)}% confidence`}
                        />
                      </span>
                      <span className="w-20 shrink-0 text-right font-mono text-xs text-ink">
                        {pct.toFixed(1)}%
                      </span>
                    </li>
                  );
                })}
              </ul>
              <p className="mt-2 text-xs text-ink-muted">
                These are the model's own confidence scores across every class it
                learned; they add up to 100%.
              </p>
            </div>
          ) : (
            taskType === "classification" && (
              <p className="mt-4 text-xs text-ink-muted">
                This algorithm does not produce class probabilities, so only the
                chosen class is available.
              </p>
            )
          )}
        </div>
      )}
    </section>
  );
}
