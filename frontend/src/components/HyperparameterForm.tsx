/**
 * A form built at runtime from the backend's `HyperparamSpec[]`.
 *
 * This is what lets the app adjust to any algorithm in the registry without a
 * hard-coded field list: whatever `/api/algorithms/` describes is what the
 * user gets to tune.
 *
 * The important rule is that an untouched field is NOT sent. `value` holds
 * only the settings the user actually changed, so scikit-learn's own defaults
 * stay in charge of everything else — which is why the number inputs show the
 * default as a placeholder rather than pre-filling it.
 */
import { useState } from "react";
import type { HyperparamSpec } from "../interface/MlTypes";

interface HyperparameterFormProps {
  /** Only used to namespace input ids; the parent remounts on a change. */
  algorithmKey: string;
  specs: HyperparamSpec[];
  /** Touched values only — this is exactly what goes into the request. */
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  disabled?: boolean;
}

function isTouched(value: Record<string, unknown>, name: string): boolean {
  // hasOwnProperty, not truthiness: a typed `0` or an unticked `false` counts.
  return Object.prototype.hasOwnProperty.call(value, name);
}

function withField(
  value: Record<string, unknown>,
  name: string,
  next: unknown
): Record<string, unknown> {
  return { ...value, [name]: next };
}

function withoutField(
  value: Record<string, unknown>,
  name: string
): Record<string, unknown> {
  const copy = { ...value };
  delete copy[name];
  return copy;
}

/** How the algorithm's own default reads in the UI. */
function defaultText(spec: HyperparamSpec): string {
  if (spec.default === null) return "auto";
  if (typeof spec.default === "boolean") return spec.default ? "on" : "off";
  if (spec.type === "select") {
    const match = spec.options?.find(
      (option) => option.value === String(spec.default)
    );
    return match ? match.label : String(spec.default);
  }
  return String(spec.default);
}

function rangeText(spec: HyperparamSpec): string | null {
  if (spec.min !== null && spec.max !== null) return `${spec.min} to ${spec.max}`;
  if (spec.min !== null) return `at least ${spec.min}`;
  if (spec.max !== null) return `at most ${spec.max}`;
  return null;
}

const HyperparameterForm = ({
  algorithmKey,
  specs,
  value,
  onChange,
  disabled = false,
}: HyperparameterFormProps) => {
  // Number inputs are uncontrolled so a half-typed "0." is not stomped on by a
  // re-render. Bumping a field's token remounts THAT field, which is how
  // "reset" and "clamp on blur" push a new value back into the DOM.
  //
  // The token is per field, not shared: "clamp on blur" fires while focus is
  // already travelling to the next input, so a single shared token would
  // remount the field focus is heading for and drop it on <body>. Remounting
  // only the field that just blurred can never steal focus, because that field
  // no longer has it. Tokens left behind by specs that disappear are harmless —
  // the parent keys this form by algorithm, so a real algorithm change
  // remounts the whole form anyway.
  const [resetTokens, setResetTokens] = useState<Record<string, number>>({});

  const touchedCount = specs.filter((spec) => isTouched(value, spec.name)).length;

  const bumpToken = (name: string) => {
    setResetTokens((tokens) => ({ ...tokens, [name]: (tokens[name] ?? 0) + 1 }));
  };

  const clearOne = (name: string) => {
    onChange(withoutField(value, name));
    bumpToken(name);
  };

  const resetAll = () => {
    onChange({});
    // Focus is on the "Reset to defaults" button here, so remounting every
    // number input is safe as well as necessary.
    setResetTokens((tokens) => {
      const next: Record<string, number> = { ...tokens };
      for (const spec of specs) next[spec.name] = (tokens[spec.name] ?? 0) + 1;
      return next;
    });
  };

  const renderField = (spec: HyperparamSpec) => {
    const touched = isTouched(value, spec.name);
    const fieldId = `hp-${algorithmKey}-${spec.name}`;
    const range = rangeText(spec);

    return (
      <div key={spec.name}>
        <div className="mb-1 flex flex-wrap items-baseline gap-x-2">
          <label htmlFor={fieldId} className="text-sm font-medium text-ink">
            {spec.label}
          </label>
          <code className="text-xs text-ink-muted">{spec.name}</code>
          {touched && (
            <>
              <span className="text-xs font-semibold text-state-warn">
                customised
              </span>
              <button
                type="button"
                disabled={disabled}
                onClick={() => clearOne(spec.name)}
                className="text-xs font-medium text-secondary underline hover:no-underline disabled:cursor-not-allowed disabled:opacity-50"
              >
                use default
              </button>
            </>
          )}
        </div>

        {spec.type === "bool" ? (
          <label
            htmlFor={fieldId}
            className="flex cursor-pointer items-center gap-3 bg-paper-sunken p-3"
          >
            <input
              id={fieldId}
              type="checkbox"
              disabled={disabled}
              checked={touched ? value[spec.name] === true : spec.default === true}
              onChange={(event) =>
                onChange(withField(value, spec.name, event.target.checked))
              }
              className="h-5 w-5 rounded-control border-rule text-accent focus:ring-accent"
            />
            <span className="text-sm text-ink">
              {touched
                ? value[spec.name] === true
                  ? "On"
                  : "Off"
                : `Default — ${defaultText(spec)}`}
            </span>
          </label>
        ) : spec.type === "select" ? (
          <select
            id={fieldId}
            disabled={disabled}
            value={touched ? String(value[spec.name] ?? "") : ""}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") onChange(withoutField(value, spec.name));
              else onChange(withField(value, spec.name, raw));
            }}
            className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-60"
          >
            <option value="">Default — {defaultText(spec)}</option>
            {(spec.options ?? []).map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            key={`${resetTokens[spec.name] ?? 0}:${spec.name}`}
            id={fieldId}
            type="number"
            disabled={disabled}
            defaultValue={
              typeof value[spec.name] === "number" ? String(value[spec.name]) : ""
            }
            placeholder={`Default — ${defaultText(spec)}`}
            min={spec.min ?? undefined}
            max={spec.max ?? undefined}
            step={spec.step ?? (spec.type === "int" ? 1 : "any")}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw.trim() === "") {
                onChange(withoutField(value, spec.name));
                return;
              }
              const parsed = Number(raw);
              if (!Number.isFinite(parsed)) return;
              onChange(withField(value, spec.name, parsed));
            }}
            onBlur={(event) => {
              const raw = event.target.value;
              if (raw.trim() === "") return;
              const entered = Number(raw);
              if (!Number.isFinite(entered)) return;

              let corrected = spec.type === "int" ? Math.round(entered) : entered;
              if (spec.min !== null && corrected < spec.min) corrected = spec.min;
              if (spec.max !== null && corrected > spec.max) corrected = spec.max;

              if (corrected !== entered) {
                onChange(withField(value, spec.name, corrected));
                bumpToken(spec.name);
              }
            }}
            className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-60"
          />
        )}

        <p className="mt-1 text-sm text-ink-muted">
          {spec.description}
          {range && spec.type !== "bool" && spec.type !== "select" && (
            <> Allowed range: {range}.</>
          )}
        </p>
      </div>
    );
  };

  if (specs.length === 0) {
    return (
      <p className="text-small text-ink-muted max-w-measure">
        This algorithm exposes no tunable settings — it runs with scikit-learn's
        defaults.
      </p>
    );
  }

  const basic = specs.filter((spec) => !spec.advanced);
  const advanced = specs.filter((spec) => spec.advanced);
  const advancedTouched = advanced.filter((spec) =>
    isTouched(value, spec.name)
  ).length;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <p className="text-sm text-ink-muted">
          {touchedCount === 0 ? (
            <>
              Nothing changed — every setting uses the algorithm's own default.
            </>
          ) : (
            <>
              <span className="font-semibold text-ink">{touchedCount}</span>{" "}
              {touchedCount === 1 ? "setting" : "settings"} customised. Untouched
              fields are not sent, so scikit-learn's defaults still apply.
            </>
          )}
        </p>
        <button
          type="button"
          disabled={disabled || touchedCount === 0}
          onClick={resetAll}
          className="ml-auto rounded-control bg-rule px-3 py-2 text-sm font-semibold text-ink transition-colors hover:bg-rule-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          Reset to defaults
        </button>
      </div>

      {basic.length > 0 && <div className="space-y-4">{basic.map(renderField)}</div>}

      {advanced.length > 0 && (
        <details className="mt-4 bg-paper-sunken p-4">
          <summary className="cursor-pointer text-sm font-semibold text-ink">
            Advanced ({advanced.length})
            {advancedTouched > 0 && (
              <span className="ml-2 font-medium text-state-warn">
                {advancedTouched} customised
              </span>
            )}
          </summary>
          <div className="mt-4 space-y-4">{advanced.map(renderField)}</div>
        </details>
      )}
    </div>
  );
};

export default HyperparameterForm;
