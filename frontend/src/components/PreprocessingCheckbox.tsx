import { useMemo, useState } from "react";

import type { PreprocessOp, PreprocessStep } from "../interface/MlTypes";
import type {
  OpParamSpec,
  PreprocessingCheckboxProps,
} from "../interface/PreprocessingTypes";
import {
  PREPROCESS_GROUPS,
  PREPROCESS_OP_META,
  applicableColumns,
} from "../interface/PreprocessingTypes";

/** Initial `params` for a freshly enabled step, already JSON-typed. */
function defaultParams(
  specs: OpParamSpec[]
): Record<string, unknown> | null {
  if (specs.length === 0) return null;
  const params: Record<string, unknown> = {};
  specs.forEach((spec) => {
    if (spec.kind === "int") params[spec.name] = Number.parseInt(spec.defaultValue, 10);
    else if (spec.kind === "float") params[spec.name] = Number.parseFloat(spec.defaultValue);
    else params[spec.name] = spec.defaultValue;
  });
  return params;
}

function rawKey(op: PreprocessOp, name: string): string {
  return `${op}.${name}`;
}

export default function PreprocessingCheckbox({
  columns,
  steps,
  onStepsChange,
  disabled = false,
}: PreprocessingCheckboxProps) {
  // Numeric inputs need a raw string while the user is mid-edit; the coerced
  // number is mirrored into the step so `steps` stays a valid request body.
  const [rawNumbers, setRawNumbers] = useState<Record<string, string>>({});

  const byOp = useMemo(() => {
    const map = new Map<PreprocessOp, PreprocessStep>();
    steps.forEach((step) => map.set(step.op, step));
    return map;
  }, [steps]);

  const columnNames = useMemo(() => columns.map((c) => c.name), [columns]);

  const replaceStep = (op: PreprocessOp, next: PreprocessStep) => {
    onStepsChange(steps.map((step) => (step.op === op ? next : step)));
  };

  const toggleOp = (op: PreprocessOp, enabled: boolean) => {
    const meta = PREPROCESS_OP_META[op];
    if (enabled) {
      const suggested = meta.params.find((p) => p.kind === "column");
      const params = defaultParams(meta.params);
      if (params && suggested) {
        const fallback = columns.find((c) => c.suitable_as_target);
        params[suggested.name] = fallback ? fallback.name : "";
      }
      onStepsChange([
        ...steps,
        {
          op,
          columns: meta.requiresColumns ? [] : null,
          params,
        },
      ]);
      return;
    }

    onStepsChange(steps.filter((step) => step.op !== op));
    setRawNumbers((current) => {
      const next = { ...current };
      meta.params.forEach((param) => delete next[rawKey(op, param.name)]);
      return next;
    });
  };

  const moveStep = (op: PreprocessOp, delta: number) => {
    const index = steps.findIndex((step) => step.op === op);
    const target = index + delta;
    if (index < 0 || target < 0 || target >= steps.length) return;
    const next = [...steps];
    const [moved] = next.splice(index, 1);
    next.splice(target, 0, moved);
    onStepsChange(next);
  };

  const setStepColumns = (op: PreprocessOp, name: string, checked: boolean) => {
    const step = byOp.get(op);
    if (!step) return;
    const meta = PREPROCESS_OP_META[op];
    const current = step.columns ?? [];
    const next = checked
      ? [...current, name]
      : current.filter((entry) => entry !== name);

    replaceStep(op, {
      ...step,
      // Falling back to null restores "all applicable columns".
      columns: next.length === 0 && !meta.requiresColumns ? null : next,
    });
  };

  const setAllColumns = (op: PreprocessOp, value: string[] | null) => {
    const step = byOp.get(op);
    if (!step) return;
    replaceStep(op, { ...step, columns: value });
  };

  const setParam = (op: PreprocessOp, spec: OpParamSpec, raw: string) => {
    const step = byOp.get(op);
    if (!step) return;
    const params = { ...(step.params ?? {}) };

    if (spec.kind === "int" || spec.kind === "float") {
      setRawNumbers((current) => ({ ...current, [rawKey(op, spec.name)]: raw }));
      const parsed =
        spec.kind === "int" ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
      params[spec.name] = Number.isFinite(parsed)
        ? parsed
        : spec.kind === "int"
          ? Number.parseInt(spec.defaultValue, 10)
          : Number.parseFloat(spec.defaultValue);
    } else {
      params[spec.name] = raw;
    }

    replaceStep(op, { ...step, params });
  };

  const paramValue = (
    op: PreprocessOp,
    step: PreprocessStep,
    spec: OpParamSpec
  ): string => {
    if (spec.kind === "int" || spec.kind === "float") {
      const raw = rawNumbers[rawKey(op, spec.name)];
      if (raw !== undefined) return raw;
    }
    const value = step.params ? step.params[spec.name] : undefined;
    if (typeof value === "string") return value;
    if (typeof value === "number") return String(value);
    return spec.defaultValue;
  };

  return (
    <div className="w-full">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-gray-800">
          Build a cleaning recipe
        </h2>
        <p className="text-sm text-ink-muted">
          Tick the operations you want. They run top to bottom in the order
          listed below, on a copy of your data — nothing changes until you apply
          it.
        </p>
      </div>

      {steps.length === 0 ? (
        <div className="mb-6 rounded-md border border-gray-300 bg-gray-100 p-4">
          <p className="text-sm text-ink-muted">
            No steps selected yet. Your recipe and its running order will appear
            here.
          </p>
        </div>
      ) : (
        <div className="mb-6 rounded-md border border-gray-300 bg-gray-100 p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-800">
              Recipe order ({steps.length}{" "}
              {steps.length === 1 ? "step" : "steps"})
            </h3>
            <button
              type="button"
              onClick={() => {
                onStepsChange([]);
                setRawNumbers({});
              }}
              disabled={disabled}
              className="text-sm font-semibold text-secondary hover:underline disabled:text-gray-400 disabled:no-underline disabled:cursor-not-allowed"
            >
              Clear all
            </button>
          </div>
          <ol className="space-y-2">
            {steps.map((step, index) => (
              <li
                key={step.op}
                className="flex items-center gap-3 rounded-md bg-white border border-gray-200 px-3 py-2"
              >
                <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-primary text-white text-xs font-bold">
                  {index + 1}
                </span>
                <span className="flex-grow text-sm text-ink">
                  {PREPROCESS_OP_META[step.op].label}
                  <span className="text-ink-muted">
                    {" "}
                    ·{" "}
                    {step.columns === null
                      ? "all applicable columns"
                      : `${step.columns.length} column${step.columns.length === 1 ? "" : "s"}`}
                  </span>
                </span>
                <button
                  type="button"
                  onClick={() => moveStep(step.op, -1)}
                  disabled={disabled || index === 0}
                  title="Run this step earlier"
                  className="px-2 py-1 rounded border border-gray-300 text-sm text-gray-800 hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  ↑
                </button>
                <button
                  type="button"
                  onClick={() => moveStep(step.op, 1)}
                  disabled={disabled || index === steps.length - 1}
                  title="Run this step later"
                  className="px-2 py-1 rounded border border-gray-300 text-sm text-gray-800 hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  ↓
                </button>
                <button
                  type="button"
                  onClick={() => toggleOp(step.op, false)}
                  disabled={disabled}
                  title="Remove this step"
                  className="px-2 py-1 rounded border border-gray-300 text-sm font-semibold text-state-bad hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Remove
                </button>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="space-y-8">
        {PREPROCESS_GROUPS.map((group) => (
          <div key={group.title}>
            <h3 className="text-lg font-semibold text-secondary">
              {group.title}
            </h3>
            <p className="text-sm text-ink-muted mb-3">{group.blurb}</p>

            <div className="space-y-3">
              {group.ops.map((op) => {
                const meta = PREPROCESS_OP_META[op];
                const step = byOp.get(op);
                const enabled = step !== undefined;
                const available = applicableColumns(op, columns);
                const selected = step?.columns ?? null;
                const needsColumns =
                  enabled && meta.requiresColumns && (selected?.length ?? 0) === 0;

                return (
                  <div
                    key={op}
                    className={`rounded-md border p-4 transition-colors ${
                      enabled
                        ? "border-primary bg-primary/5"
                        : "border-gray-200 bg-white"
                    }`}
                  >
                    <div className="flex items-start">
                      <input
                        type="checkbox"
                        id={`op-${op}`}
                        name={op}
                        checked={enabled}
                        disabled={disabled}
                        onChange={(event) => toggleOp(op, event.target.checked)}
                        className="mt-1 h-5 w-5 flex-shrink-0 rounded border-gray-300 text-accent focus:ring-accent disabled:cursor-not-allowed"
                      />
                      <div className="ml-3 flex-grow">
                        <label
                          htmlFor={`op-${op}`}
                          className="font-semibold text-gray-800 cursor-pointer"
                        >
                          {meta.label}
                        </label>
                        <p className="text-sm text-ink-muted">{meta.help}</p>
                        {available.length === 0 && (
                          <p className="mt-1 text-xs font-semibold text-state-warn">
                            <span aria-hidden="true">⚠</span> No column in this
                            dataset matches this operation — it will be skipped.
                          </p>
                        )}
                      </div>
                    </div>

                    {enabled && step && (
                      <div className="mt-4 ml-8 space-y-4">
                        <details className="rounded-md border border-gray-300 bg-white">
                          <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-gray-800">
                            Columns:{" "}
                            <span className="font-normal text-ink-muted">
                              {selected === null
                                ? `all applicable (${available.length})`
                                : `${selected.length} selected`}
                            </span>
                          </summary>
                          <div className="border-t border-gray-200 p-3">
                            <div className="mb-2 flex gap-3">
                              <button
                                type="button"
                                disabled={disabled}
                                onClick={() =>
                                  setAllColumns(
                                    op,
                                    available.map((column) => column.name)
                                  )
                                }
                                className="text-xs font-semibold text-secondary hover:underline disabled:text-gray-400"
                              >
                                Select every applicable column
                              </button>
                              <button
                                type="button"
                                disabled={disabled}
                                onClick={() =>
                                  setAllColumns(
                                    op,
                                    meta.requiresColumns ? [] : null
                                  )
                                }
                                className="text-xs font-semibold text-secondary hover:underline disabled:text-gray-400"
                              >
                                {meta.requiresColumns
                                  ? "Clear selection"
                                  : "Use all applicable"}
                              </button>
                            </div>

                            {available.length === 0 ? (
                              <p className="text-xs text-ink-muted">
                                Nothing to choose from.
                              </p>
                            ) : (
                              <div className="max-h-48 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 gap-1">
                                {available.map((column) => {
                                  const checked =
                                    selected !== null &&
                                    selected.includes(column.name);
                                  return (
                                    <label
                                      key={column.name}
                                      className="flex items-center gap-2 text-sm text-ink rounded px-2 py-1 hover:bg-gray-100 cursor-pointer"
                                      title={`${column.kind} · ${column.missing_pct}% missing`}
                                    >
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        disabled={disabled}
                                        onChange={(event) =>
                                          setStepColumns(
                                            op,
                                            column.name,
                                            event.target.checked
                                          )
                                        }
                                        className="h-4 w-4 rounded border-gray-300 text-accent focus:ring-accent"
                                      />
                                      <span className="truncate">
                                        {column.name}
                                      </span>
                                      <span className="ml-auto text-xs text-ink-muted">
                                        {column.kind}
                                      </span>
                                    </label>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        </details>

                        {needsColumns && (
                          <p className="text-sm font-semibold text-state-warn">
                            <span aria-hidden="true">⚠</span> Choose at least one
                            column — this operation has no safe default.
                          </p>
                        )}

                        {meta.params.length > 0 && (
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {meta.params.map((spec) => {
                              const inputId = `param-${op}-${spec.name}`;
                              const value = paramValue(op, step, spec);
                              return (
                                <div key={spec.name}>
                                  <label
                                    htmlFor={inputId}
                                    className="block text-sm font-semibold text-gray-800 mb-1"
                                  >
                                    {spec.label}
                                  </label>
                                  {spec.kind === "column" ? (
                                    <select
                                      id={inputId}
                                      value={value}
                                      disabled={disabled}
                                      onChange={(event) =>
                                        setParam(op, spec, event.target.value)
                                      }
                                      className="w-full p-2 bg-gray-200 border border-gray-300 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-60"
                                    >
                                      <option value="">Choose a column…</option>
                                      {columnNames.map((name) => (
                                        <option key={name} value={name}>
                                          {name}
                                        </option>
                                      ))}
                                    </select>
                                  ) : (
                                    <input
                                      id={inputId}
                                      type={spec.kind === "text" ? "text" : "number"}
                                      value={value}
                                      disabled={disabled}
                                      min={spec.min ?? undefined}
                                      max={spec.max ?? undefined}
                                      step={spec.kind === "int" ? 1 : "any"}
                                      onChange={(event) =>
                                        setParam(op, spec, event.target.value)
                                      }
                                      className="w-full p-2 bg-gray-200 border border-gray-300 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-60"
                                    />
                                  )}
                                  <p className="text-xs text-ink-muted mt-1">
                                    {spec.help}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
