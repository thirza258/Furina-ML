/**
 * Column pickers for the training form.
 *
 * Both are driven entirely by the real `ColumnProfile[]` the backend produced
 * when the CSV was uploaded, so what the user sees here is what the model will
 * actually be fitted on. Two exports:
 *
 *   <TargetColumnSelector />  — single select for the thing to predict.
 *   <FeatureColumnSelector /> — multi select for the columns to predict it from.
 */
import type { ColumnProfile } from "../interface/MlTypes";
import { WARNING_MARK } from "../explainer/marks";

/* ------------------------------------------------------------------ */
/* Shared helpers                                                      */
/* ------------------------------------------------------------------ */

/** Non-null row count for a column, derived from the dataset's row count. */
function nonNullCount(column: ColumnProfile, nRows: number): number {
  return Math.max(nRows - column.missing, 0);
}

/**
 * Why the backend refused this column as a target. The backend's rule is
 * "nunique < 2, or all-null, or (nunique == non-null count and kind == text)",
 * so these branches mirror it in the order it applies them.
 */
function unsuitableReason(column: ColumnProfile, nRows: number): string {
  const usable = nonNullCount(column, nRows);

  if (nRows > 0 && usable === 0) {
    return "every value is missing — there is nothing here to learn from";
  }
  if (column.unique === 0) {
    return "the column is empty";
  }
  if (column.unique === 1) {
    const only = column.sample_values[0];
    return only === undefined
      ? "every row holds the same value — a model cannot learn a constant"
      : `every row holds the same value (“${only}”) — a model cannot learn a constant`;
  }
  if (usable > 0 && column.unique >= usable) {
    return "every value is unique — this looks like an ID, not something to predict";
  }
  return "this column cannot be used as a target";
}

/**
 * An ID-like column: one distinct value per row, in a column whose kind or
 * name says "identifier". Compared against the non-null count rather than the
 * row count so a key column with a few blanks is still caught.
 */
function isIdLike(column: ColumnProfile, nRows: number): boolean {
  const usable = nonNullCount(column, nRows);
  const allDistinct = usable > 1 && column.unique >= usable;
  const namedLikeAnId = /(^|[^a-z])(id|uuid|guid|index|key|no|num)([^a-z]|$)/i.test(
    column.name
  );

  if (!allDistinct) return false;
  if (column.kind === "text" || column.kind === "categorical") return true;
  return namedLikeAnId;
}

function KindBadge({ column }: { column: ColumnProfile }) {
  return (
    <span className="inline-block rounded-control bg-paper-sunken px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-ink-muted">
      {column.kind}
    </span>
  );
}

function plural(count: number, one: string, many: string): string {
  return count === 1 ? one : many;
}

/* ------------------------------------------------------------------ */
/* Target                                                              */
/* ------------------------------------------------------------------ */

interface TargetColumnSelectorProps {
  columns: ColumnProfile[];
  /** Dataset row count — used to explain "every value is unique". */
  nRows: number;
  value: string;
  onChange: (columnName: string) => void;
  disabled?: boolean;
}

export function TargetColumnSelector({
  columns,
  nRows,
  value,
  onChange,
  disabled = false,
}: TargetColumnSelectorProps) {
  const selected = columns.find((column) => column.name === value) ?? null;
  const blocked = columns.filter((column) => !column.suitable_as_target);
  const usable = columns.filter((column) => column.suitable_as_target);

  return (
    <div>
      <select
        id="target-select"
        value={value}
        disabled={disabled || usable.length === 0}
        onChange={(event) => {
          const next = columns.find((column) => column.name === event.target.value);
          // Belt and braces: unusable options are already disabled, but a
          // target the backend will reject must never reach the request.
          if (next && !next.suitable_as_target) return;
          onChange(event.target.value);
        }}
        className="w-full p-3 bg-paper-sunken border border-rule rounded-control text-ink focus:outline-none focus:ring-2 focus:ring-primary disabled:cursor-not-allowed disabled:opacity-60"
      >
        <option value="">Choose the column to predict…</option>
        {columns.map((column) => (
          <option
            key={column.name}
            value={column.name}
            disabled={!column.suitable_as_target}
          >
            {column.suitable_as_target ? "" : `${WARNING_MARK} `}
            {column.name} — {column.kind}, {column.unique}{" "}
            {plural(column.unique, "unique value", "unique values")}
            {column.suitable_as_target ? "" : " — not usable"}
          </option>
        ))}
      </select>

      {usable.length === 0 && columns.length > 0 && (
        <p className="mt-2 text-sm font-medium text-state-bad">
          {WARNING_MARK} No column in this dataset can be used as a target. Every column is
          either constant, empty, or a unique identifier.
        </p>
      )}

      {selected && (
        <div className="mt-3 rounded-control border border-rule bg-paper-sunken p-4">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <span className="font-semibold text-ink">{selected.name}</span>
            <KindBadge column={selected} />
            <span className="text-sm text-ink-muted">
              {selected.unique}{" "}
              {plural(selected.unique, "unique value", "unique values")}
            </span>
            <span className="text-sm text-ink-muted">
              {selected.missing} missing ({selected.missing_pct}%)
            </span>
            <span className="text-sm text-ink-muted">dtype {selected.dtype}</span>
          </div>

          <p className="mt-2 text-sm text-ink-muted">
            {selected.suggested_task === null ? (
              <>
                Furina could not infer a task type from this column — choose
                Classification or Regression explicitly below.
              </>
            ) : (
              <>
                Suggested task:{" "}
                <span className="font-semibold text-ink">
                  {selected.suggested_task === "classification"
                    ? "Classification"
                    : "Regression"}
                </span>
              </>
            )}
          </p>

          {selected.sample_values.length > 0 && (
            <p className="mt-1 text-sm text-ink-muted">
              Sample values: {selected.sample_values.join(", ")}
            </p>
          )}

          {selected.suggested_task === "regression" &&
            selected.min !== null &&
            selected.max !== null && (
              <p className="mt-1 text-sm text-ink-muted">
                Range {selected.min} to {selected.max}
                {selected.mean !== null && <> · mean {selected.mean}</>}
              </p>
            )}

          {selected.top_values && selected.top_values.length > 0 && (
            <p className="mt-1 text-sm text-ink-muted">
              Most common:{" "}
              {selected.top_values
                .slice(0, 4)
                .map((top) => `${top.value} (${top.count})`)
                .join(", ")}
            </p>
          )}

          {selected.missing > 0 && (
            <p className="mt-2 text-sm font-medium text-state-warn">
              {WARNING_MARK} {selected.missing} {plural(selected.missing, "row has", "rows have")}{" "}
              no value for this column. Those rows are dropped before training.
            </p>
          )}
        </div>
      )}

      {blocked.length > 0 && (
        <details className="mt-3 rounded-control border border-rule bg-paper-sunken p-3">
          <summary className="cursor-pointer text-sm font-medium text-ink">
            {blocked.length} {plural(blocked.length, "column", "columns")} cannot
            be a target — why?
          </summary>
          <ul className="mt-2 space-y-1">
            {blocked.map((column) => (
              <li key={column.name} className="text-sm text-ink-muted">
                <span className="font-semibold text-ink">{column.name}</span>:{" "}
                {unsuitableReason(column, nRows)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Features                                                            */
/* ------------------------------------------------------------------ */

interface FeatureColumnSelectorProps {
  columns: ColumnProfile[];
  nRows: number;
  /** Excluded from the list automatically — you cannot predict a column from itself. */
  target: string;
  selected: string[];
  onChange: (columnNames: string[]) => void;
  disabled?: boolean;
}

export function FeatureColumnSelector({
  columns,
  nRows,
  target,
  selected,
  onChange,
  disabled = false,
}: FeatureColumnSelectorProps) {
  const available = columns.filter((column) => column.name !== target);
  const chosen = new Set(
    selected.filter((name) => name !== target && columns.some((c) => c.name === name))
  );

  /** Always emit in dataset column order so the request is stable. */
  const emit = (names: Set<string>) => {
    onChange(available.filter((column) => names.has(column.name)).map((c) => c.name));
  };

  const toggle = (name: string) => {
    const next = new Set(chosen);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    emit(next);
  };

  const idLike = available.filter((column) => isIdLike(column, nRows));
  const idLikeChosen = idLike.filter((column) => chosen.has(column.name));
  const idLikeExcluded = idLike.filter((column) => !chosen.has(column.name));

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={disabled || chosen.size === available.length}
          onClick={() => emit(new Set(available.map((column) => column.name)))}
          className="rounded-control bg-rule px-3 py-2 text-sm font-semibold text-ink transition-colors hover:bg-rule-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          Select all
        </button>
        <button
          type="button"
          disabled={disabled || chosen.size === 0}
          onClick={() => emit(new Set())}
          className="rounded-control bg-rule px-3 py-2 text-sm font-semibold text-ink transition-colors hover:bg-rule-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          Select none
        </button>
        <button
          type="button"
          disabled={disabled || idLikeChosen.length === 0}
          onClick={() => {
            const next = new Set(chosen);
            for (const column of idLikeChosen) next.delete(column.name);
            emit(next);
          }}
          title="Columns with one distinct value per row carry no signal — they let a model memorise instead of generalise."
          className="rounded-control bg-rule px-3 py-2 text-sm font-semibold text-ink transition-colors hover:bg-rule-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          Exclude ID-like columns
          {idLikeChosen.length > 0 && <> ({idLikeChosen.length})</>}
        </button>

        <span className="ml-auto text-sm font-medium text-ink-muted">
          {chosen.size} of {available.length}{" "}
          {plural(available.length, "column", "columns")} selected
        </span>
      </div>

      {target !== "" && (
        <p className="mb-2 text-sm text-ink-muted">
          <span className="font-semibold text-ink">{target}</span> is the target,
          so it is excluded from the features automatically.
        </p>
      )}

      {available.length === 0 ? (
        <p className="rounded-control border border-rule bg-paper-sunken p-4 text-sm text-ink-muted">
          There are no other columns to learn from. A dataset needs at least one
          feature column besides the target.
        </p>
      ) : (
        <div className="max-h-72 overflow-y-auto rounded-control border border-rule bg-paper-sunken p-2">
          {available.map((column) => {
            const checked = chosen.has(column.name);
            return (
              <label
                key={column.name}
                className={`flex cursor-pointer items-center gap-3 rounded-control px-2 py-2 transition-colors hover:bg-paper-sunken ${
                  disabled ? "cursor-not-allowed opacity-60" : ""
                }`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled}
                  onChange={() => toggle(column.name)}
                  className="h-5 w-5 flex-shrink-0 rounded-control border-rule text-accent focus:ring-accent"
                />
                <span className="min-w-0 flex-grow truncate font-medium text-ink">
                  {column.name}
                </span>
                <KindBadge column={column} />
                {isIdLike(column, nRows) && (
                  <span className="inline-block rounded-control bg-paper-sunken px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-state-warn">
                    {WARNING_MARK} ID-like
                  </span>
                )}
                <span className="w-28 flex-shrink-0 text-right text-xs text-ink-muted">
                  {column.unique} unique
                </span>
                <span className="w-24 flex-shrink-0 text-right text-xs text-ink-muted">
                  {column.missing_pct}% missing
                </span>
              </label>
            );
          })}
        </div>
      )}

      {idLikeExcluded.length > 0 && (
        <p className="mt-2 text-sm text-ink-muted">
          {idLikeExcluded.length} ID-like{" "}
          {plural(idLikeExcluded.length, "column is", "columns are")} currently
          excluded: {idLikeExcluded.map((column) => column.name).join(", ")}.
        </p>
      )}

      {chosen.size === 0 && available.length > 0 && (
        <p className="mt-2 text-sm font-medium text-state-bad">
          {WARNING_MARK} Select at least one feature column — a model needs something to learn
          from.
        </p>
      )}
    </div>
  );
}
