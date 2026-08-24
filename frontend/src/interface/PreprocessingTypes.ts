/**
 * UI-side types for the Data Preprocessing page, plus the catalogue of
 * preprocessing operations the recipe builder renders.
 *
 * Nothing here duplicates an API shape: every wire type is imported from
 * `MlTypes.ts` and re-used as-is. The recipe the user builds IS a
 * `PreprocessStep[]` — the exact body the API expects — so there is no
 * translation layer to drift out of sync.
 *
 * The operation catalogue lives here rather than inside the checkbox component
 * because two files need it: the recipe builder (to render every row) and the
 * page (to print human labels in the preprocessing report).
 */
import type {
  ColumnKind,
  ColumnProfile,
  DatasetDetail,
  DatasetPreview,
  PreprocessOp,
  PreprocessStep,
} from "./MlTypes";

/* ------------------------------------------------------------------ */
/* Component props                                                     */
/* ------------------------------------------------------------------ */

export interface InsertDataFileProps {
  /** Called with the freshly created dataset once the upload succeeds. */
  onUploadSuccess: (dataset: DatasetDetail) => void;
}

export interface TableViewerProps {
  preview: DatasetPreview;
  title?: string;
  subtitle?: string;
  /** Total rows in the dataset, when the preview is only the first N of them. */
  totalRows?: number;
}

export interface DatasetProfileProps {
  columns: ColumnProfile[];
}

export interface PreprocessingCheckboxProps {
  /** Real columns of the active dataset; drives every column picker. */
  columns: ColumnProfile[];
  /** The recipe, in execution order. Only enabled steps are present. */
  steps: PreprocessStep[];
  onStepsChange: (steps: PreprocessStep[]) => void;
  /** True while a preview/apply request is in flight. */
  disabled?: boolean;
}

/* ------------------------------------------------------------------ */
/* Small local UI types                                                */
/* ------------------------------------------------------------------ */

/** Columns of the profile table the user can sort by. */
export type ProfileSortKey = "name" | "kind" | "missing" | "unique";

export type SortDirection = "asc" | "desc";

/**
 * The delimiter choices offered on the upload form; values go on the wire.
 *
 * Tab travels as the alias `"tab"`, never as a literal "\t": DRF's CharField
 * trims whitespace before validation, so a bare tab reaches the serializer as
 * "" and is downgraded to auto-detect — the one setting the user picked would
 * be the one thrown away. `"tab"` survives that and `DELIMITER_CHOICES` maps it
 * to "\t", which is the value the backend is meant to act on.
 *
 * Tab still auto-detects today, for a second reason this file cannot reach:
 * `core/ml/io.py::_resolve_delimiter` looks its argument up as
 * `requested.strip().lower()`, and the alias table maps "" to "auto", so even a
 * correctly resolved "\t" collapses back to auto-detect. Sending "tab" is the
 * half of that fix which lives on this side of the wire.
 */
export type DelimiterChoice = "auto" | "," | ";" | "tab" | "|";

export interface DelimiterOption {
  value: DelimiterChoice;
  label: string;
}

export const DELIMITER_OPTIONS: DelimiterOption[] = [
  { value: "auto", label: "Auto-detect" },
  { value: ",", label: "Comma  ," },
  { value: ";", label: "Semicolon  ;" },
  { value: "tab", label: "Tab" },
  { value: "|", label: "Pipe  |" },
];

/** Mirrors `core.ml.io.MAX_UPLOAD_BYTES`; rejected client-side before upload. */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

export const ACCEPTED_EXTENSIONS = [".csv", ".tsv", ".txt"];

/* ------------------------------------------------------------------ */
/* Operation catalogue                                                 */
/* ------------------------------------------------------------------ */

/** How a parameter is entered and what JSON type it becomes. */
export type OpParamKind = "int" | "float" | "text" | "column";

export interface OpParamSpec {
  /** Exact key inside `PreprocessStep.params`. */
  name: string;
  label: string;
  kind: OpParamKind;
  /** Held as a string by the form input and coerced on submit. */
  defaultValue: string;
  min: number | null;
  max: number | null;
  help: string;
}

export interface OpMeta {
  label: string;
  /** One plain-English sentence shown under the row's label. */
  help: string;
  /** Column kinds the op can act on. null means "any column". */
  appliesTo: ColumnKind[] | null;
  /** True when running with no explicit column selection would be destructive. */
  requiresColumns: boolean;
  params: OpParamSpec[];
}

const NUMERIC: ColumnKind[] = ["numeric"];
const TEXTUAL: ColumnKind[] = ["categorical", "text"];
const DISCRETE: ColumnKind[] = ["categorical", "boolean", "text"];

/**
 * Every op in the API contract, keyed by op so the compiler rejects the file
 * if the backend ever gains one and this map is not updated.
 */
export const PREPROCESS_OP_META: Record<PreprocessOp, OpMeta> = {
  remove_duplicates: {
    label: "Remove duplicate rows",
    help: "Drops rows that are identical to an earlier row. Pick columns to compare only those.",
    appliesTo: null,
    requiresColumns: false,
    params: [],
  },
  drop_columns: {
    label: "Drop columns",
    help: "Permanently removes the selected columns. Useful for ids and free-text notes.",
    appliesTo: null,
    requiresColumns: true,
    params: [],
  },
  drop_missing_rows: {
    label: "Drop rows with missing values",
    help: "Deletes any row that is missing a value in the selected columns.",
    appliesTo: null,
    requiresColumns: false,
    params: [],
  },
  fill_mean: {
    label: "Fill missing values with the mean",
    help: "Replaces gaps in a numeric column with that column's average.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [],
  },
  fill_median: {
    label: "Fill missing values with the median",
    help: "Like the mean, but unaffected by a few extreme values.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [],
  },
  fill_mode: {
    label: "Fill missing values with the mode",
    help: "Replaces gaps with the most common value in the column.",
    appliesTo: null,
    requiresColumns: false,
    params: [],
  },
  fill_constant: {
    label: "Fill missing values with a constant",
    help: "Replaces every gap with a value you choose.",
    appliesTo: null,
    requiresColumns: false,
    params: [
      {
        name: "value",
        label: "Value",
        kind: "text",
        defaultValue: "0",
        min: null,
        max: null,
        help: "Written into every missing cell of the selected columns.",
      },
    ],
  },
  fill_empty_string: {
    label: "Fill missing text with an empty string",
    help: "Turns missing text into \"\" so text steps downstream do not skip the row.",
    appliesTo: TEXTUAL,
    requiresColumns: false,
    params: [],
  },
  strip_whitespace: {
    label: "Trim surrounding whitespace",
    help: "Removes leading and trailing spaces so \" cat \" and \"cat\" stop being two categories.",
    appliesTo: TEXTUAL,
    requiresColumns: false,
    params: [],
  },
  lowercase: {
    label: "Lowercase text",
    help: "Folds \"Cat\", \"CAT\" and \"cat\" into a single value.",
    appliesTo: TEXTUAL,
    requiresColumns: false,
    params: [],
  },
  remove_punctuation: {
    label: "Remove punctuation",
    help: "Strips punctuation characters out of text columns.",
    appliesTo: TEXTUAL,
    requiresColumns: false,
    params: [],
  },
  remove_outliers_iqr: {
    label: "Remove outliers (IQR)",
    help: "Deletes rows lying more than 1.5 interquartile ranges outside the middle 50%.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [],
  },
  remove_outliers_zscore: {
    label: "Remove outliers (z-score)",
    help: "Deletes rows whose value sits more than N standard deviations from the mean.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [
      {
        name: "threshold",
        label: "Threshold",
        kind: "float",
        defaultValue: "3",
        min: 0.5,
        max: 10,
        help: "Standard deviations from the mean. 3 is the usual choice.",
      },
    ],
  },
  minmax_scale: {
    label: "Min-max scale",
    help: "Squeezes each numeric column into the 0-1 range.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [],
  },
  standard_scale: {
    label: "Standardise (z-score scale)",
    help: "Re-centres each numeric column on 0 with a standard deviation of 1.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [],
  },
  robust_scale: {
    label: "Robust scale",
    help: "Scales using the median and interquartile range, so outliers matter less.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [],
  },
  one_hot_encode: {
    label: "One-hot encode",
    help: "Turns each category into its own 0/1 column. Best for unordered categories.",
    appliesTo: DISCRETE,
    requiresColumns: false,
    params: [],
  },
  label_encode: {
    label: "Label encode",
    help: "Replaces each category with an integer code. Compact, but implies an order.",
    appliesTo: DISCRETE,
    requiresColumns: false,
    params: [],
  },
  pca: {
    label: "PCA",
    help: "Replaces the numeric columns with a few combined components that keep most of the variation.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [
      {
        name: "n_components",
        label: "Components",
        kind: "int",
        defaultValue: "2",
        min: 1,
        max: 100,
        help: "How many components to keep. Must not exceed the number of numeric columns.",
      },
    ],
  },
  select_k_best: {
    label: "Select K best features",
    help: "Keeps only the K numeric columns most related to the target column.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [
      {
        name: "k",
        label: "K",
        kind: "int",
        defaultValue: "5",
        min: 1,
        max: 500,
        help: "How many features to keep.",
      },
      {
        name: "target",
        label: "Target column",
        kind: "column",
        defaultValue: "",
        min: null,
        max: null,
        help: "The column you will later predict. It is scored against, not dropped.",
      },
    ],
  },
  lda: {
    label: "LDA",
    help: "Projects the numeric columns onto the axes that separate the target's classes best.",
    appliesTo: NUMERIC,
    requiresColumns: false,
    params: [
      {
        name: "n_components",
        label: "Components",
        kind: "int",
        defaultValue: "1",
        min: 1,
        max: 100,
        help: "At most (number of classes - 1).",
      },
      {
        name: "target",
        label: "Target column",
        kind: "column",
        defaultValue: "",
        min: null,
        max: null,
        help: "The categorical column whose classes LDA should separate.",
      },
    ],
  },
};

export interface PreprocessGroup {
  title: string;
  /** Why this group exists, one line, shown under the heading. */
  blurb: string;
  ops: PreprocessOp[];
}

/** The page's section headings, covering all 21 contract ops exactly once. */
export const PREPROCESS_GROUPS: PreprocessGroup[] = [
  {
    title: "Data Cleaning",
    blurb: "Fix gaps, duplicates and extreme values before anything else runs.",
    ops: [
      "remove_duplicates",
      "drop_columns",
      "drop_missing_rows",
      "fill_mean",
      "fill_median",
      "fill_mode",
      "fill_constant",
      "fill_empty_string",
      "remove_outliers_iqr",
      "remove_outliers_zscore",
    ],
  },
  {
    title: "Data Transformation",
    blurb: "Put numeric columns on a comparable scale.",
    ops: ["minmax_scale", "standard_scale", "robust_scale"],
  },
  {
    title: "Data Encoding",
    blurb: "Turn categories into numbers a model can read.",
    ops: ["one_hot_encode", "label_encode"],
  },
  {
    title: "Text cleaning",
    blurb: "Normalise text so near-identical values stop counting as different.",
    ops: ["strip_whitespace", "lowercase", "remove_punctuation"],
  },
  {
    title: "Feature Engineering",
    blurb: "Reduce or select columns once the data is clean.",
    ops: ["pca", "select_k_best", "lda"],
  },
];

/** Human label for an op string coming back in a PreprocessReport. */
export function opLabel(op: string): string {
  const known = Object.prototype.hasOwnProperty.call(PREPROCESS_OP_META, op)
    ? PREPROCESS_OP_META[op as PreprocessOp]
    : null;
  return known ? known.label : op.replace(/_/g, " ");
}

/** Columns an op can legitimately be pointed at, given the real profile. */
export function applicableColumns(
  op: PreprocessOp,
  columns: ColumnProfile[]
): ColumnProfile[] {
  const kinds = PREPROCESS_OP_META[op].appliesTo;
  if (kinds === null) return columns;
  return columns.filter((column) => kinds.includes(column.kind));
}

/**
 * Problems that would make the API reject or silently skip the recipe.
 * Returned as human sentences so the page can render them directly.
 */
export function validateRecipe(
  steps: PreprocessStep[],
  columns: ColumnProfile[]
): string[] {
  const known = new Set(columns.map((column) => column.name));
  const problems: string[] = [];

  steps.forEach((step) => {
    const meta = PREPROCESS_OP_META[step.op];

    if (meta.requiresColumns && (!step.columns || step.columns.length === 0)) {
      problems.push(`${meta.label}: choose at least one column.`);
    }

    const missing = (step.columns ?? []).filter((name) => !known.has(name));
    if (missing.length > 0) {
      problems.push(
        `${meta.label}: ${missing.join(", ")} is not a column of this dataset.`
      );
    }

    meta.params.forEach((param) => {
      const value = step.params ? step.params[param.name] : undefined;
      if (param.kind === "column") {
        if (typeof value !== "string" || value === "") {
          problems.push(`${meta.label}: choose a ${param.label.toLowerCase()}.`);
        } else if (!known.has(value)) {
          problems.push(
            `${meta.label}: ${value} is not a column of this dataset.`
          );
        }
        return;
      }
      if (param.kind === "int" || param.kind === "float") {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          problems.push(`${meta.label}: ${param.label.toLowerCase()} must be a number.`);
        }
      }
    });
  });

  return problems;
}

/**
 * Drop column references that no longer exist, so a recipe built against the
 * previous dataset does not 400 the moment the user switches datasets.
 */
export function pruneRecipe(
  steps: PreprocessStep[],
  columns: ColumnProfile[]
): PreprocessStep[] {
  const known = new Set(columns.map((column) => column.name));

  return steps.map((step) => {
    const meta = PREPROCESS_OP_META[step.op];
    const kept = step.columns
      ? step.columns.filter((name) => known.has(name))
      : null;

    const params = step.params ? { ...step.params } : null;
    if (params) {
      meta.params.forEach((param) => {
        if (param.kind !== "column") return;
        const value = params[param.name];
        if (typeof value !== "string" || !known.has(value)) {
          params[param.name] = "";
        }
      });
    }

    return {
      op: step.op,
      columns:
        kept === null
          ? null
          : kept.length === 0 && !meta.requiresColumns
            ? null
            : kept,
      params,
    };
  });
}
