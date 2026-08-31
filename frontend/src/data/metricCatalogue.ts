/**
 * The metric catalogue: what each number in a run's metric blob means, how to
 * read it, and which direction is better.
 *
 * This lives outside any one component because two screens now depend on it --
 * the evaluation page's stat tiles and the dashboard's run comparison. A second
 * copy would drift, and a dashboard that sorted "log loss" descending because it
 * did not know lower is better would be quietly, confidently wrong.
 */
import type { MetricMap, TaskType } from "../interface/MlTypes";

/** How a metric's number should be read. */
export type MetricFormat =
  /** 0..1 — safe to also show as a percentage. */
  | "ratio"
  /** -1..1 — a percentage would be nonsense. */
  | "signed"
  /** In the target's own units — percentages here are the classic bug. */
  | "raw";

export interface MetricDef {
  key: string;
  label: string;
  explain: string;
  format: MetricFormat;
  /** Which direction is better, used for the train-vs-test overfitting hint. */
  higherIsBetter: boolean;
  /** True when the metric lives on a fixed scale, so an absolute gap is meaningful. */
  bounded: boolean;
  /** Shown instead of a number when the backend sent null. */
  nullReason: string;
  /**
   * Overrides the sublabel under the big number. Needed because "raw" means
   * "do not percentage this", not "this is in the target's units" — log loss is
   * a classification metric whose target is a category with no units at all.
   */
  note?: string;
}

export const CLASSIFICATION_METRICS: MetricDef[] = [
  {
    key: "accuracy",
    label: "Accuracy",
    explain:
      "The share of test rows the model labelled correctly. Misleading when one class is far more common than the others.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report an accuracy score.",
  },
  {
    key: "precision",
    label: "Precision",
    explain:
      "When the model calls a row a given class, this is how often it is right. High precision means few false alarms.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report a precision score.",
  },
  {
    key: "recall",
    label: "Recall",
    explain:
      "Of the rows that truly belong to a class, this is the share the model actually found. High recall means few missed cases.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report a recall score.",
  },
  {
    key: "f1",
    label: "F1 score",
    explain:
      "One number balancing precision and recall, so a model that is good at only one of them cannot score well.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report an F1 score.",
  },
  {
    key: "f2",
    label: "F2 score",
    explain:
      "Like F1, but recall counts four times as much. Use it when missing a real case costs more than a false alarm.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report an F2 score.",
  },
  {
    key: "balanced_accuracy",
    label: "Balanced accuracy",
    explain:
      "Accuracy after giving every class equal weight, so a rare class cannot be hidden behind a common one.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report a balanced accuracy score.",
  },
  {
    key: "roc_auc",
    label: "ROC AUC",
    explain:
      "The chance the model ranks a random positive row above a random negative one. 0.5 is a coin flip, 1.0 is perfect.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason:
      "Needs predicted probabilities and every class present in the test split — this model or split has neither.",
  },
  {
    key: "log_loss",
    label: "Log loss",
    explain:
      "How badly the model's confidence was misplaced. Lower is better, and confident-but-wrong predictions are punished hardest.",
    format: "raw",
    higherIsBetter: false,
    bounded: false,
    note: "a cross-entropy — 0 is perfect, lower is better",
    nullReason:
      "Needs predicted probabilities and every class present in the test split — this model or split has neither.",
  },
  {
    key: "matthews_corrcoef",
    label: "Matthews correlation (MCC)",
    explain:
      "A balanced score from -1 to +1: +1 is perfect, 0 is random guessing, below 0 is worse than guessing.",
    format: "signed",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report a Matthews correlation coefficient.",
  },
];

export const REGRESSION_METRICS: MetricDef[] = [
  {
    key: "r2",
    label: "R² score",
    explain:
      "The share of the target's variation the model explains. 1.0 is perfect; 0 is no better than always predicting the average.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report an R² score.",
  },
  {
    key: "mae",
    label: "Mean absolute error (MAE)",
    explain:
      "On average the prediction misses by this much, in the target's own units. Not a percentage.",
    format: "raw",
    higherIsBetter: false,
    bounded: false,
    nullReason: "This run did not report a mean absolute error.",
  },
  {
    key: "rmse",
    label: "Root mean squared error (RMSE)",
    explain:
      "Like MAE but big misses count far more, so it rises when the model is occasionally very wrong. Target's own units.",
    format: "raw",
    higherIsBetter: false,
    bounded: false,
    nullReason: "This run did not report a root mean squared error.",
  },
  {
    key: "mse",
    label: "Mean squared error (MSE)",
    explain:
      "The average squared miss — RMSE before the square root. Good for comparing models, but its units are the target squared.",
    format: "raw",
    higherIsBetter: false,
    bounded: false,
    note: "in the target's units, squared",
    nullReason: "This run did not report a mean squared error.",
  },
  {
    key: "mape",
    label: "Mean absolute percentage error (MAPE)",
    explain:
      "The average miss as a share of the true value, which makes it comparable across targets of very different sizes.",
    format: "ratio",
    higherIsBetter: false,
    bounded: false,
    nullReason:
      "Undefined here because at least one true value is 0, and you cannot take a percentage of zero.",
  },
  {
    key: "explained_variance",
    label: "Explained variance",
    explain:
      "How much of the target's spread the model captures. It matches R² unless the predictions are biased consistently high or low.",
    format: "ratio",
    higherIsBetter: true,
    bounded: true,
    nullReason: "This run did not report an explained variance score.",
  },
  {
    key: "max_error",
    label: "Max error",
    explain:
      "The single worst miss anywhere in the test set, in the target's own units. Your realistic worst case.",
    format: "raw",
    higherIsBetter: false,
    bounded: false,
    nullReason: "This run did not report a maximum error.",
  },
];

export function metricDefsFor(taskType: TaskType): MetricDef[] {
  return taskType === "classification"
    ? CLASSIFICATION_METRICS
    : REGRESSION_METRICS;
}

/** Readable digits for numbers that may be 0.0003 or 48,201.7. */
export function formatNumber(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude < 0.001) return value.toExponential(2);
  if (magnitude >= 10000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  if (magnitude >= 100) return value.toFixed(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export function readMetric(metrics: MetricMap, key: string): number | null {
  const value = metrics[key];
  if (value === null || value === undefined) return null;
  return Number.isFinite(value) ? value : null;
}

/** The big number on a tile, plus the secondary reading under it. */
export function formatValue(
  def: MetricDef,
  value: number
): { main: string; note: string | null } {
  if (def.format === "raw") {
    // Raw units. Never percentaged — an MAE of 0.42 is 0.42 dollars, not 42%.
    return {
      main: formatNumber(value),
      note: def.note ?? "in the target's own units",
    };
  }
  if (def.format === "signed") {
    return { main: value.toFixed(3), note: "on a -1 to +1 scale" };
  }
  if (value < 0) {
    return {
      main: value.toFixed(3),
      note: "below 0 — worse than always predicting the average",
    };
  }
  // Percentage stays on above 1.0 too: a MAPE of 2.5 is a 250% error, and a
  // bare "2.5" on a tile labelled "percentage error" reads as 2.5%.
  return { main: value.toFixed(3), note: `${(value * 100).toFixed(1)}%` };
}

/** Same value, compact, for the comparison table. */
export function formatCompact(def: MetricDef, value: number | null): string {
  if (value === null) return "n/a";
  if (def.format === "raw") return formatNumber(value);
  return value.toFixed(3);
}

/**
 * A material gap means the model learned the training rows rather than the
 * pattern. Bounded 0..1 metrics use an absolute gap; error metrics in raw
 * units have to use a relative one, because 40000 vs 41000 is not overfitting.
 */
export function isOverfitting(def: MetricDef, train: number, test: number): boolean {
  // Whichever direction is better, the suspicious case is the same: the train
  // score is ahead of the test score.
  const gap = def.higherIsBetter ? train - test : test - train;
  if (gap <= 0) return false;
  if (def.bounded) return gap > 0.05;
  const reference = Math.abs(train);
  if (reference === 0) return test !== 0;
  return gap / reference > 0.1;
}
