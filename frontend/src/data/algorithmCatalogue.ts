/**
 * The estimator catalogue, and the grouping both pages use to display it.
 *
 * `/api/algorithms/` is the authority. `DOCUMENTED_FAMILIES` is the same list
 * as published in the API contract, kept here so a page can render something
 * true and complete on its first paint — and still say something useful when
 * the backend is unreachable. It is a fallback, never a substitute: whenever
 * the server answers, the live list wins.
 */
import type { AlgorithmSpec } from "../interface/MlTypes";

export interface AlgorithmFamily {
  family: string;
  labels: string[];
}

export const DOCUMENTED_FAMILIES: AlgorithmFamily[] = [
  {
    family: "Linear models",
    labels: [
      "Logistic Regression",
      "Ridge Classifier",
      "Linear Regression",
      "Ridge Regression",
      "Lasso Regression",
      "Elastic Net",
      "Stochastic Gradient Descent",
    ],
  },
  {
    family: "Support vector machines",
    labels: ["Support Vector Machine (SVC / SVR)"],
  },
  {
    family: "Trees and ensembles",
    labels: [
      "Decision Tree",
      "Random Forest",
      "Extra Trees",
      "Gradient Boosting",
      "Histogram Gradient Boosting",
      "AdaBoost",
      "XGBoost",
      "LightGBM",
    ],
  },
  { family: "Nearest neighbours", labels: ["K-Nearest Neighbours"] },
  { family: "Naive Bayes", labels: ["Gaussian Naive Bayes"] },
  { family: "Neural networks", labels: ["Neural Network (MLP)"] },
];

/** How many estimators the documented catalogue describes. */
export const DOCUMENTED_COUNT = DOCUMENTED_FAMILIES.reduce(
  (total, family) => total + family.labels.length,
  0
);

/** Bucket a live `/api/algorithms/` response by family, preserving its order. */
export function groupByFamily(algorithms: AlgorithmSpec[]): AlgorithmFamily[] {
  const families = new Map<string, string[]>();
  for (const algorithm of algorithms) {
    const bucket = families.get(algorithm.family);
    if (bucket) bucket.push(algorithm.label);
    else families.set(algorithm.family, [algorithm.label]);
  }
  return Array.from(families, ([family, labels]) => ({ family, labels }));
}
