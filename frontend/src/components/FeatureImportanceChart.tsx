/**
 * Feature importance as a horizontal bar list.
 *
 * One series, so one hue and no legend — the heading already says what the
 * bars are. Every bar prints its own value, so the chart is readable without
 * relying on length alone.
 */
import { useState } from "react";

import type { FeatureImportance } from "../interface/MlTypes";

const TOP_N = 15;

function formatImportance(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude < 0.001) return value.toExponential(2);
  if (magnitude >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

interface FeatureImportanceChartProps {
  importance: FeatureImportance[] | null;
  /** Named in the "why is this missing" message so the user knows what to change. */
  algorithmLabel: string;
}

export default function FeatureImportanceChart({
  importance,
  algorithmLabel,
}: FeatureImportanceChartProps) {
  const [showAll, setShowAll] = useState(false);

  if (importance === null || importance.length === 0) {
    return (
      <section>
        <h2 className="text-xl font-semibold text-ink mb-4">
          Feature importance
        </h2>
        <p className="rounded-control border border-rule bg-paper p-4 text-sm text-ink-muted">
          {algorithmLabel} does not expose feature importances, so there is
          nothing to rank here. Tree-based models (random forest, gradient
          boosting, decision tree) and linear models do expose them — retrain
          with one of those if you need to know which columns carried the
          prediction.
        </p>
      </section>
    );
  }

  // Sorted and scaled by the same quantity, so bar length decreases all the
  // way down the list even if the backend hands back signed coefficients.
  const sorted = [...importance].sort(
    (a, b) => Math.abs(b.importance) - Math.abs(a.importance)
  );
  const maxMagnitude = Math.abs(sorted[0]?.importance ?? 0);
  const visible = showAll ? sorted : sorted.slice(0, TOP_N);
  const hiddenCount = sorted.length - visible.length;

  return (
    <section>
      <h2 className="text-xl font-semibold text-ink mb-2">
        Feature importance
      </h2>
      <p className="mb-4 text-sm text-ink-muted">
        How much each input column contributed to the model's predictions,
        largest first. This says a column was useful, not that it causes the
        outcome.
      </p>

      <ul className="space-y-2">
        {visible.map((entry) => {
          const width =
            maxMagnitude > 0
              ? Math.max(
                  1,
                  (Math.abs(entry.importance) / maxMagnitude) * 100
                )
              : 0;
          return (
            <li key={entry.feature} className="flex items-center gap-3">
              <span
                className="w-1/3 max-w-[16rem] shrink-0 truncate text-sm text-ink"
                title={entry.feature}
              >
                {entry.feature}
              </span>
              <span className="h-3 flex-1 rounded-[4px] bg-paper-sunken">
                <span
                  className="block h-3 rounded-[4px] bg-series-1 transition-opacity hover:opacity-80"
                  style={{ width: `${width}%` }}
                  title={`${entry.feature}: ${formatImportance(entry.importance)}`}
                />
              </span>
              <span className="w-24 shrink-0 text-right font-mono text-xs text-ink">
                {formatImportance(entry.importance)}
              </span>
            </li>
          );
        })}
      </ul>

      {sorted.length > TOP_N && (
        <button
          type="button"
          onClick={() => setShowAll((previous) => !previous)}
          className="mt-4 rounded-control bg-rule px-4 py-2 text-sm font-semibold text-ink transition-colors hover:bg-rule-strong"
        >
          {showAll
            ? `Show top ${TOP_N} only`
            : `Show all ${sorted.length} features (${hiddenCount} hidden)`}
        </button>
      )}

      <p className="mt-3 text-xs text-ink-muted">
        Text and category columns are one-hot encoded before training, so they
        appear here once per value as <code>column_value</code> — for example{" "}
        <code>city_Jakarta</code> is the "city is Jakarta" indicator, not the
        whole <code>city</code> column.
      </p>
    </section>
  );
}
