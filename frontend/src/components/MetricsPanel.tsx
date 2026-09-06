/**
 * Headline metrics for a finished training run.
 *
 * This is a stat-tile row, deliberately not a chart: the metrics are unrelated
 * quantities on different scales, so bars next to each other would invite a
 * comparison that means nothing. Every tile carries a plain-English sentence
 * because a beginner has no idea what MCC or explained variance is.
 */
import type { MetricMap, TaskType } from "../interface/MlTypes";
import { WARNING_MARK } from "../explainer/marks";
import {
  formatCompact,
  formatValue,
  isOverfitting,
  metricDefsFor,
  readMetric,
} from "../data/metricCatalogue";

interface MetricsPanelProps {
  taskType: TaskType;
  metrics: MetricMap;
  trainMetrics: MetricMap;
}

export default function MetricsPanel({
  taskType,
  metrics,
  trainMetrics,
}: MetricsPanelProps) {
  const defs = metricDefsFor(taskType);
  const rows = defs.map((def) => {
    const test = readMetric(metrics, def.key);
    const train = readMetric(trainMetrics, def.key);
    return {
      def,
      test,
      train,
      overfit:
        train !== null && test !== null && isOverfitting(def, train, test),
    };
  });
  const overfitCount = rows.filter((row) => row.overfit).length;
  const hasAnyValue = rows.some((row) => row.test !== null);

  return (
    <section>
      <h2 className="font-serif text-h3 text-ink mb-4">
        Test-set metrics
      </h2>

      {!hasAnyValue && (
        <p className="mb-4 text-small text-ink-muted max-w-measure">
          This run finished without producing any metrics. That usually means
          training failed — check the error box above.
        </p>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {rows.map(({ def, test }) => {
          const formatted = test === null ? null : formatValue(def, test);
          return (
            <div
              key={def.key}
              className="bg-paper-sunken p-4"
            >
              <p className="text-h2 font-mono text-ink leading-tight tabular">
                {formatted ? formatted.main : "n/a"}
              </p>
              <p className="mt-1 text-sm font-semibold text-ink">{def.label}</p>
              {formatted?.note && (
                <p className="text-xs font-mono text-ink-muted">
                  {formatted.note}
                </p>
              )}
              <p className="mt-2 text-xs leading-relaxed text-ink-muted">
                {test === null ? def.nullReason : def.explain}
              </p>
            </div>
          );
        })}
      </div>

      <h3 className="mt-8 mb-3 text-lg font-semibold text-ink">
        Train vs test
      </h3>
      <p className="mb-3 text-sm text-ink-muted">
        The same metrics scored on the rows the model learned from
        <span className="mx-1 inline-block h-2.5 w-2.5 rounded-control bg-series-1 align-middle" />
        and on the rows it had never seen
        <span className="mx-1 inline-block h-2.5 w-2.5 rounded-control bg-series-2 align-middle" />
        . Only the test column tells you how the model will behave on new data.
      </p>

      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-rule text-left text-ink-muted">
              <th className="py-2 pr-4 font-semibold">Metric</th>
              <th className="py-2 pr-4 font-semibold">
                <span className="mr-2 inline-block h-2.5 w-2.5 rounded-control bg-series-1 align-middle" />
                Train
              </th>
              <th className="py-2 pr-4 font-semibold">
                <span className="mr-2 inline-block h-2.5 w-2.5 rounded-control bg-series-2 align-middle" />
                Test
              </th>
              <th className="py-2 font-semibold">Reading</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ def, train, test, overfit }) => (
              <tr
                key={def.key}
                className="border-b border-rule hover:bg-paper-sunken"
              >
                <td className="py-2 pr-4 text-ink">{def.label}</td>
                <td className="py-2 pr-4 font-mono text-ink">
                  {formatCompact(def, train)}
                </td>
                <td className="py-2 pr-4 font-mono text-ink">
                  {formatCompact(def, test)}
                </td>
                <td className="py-2 text-xs text-ink-muted">
                  {overfit ? (
                    <span className="font-semibold text-state-warn">
                      {WARNING_MARK} Overfitting — much better on train than on test
                    </span>
                  ) : train === null || test === null ? (
                    "not comparable"
                  ) : (
                    "train and test agree"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {overfitCount > 0 && (
        <p className="mt-3 rounded-control border border-state-warn/40 bg-state-warn/10 p-3 text-sm text-state-warn">
          {WARNING_MARK} {overfitCount} metric{overfitCount === 1 ? "" : "s"} scored much
          better on the training rows than on the held-out rows. The model has
          partly memorised the training data. Try a simpler model, stronger
          regularisation, or more rows.
        </p>
      )}
    </section>
  );
}
