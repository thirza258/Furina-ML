/**
 * A ranked bar chart comparing one metric across training runs.
 *
 * Deliberately not a library chart, and deliberately not SVG: the row count is
 * variable and the labels are run names of unpredictable length, which is
 * exactly the case where SVG text either overflows or gets scaled by a
 * viewBox. Plain elements on a shared scale stay responsive, keep real hover
 * and focus targets, and let the labels wrap.
 *
 * Design decisions worth keeping:
 *  - Bars are sorted best-first by the metric's own direction, so for log loss
 *    or MAE the *shortest* bar leads. The axis says so out loud, because a
 *    reader who assumes longer-is-better would read the chart backwards.
 *  - Two series only appear when training scores are asked for. One series
 *    needs no legend — the heading names it.
 *  - Every bar carries its number. `series-2` sits at 2.6:1 against the paper,
 *    under the 3:1 fill threshold, so the value label is the required relief,
 *    not decoration.
 */
import type { TrainingRunSummary } from "../interface/MlTypes";
import { formatCompact, readMetric, type MetricDef } from "../data/metricCatalogue";

export interface MetricBarRow {
  run: TrainingRunSummary;
  value: number;
}

interface MetricBarsProps {
  rows: MetricBarRow[];
  def: MetricDef;
  showTrain: boolean;
  onOpen: (run: TrainingRunSummary) => void;
}

/** Where the axis starts and ends, given what the metric can actually be. */
function domainFor(def: MetricDef, values: number[]): { min: number; max: number } {
  if (def.format === "signed") return { min: -1, max: 1 };
  const highest = Math.max(...values, 0);
  const lowest = Math.min(...values, 0);
  if (def.format === "ratio" && lowest >= 0 && highest <= 1) {
    // A bounded 0..1 score is only honest against the full scale: rescaling to
    // the observed range turns a 0.71-vs-0.70 tie into a landslide.
    return { min: 0, max: 1 };
  }
  if (highest === lowest) return { min: Math.min(0, lowest), max: highest || 1 };
  return { min: Math.min(0, lowest), max: highest * 1.05 };
}

function ticksFor(min: number, max: number): number[] {
  const span = max - min;
  if (span <= 0) return [min];
  const step = span / 4;
  return [0, 1, 2, 3, 4].map((i) => min + step * i);
}

function tickLabel(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 10000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

export default function MetricBars({ rows, def, showTrain, onOpen }: MetricBarsProps) {
  const testValues = rows.map((row) => row.value);
  const trainValues = showTrain
    ? rows
        .map((row) => readMetric(row.run.train_metrics, def.key))
        .filter((value): value is number => value !== null)
    : [];
  const { min, max } = domainFor(def, [...testValues, ...trainValues]);
  const span = max - min || 1;
  const position = (value: number) =>
    Math.max(0, Math.min(100, ((value - min) / span) * 100));

  const ticks = ticksFor(min, max);

  return (
    <figure className="m-0">
      {/* A legend exists only once identity is ambiguous. */}
      {showTrain && (
        <div className="flex items-center gap-5 mb-4 font-sans text-caption text-ink-muted">
          <span className="flex items-center gap-2">
            <span aria-hidden="true" className="inline-block h-2.5 w-4 bg-series-1" />
            Train split
          </span>
          <span className="flex items-center gap-2">
            <span aria-hidden="true" className="inline-block h-2.5 w-4 bg-series-2" />
            Test split
          </span>
        </div>
      )}

      <ul className="space-y-3">
        {rows.map(({ run, value }) => {
          const trainValue = showTrain ? readMetric(run.train_metrics, def.key) : null;
          const label = run.name?.trim() || run.algorithm_label || run.algorithm;
          return (
            <li key={run.id} className="grid gap-x-3 sm:grid-cols-[minmax(6rem,11rem)_1fr]">
              <button
                type="button"
                onClick={() => onOpen(run)}
                title={`${label} — open the full evaluation`}
                className="text-left font-sans text-small text-primary hover:underline truncate cursor-pointer"
              >
                {label}
              </button>

              {/* No gridlines. Every bar is labelled with its own value and
                  the axis sits under the whole group, so ruling the plot would
                  add lines without adding information. */}
              <div className="relative">
                <div className="relative space-y-0.5 py-0.5">
                  {showTrain && trainValue !== null && (
                    <Bar
                      widthPercent={position(trainValue)}
                      tone="bg-series-1"
                      text={formatCompact(def, trainValue)}
                      title={`${label} — ${def.label} on the training split: ${formatCompact(def, trainValue)}`}
                    />
                  )}
                  <Bar
                    widthPercent={position(value)}
                    tone={showTrain ? "bg-series-2" : "bg-ramp-4"}
                    text={formatCompact(def, value)}
                    title={`${label} — ${def.label} on the test split: ${formatCompact(def, value)}`}
                  />
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      {/* The axis, aligned to the bar column. */}
      <div className="grid gap-x-3 sm:grid-cols-[minmax(6rem,11rem)_1fr] mt-2">
        <span />
        {/* Labels are positioned from the same percentage as the gridlines, so
            the two cannot drift apart; the ends are nudged inward so neither
            overhangs the plot. */}
        <div className="relative h-4 border-t border-rule pt-1">
          {ticks.map((tick, index) => {
            const left = (index / (ticks.length - 1)) * 100;
            const nudge =
              index === 0 ? "translateX(0)" :
              index === ticks.length - 1 ? "translateX(-100%)" :
              "translateX(-50%)";
            return (
              <span
                key={tick}
                className="absolute top-1 font-sans text-caption text-ink-muted tabular whitespace-nowrap"
                style={{ left: `${left}%`, transform: nudge }}
              >
                {tickLabel(tick)}
              </span>
            );
          })}
        </div>
      </div>

      <figcaption className="mt-3 font-sans text-caption text-ink-muted">
        {def.label} on the held-out test split
        {def.higherIsBetter
          ? " — higher is better, so the longest bar leads."
          : " — lower is better, so the shortest bar leads and the ranking is best-first, not longest-first."}
      </figcaption>
    </figure>
  );
}

function Bar({
  widthPercent, tone, text, title,
}: { widthPercent: number; tone: string; text: string; title: string }) {
  // The label sits inside the bar when there is room and outside when there is
  // not, so it is never clipped and never floats away from its mark.
  const inside = widthPercent > 22;
  return (
    <div className="flex items-center gap-2" title={title}>
      <div className="h-5 flex-1 bg-paper-sunken">
        <div
          className={`h-5 ${tone} rounded-r-[4px] flex items-center justify-end pr-1.5 transition-[width] duration-200`}
          style={{ width: `${Math.max(widthPercent, 0.5)}%` }}
        >
          {inside && (
            <span className="font-mono text-caption text-white tabular">{text}</span>
          )}
        </div>
      </div>
      {!inside && (
        <span className="font-mono text-caption text-ink tabular w-14 shrink-0">{text}</span>
      )}
    </div>
  );
}
