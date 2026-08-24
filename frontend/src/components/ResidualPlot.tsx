/**
 * Actual vs predicted scatter for regression runs.
 *
 * The dashed diagonal is y = x — a perfect model puts every point on it — so
 * both axes must share one domain or the "perfect" line would not be at 45
 * degrees and the whole plot would lie. The SVG keeps a fixed size and its
 * container scrolls, which is what keeps dots at the 8px minimum on a narrow
 * screen instead of shrinking them away.
 */
import type { ResidualPoint } from "../interface/MlTypes";

const WIDTH = 660;
const HEIGHT = 440;
const MARGIN = { top: 16, right: 24, bottom: 52, left: 76 };
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;
const TICK_COUNT = 5;
/** More dots than this stops being readable and starts being a smear. */
const MAX_POINTS = 500;

function formatTick(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude < 0.01) return value.toExponential(1);
  if (magnitude >= 10000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  if (magnitude >= 100) return value.toFixed(0);
  return value.toFixed(2);
}

interface ResidualPlotProps {
  points: ResidualPoint[] | null;
  targetColumn: string;
}

export default function ResidualPlot({
  points,
  targetColumn,
}: ResidualPlotProps) {
  const usable = (points ?? []).filter(
    (point) =>
      Number.isFinite(point.actual) && Number.isFinite(point.predicted)
  );

  if (usable.length === 0) {
    return (
      <section>
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Actual vs predicted
        </h2>
        <p className="rounded-md border border-gray-300 bg-gray-100 p-4 text-sm text-ink-muted">
          This run did not return any actual-vs-predicted samples, so there is
          nothing to plot.
        </p>
      </section>
    );
  }

  const shown = usable.slice(0, MAX_POINTS);

  const values = usable.flatMap((point) => [point.actual, point.predicted]);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const span = rawMax - rawMin;
  // A dataset where every value is identical would otherwise divide by zero.
  const pad = span === 0 ? Math.max(Math.abs(rawMax) * 0.1, 1) : span * 0.05;
  const domainMin = rawMin - pad;
  const domainMax = rawMax + pad;
  const domainSpan = domainMax - domainMin;

  const toX = (value: number) =>
    MARGIN.left + ((value - domainMin) / domainSpan) * PLOT_WIDTH;
  const toY = (value: number) =>
    MARGIN.top + PLOT_HEIGHT - ((value - domainMin) / domainSpan) * PLOT_HEIGHT;

  const ticks = Array.from({ length: TICK_COUNT }, (_unused, index) => {
    const value = domainMin + (domainSpan * index) / (TICK_COUNT - 1);
    return { value, x: toX(value), y: toY(value) };
  });

  return (
    <section>
      <h2 className="text-xl font-semibold text-gray-800 mb-2">
        Actual vs predicted
      </h2>
      <p className="mb-4 text-sm text-ink-muted">
        Every dot is one held-out row: where <code>{targetColumn}</code> really
        landed (across) against what the model predicted (up).
      </p>

      <div className="overflow-x-auto">
        <svg
          role="img"
          width={WIDTH}
          height={HEIGHT}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        >
          <title>
            {`Scatter plot of actual versus predicted ${targetColumn} for ${shown.length} held-out rows, with a dashed y = x reference line.`}
          </title>

          {ticks.map((tick) => (
            <g key={`grid-${tick.value}`}>
              <line
                x1={tick.x}
                y1={MARGIN.top}
                x2={tick.x}
                y2={MARGIN.top + PLOT_HEIGHT}
                className="stroke-gray-200"
                strokeWidth={1}
              />
              <line
                x1={MARGIN.left}
                y1={tick.y}
                x2={MARGIN.left + PLOT_WIDTH}
                y2={tick.y}
                className="stroke-gray-200"
                strokeWidth={1}
              />
            </g>
          ))}

          {/* y = x: a perfect prediction sits exactly on this rule. */}
          <line
            x1={toX(domainMin)}
            y1={toY(domainMin)}
            x2={toX(domainMax)}
            y2={toY(domainMax)}
            className="stroke-gray-400"
            strokeWidth={1}
            strokeDasharray="4 4"
          />

          <line
            x1={MARGIN.left}
            y1={MARGIN.top + PLOT_HEIGHT}
            x2={MARGIN.left + PLOT_WIDTH}
            y2={MARGIN.top + PLOT_HEIGHT}
            className="stroke-gray-300"
            strokeWidth={1}
          />
          <line
            x1={MARGIN.left}
            y1={MARGIN.top}
            x2={MARGIN.left}
            y2={MARGIN.top + PLOT_HEIGHT}
            className="stroke-gray-300"
            strokeWidth={1}
          />

          {ticks.map((tick) => (
            <text
              key={`x-tick-${tick.value}`}
              x={tick.x}
              y={MARGIN.top + PLOT_HEIGHT + 18}
              textAnchor="middle"
              className="fill-ink-muted text-[11px]"
            >
              {formatTick(tick.value)}
            </text>
          ))}
          {ticks.map((tick) => (
            <text
              key={`y-tick-${tick.value}`}
              x={MARGIN.left - 10}
              y={tick.y + 4}
              textAnchor="end"
              className="fill-ink-muted text-[11px]"
            >
              {formatTick(tick.value)}
            </text>
          ))}

          <text
            x={MARGIN.left + PLOT_WIDTH / 2}
            y={HEIGHT - 12}
            textAnchor="middle"
            className="fill-ink-muted text-[12px]"
          >
            Actual {targetColumn}
          </text>
          <text
            x={-(MARGIN.top + PLOT_HEIGHT / 2)}
            y={16}
            transform="rotate(-90)"
            textAnchor="middle"
            className="fill-ink-muted text-[12px]"
          >
            Predicted {targetColumn}
          </text>

          {shown.map((point, index) => (
            <circle
              key={`${point.actual}-${point.predicted}-${index}`}
              cx={toX(point.actual)}
              cy={toY(point.predicted)}
              r={5}
              className="fill-series-2 stroke-white hover:fill-series-1"
              strokeWidth={2}
            >
              <title>
                {`Actual ${formatTick(point.actual)}, predicted ${formatTick(
                  point.predicted
                )} (off by ${formatTick(point.predicted - point.actual)})`}
              </title>
            </circle>
          ))}
        </svg>
      </div>

      <p className="mt-2 text-xs text-ink-muted">
        Points sitting on the dashed line are accurate predictions. A point
        above the line means the model guessed too high for that row, below
        means too low, and a cloud that drifts off the line at one end means the
        model is biased in that part of the range.
        {usable.length > shown.length &&
          ` Showing the first ${shown.length} of ${usable.length} sampled rows.`}
      </p>
    </section>
  );
}
