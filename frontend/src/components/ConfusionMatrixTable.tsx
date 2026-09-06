/**
 * The confusion matrix as a labelled NxN grid heatmap.
 *
 * It has to work for two classes and for ten, so nothing here is hardcoded to
 * a 2x2 "true positive / false negative" layout. Cell shade is scaled to the
 * row total rather than to the whole matrix, so a class with only a handful of
 * test rows still reads instead of washing out next to a dominant class.
 */
import type { ConfusionMatrixData, PerClassMetric } from "../interface/MlTypes";

/**
 * Full class names, because Tailwind scans source text: an interpolated
 * `bg-ramp-${i}` would be purged from the stylesheet.
 */
const RAMP_BG = [
  "bg-ramp-0",
  "bg-ramp-1",
  "bg-ramp-2",
  "bg-ramp-3",
  "bg-ramp-4",
  "bg-ramp-5",
  "bg-ramp-6",
];

/** Verified contrast: steps 0-4 carry ink, steps 5-6 carry white. */
function rampTextClass(step: number): string {
  return step >= 5 ? "text-white" : "text-ink";
}

/** 0 stays at the lightest step; anything above 0 gets at least step 1. */
function rampStep(count: number, rowTotal: number): number {
  if (count <= 0 || rowTotal <= 0) return 0;
  const share = count / rowTotal;
  return Math.min(6, Math.max(1, Math.round(share * 6)));
}

function formatShare(count: number, rowTotal: number): string {
  if (rowTotal <= 0) return "—";
  return `${((count / rowTotal) * 100).toFixed(1)}%`;
}

interface ConfusionMatrixTableProps {
  data: ConfusionMatrixData;
  perClass: PerClassMetric[] | null;
}

export default function ConfusionMatrixTable({
  data,
  perClass,
}: ConfusionMatrixTableProps) {
  const { labels, matrix } = data;

  if (labels.length === 0 || matrix.length === 0) {
    return (
      <section>
        <h2 className="font-serif text-h3 text-ink mb-4">
          Confusion matrix
        </h2>
        <p className="text-small text-ink-muted max-w-measure">
          The backend returned an empty confusion matrix for this run, so there
          is nothing to plot.
        </p>
      </section>
    );
  }

  const rowTotals = matrix.map((row) =>
    row.reduce((sum, cell) => sum + cell, 0)
  );
  const grandTotal = rowTotals.reduce((sum, total) => sum + total, 0);
  const correct = labels.reduce((sum, _label, index) => {
    const row = matrix[index];
    const cell = row === undefined ? undefined : row[index];
    return sum + (cell ?? 0);
  }, 0);

  return (
    <section>
      <h2 className="font-serif text-h3 text-ink mb-2">
        Confusion matrix
      </h2>
      <p className="mb-4 text-sm text-ink-muted">
        Each row is what the rows in the test set actually were; each column is
        what the model said. Outlined cells on the diagonal are the correct
        predictions — everything off the diagonal is a mistake, and the column
        it lands in tells you which class the model confused it with.{" "}
        {grandTotal > 0 && (
          <span className="text-ink">
            {correct} of {grandTotal} test rows landed on the diagonal.
          </span>
        )}
      </p>

      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-[2px] text-sm">
          <caption className="caption-bottom pt-3 text-left text-xs text-ink-muted">
            Shade shows each cell's share of its own row, so a small class is
            still readable. Counts are printed in every cell.
          </caption>
          <thead>
            <tr>
              <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-ink-muted">
                Actual ↓ / Predicted →
              </th>
              {labels.map((label) => (
                <th
                  key={label}
                  scope="col"
                  className="max-w-[10rem] truncate px-3 py-2 text-center text-xs font-semibold text-ink"
                  title={label}
                >
                  {label}
                </th>
              ))}
              <th
                scope="col"
                className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wide text-ink-muted"
              >
                Row total
              </th>
            </tr>
          </thead>
          <tbody>
            {labels.map((actualLabel, rowIndex) => {
              const row = matrix[rowIndex] ?? [];
              const rowTotal = rowTotals[rowIndex] ?? 0;
              return (
                <tr key={actualLabel}>
                  <th
                    scope="row"
                    className="max-w-[12rem] truncate px-3 py-2 text-left text-xs font-semibold text-ink"
                    title={actualLabel}
                  >
                    {actualLabel}
                  </th>
                  {labels.map((predictedLabel, colIndex) => {
                    const count = row[colIndex] ?? 0;
                    const step = rampStep(count, rowTotal);
                    const isDiagonal = rowIndex === colIndex;
                    return (
                      <td
                        key={predictedLabel}
                        title={`Actual ${actualLabel}, predicted ${predictedLabel}: ${count} samples`}
                        className={`min-w-[5.5rem] px-3 py-2 text-center transition-opacity hover:opacity-80 ${
                          RAMP_BG[step] ?? "bg-ramp-0"
                        } ${rampTextClass(step)} ${
                          isDiagonal
                            ? "rounded-control ring-2 ring-inset ring-ink font-bold"
                            : ""
                        }`}
                      >
                        <span className="block font-mono text-base leading-tight">
                          {count}
                        </span>
                        <span className="block font-mono text-[11px] opacity-80">
                          {formatShare(count, rowTotal)}
                        </span>
                      </td>
                    );
                  })}
                  <td className="px-3 py-2 text-right font-mono text-xs text-ink-muted">
                    {rowTotal}
                    {rowTotal === 0 && (
                      <span className="block text-[11px]">no test rows</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3 className="mt-8 mb-3 text-lg font-semibold text-ink">
        Per-class scores
      </h3>
      {perClass && perClass.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-rule text-left text-ink-muted">
                <th className="py-2 pr-4 font-semibold">Class</th>
                <th className="py-2 pr-4 font-semibold">Precision</th>
                <th className="py-2 pr-4 font-semibold">Recall</th>
                <th className="py-2 pr-4 font-semibold">F1</th>
                <th className="py-2 font-semibold">Support</th>
              </tr>
            </thead>
            <tbody>
              {perClass.map((entry) => (
                <tr
                  key={entry.label}
                  className="border-b border-rule hover:bg-paper-sunken"
                >
                  <td
                    className="max-w-[14rem] truncate py-2 pr-4 text-ink"
                    title={entry.label}
                  >
                    {entry.label}
                  </td>
                  <td className="py-2 pr-4 font-mono text-ink">
                    {entry.precision.toFixed(3)}
                  </td>
                  <td className="py-2 pr-4 font-mono text-ink">
                    {entry.recall.toFixed(3)}
                  </td>
                  <td className="py-2 pr-4 font-mono text-ink">
                    {entry.f1.toFixed(3)}
                  </td>
                  <td className="py-2 font-mono text-ink-muted">
                    {entry.support}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-ink-muted">
            Support is how many test rows really belonged to that class. A class
            with tiny support has scores you should not trust far.
          </p>
        </div>
      ) : (
        <p className="text-small text-ink-muted max-w-measure">
          This run did not report per-class precision and recall.
        </p>
      )}
    </section>
  );
}
