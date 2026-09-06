import { useMemo, useState } from "react";

import type { ColumnProfile } from "../interface/MlTypes";
import { WARNING_MARK } from "../explainer/marks";
import type {
  DatasetProfileProps,
  ProfileSortKey,
  SortDirection,
} from "../interface/PreprocessingTypes";

/** Sequential ramp step for a 0..100 magnitude. Never a rainbow. */
function missingRampClass(pct: number): string {
  if (pct <= 0) return "bg-rule";
  if (pct < 5) return "bg-ramp-1";
  if (pct < 15) return "bg-ramp-2";
  if (pct < 30) return "bg-ramp-3";
  if (pct < 50) return "bg-ramp-4";
  if (pct < 75) return "bg-ramp-5";
  return "bg-ramp-6";
}

function formatNumber(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e6 || magnitude < 1e-3)) {
    return value.toExponential(2);
  }
  return Number.isInteger(value) ? value.toString() : value.toFixed(3);
}

const KIND_ORDER: Record<string, number> = {
  numeric: 0,
  boolean: 1,
  categorical: 2,
  datetime: 3,
  text: 4,
};

interface SortHeaderProps {
  label: string;
  sortKey: ProfileSortKey;
  activeKey: ProfileSortKey;
  direction: SortDirection;
  onSort: (key: ProfileSortKey) => void;
  align?: "left" | "right";
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  direction,
  onSort,
  align = "left",
}: SortHeaderProps) {
  const active = activeKey === sortKey;
  return (
    <th
      scope="col"
      aria-sort={
        active ? (direction === "asc" ? "ascending" : "descending") : "none"
      }
      className={`px-3 py-2 ${align === "right" ? "text-right" : "text-left"}`}
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        title={`Sort by ${label.toLowerCase()}`}
        className={`inline-flex items-center gap-1 font-semibold hover:text-secondary ${
          active ? "text-secondary" : "text-ink-muted"
        }`}
      >
        {label}
        <span aria-hidden="true">
          {active ? (direction === "asc" ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </th>
  );
}

export default function DatasetProfile({ columns }: DatasetProfileProps) {
  const [sortKey, setSortKey] = useState<ProfileSortKey>("name");
  const [direction, setDirection] = useState<SortDirection>("asc");

  const handleSort = (key: ProfileSortKey) => {
    if (key === sortKey) {
      setDirection(direction === "asc" ? "desc" : "asc");
      return;
    }
    setSortKey(key);
    // Magnitude columns are far more useful worst-first.
    setDirection(key === "name" || key === "kind" ? "asc" : "desc");
  };

  const sorted = useMemo(() => {
    const factor = direction === "asc" ? 1 : -1;
    const compare = (a: ColumnProfile, b: ColumnProfile): number => {
      switch (sortKey) {
        case "missing":
          return (a.missing_pct - b.missing_pct) * factor;
        case "unique":
          return (a.unique - b.unique) * factor;
        case "kind": {
          const rank = (KIND_ORDER[a.kind] ?? 99) - (KIND_ORDER[b.kind] ?? 99);
          return (rank !== 0 ? rank : a.name.localeCompare(b.name)) * factor;
        }
        default:
          return a.name.localeCompare(b.name) * factor;
      }
    };
    return [...columns].sort(compare);
  }, [columns, sortKey, direction]);

  const withMissing = columns.filter((column) => column.missing > 0).length;
  const targetable = columns.filter((column) => column.suitable_as_target).length;

  if (columns.length === 0) {
    return (
      <div className="py-8 text-center">
        <p className="font-semibold text-ink">No column profile yet</p>
        <p className="text-sm text-ink-muted mt-1">
          The profile appears as soon as a dataset is loaded.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4">
        <h2 className="font-serif text-h3 text-ink">Column profile</h2>
        <p className="text-sm text-ink-muted">
          {columns.length} columns · {withMissing} with missing values ·{" "}
          {targetable} usable as a prediction target. Click a heading to sort.
        </p>
      </div>

      <div className="overflow-x-auto rounded-control border border-rule">
        <table className="min-w-full text-sm">
          <caption className="sr-only">
            Per-column profile of the active dataset.
          </caption>
          <thead className="bg-paper border-b border-rule text-xs uppercase tracking-wide">
            <tr>
              <SortHeader
                label="Column"
                sortKey="name"
                activeKey={sortKey}
                direction={direction}
                onSort={handleSort}
              />
              <SortHeader
                label="Kind"
                sortKey="kind"
                activeKey={sortKey}
                direction={direction}
                onSort={handleSort}
              />
              <th scope="col" className="px-3 py-2 text-left font-semibold text-ink-muted">
                Stored type
              </th>
              <SortHeader
                label="Missing"
                sortKey="missing"
                activeKey={sortKey}
                direction={direction}
                onSort={handleSort}
              />
              <SortHeader
                label="Unique"
                sortKey="unique"
                activeKey={sortKey}
                direction={direction}
                onSort={handleSort}
                align="right"
              />
              <th scope="col" className="px-3 py-2 text-left font-semibold text-ink-muted">
                Values
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((column) => {
              const top = column.top_values ?? [];
              const samples = column.sample_values;
              return (
                <tr
                  key={column.name}
                  className="border-b border-rule last:border-b-0 hover:bg-paper-sunken align-top"
                >
                  <td className="px-3 py-3">
                    <div className="font-semibold text-ink break-words">
                      {column.name}
                    </div>
                    {!column.suitable_as_target && (
                      <div
                        className="mt-1 inline-flex items-center gap-1 text-xs font-semibold text-state-warn"
                        title="Every row is the same value, every row is unique, or the column is entirely empty — nothing to learn from."
                      >
                        <span aria-hidden="true">{WARNING_MARK}</span>
                        Not usable as a target
                      </div>
                    )}
                    {column.suitable_as_target && column.suggested_task && (
                      <div className="mt-1 text-xs text-ink-muted">
                        Target would be {column.suggested_task}
                      </div>
                    )}
                  </td>

                  <td className="px-3 py-3">
                    <span className="inline-block rounded-control px-2 py-0.5 text-xs font-semibold bg-paper-sunken text-ink">
                      {column.kind}
                    </span>
                  </td>

                  <td className="px-3 py-3 text-ink-muted font-mono text-xs">
                    {column.dtype}
                  </td>

                  <td className="px-3 py-3 min-w-[10rem]">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-2 w-24 rounded-full bg-paper-sunken overflow-hidden flex-shrink-0"
                        title={`${column.missing} missing (${column.missing_pct}%)`}
                      >
                        <div
                          className={`h-2 rounded-full ${missingRampClass(column.missing_pct)}`}
                          style={{
                            width: `${Math.min(100, Math.max(column.missing_pct, column.missing > 0 ? 3 : 0))}%`,
                          }}
                        />
                      </div>
                      <span className="text-ink whitespace-nowrap">
                        {column.missing.toLocaleString()}
                        <span className="text-ink-muted"> ({column.missing_pct}%)</span>
                      </span>
                    </div>
                  </td>

                  <td className="px-3 py-3 text-right text-ink">
                    {column.unique.toLocaleString()}
                  </td>

                  <td className="px-3 py-3">
                    {column.kind === "numeric" ? (
                      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink">
                        <span>
                          <span className="text-ink-muted">min </span>
                          {formatNumber(column.min)}
                        </span>
                        <span>
                          <span className="text-ink-muted">mean </span>
                          {formatNumber(column.mean)}
                        </span>
                        <span>
                          <span className="text-ink-muted">max </span>
                          {formatNumber(column.max)}
                        </span>
                        <span>
                          <span className="text-ink-muted">sd </span>
                          {formatNumber(column.std)}
                        </span>
                      </div>
                    ) : top.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {top.map((entry) => (
                          <span
                            key={`${column.name}-${entry.value}`}
                            title={`${entry.value} — ${entry.count.toLocaleString()} rows`}
                            className="inline-block max-w-[14rem] truncate rounded-control bg-paper border border-rule px-2 py-0.5 text-xs text-ink"
                          >
                            {entry.value === "" ? "(empty)" : entry.value}
                            <span className="text-ink-muted"> ×{entry.count}</span>
                          </span>
                        ))}
                      </div>
                    ) : samples.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {samples.map((sample, index) => (
                          <span
                            key={`${column.name}-sample-${index}`}
                            className="inline-block max-w-[14rem] truncate rounded-control bg-paper border border-rule px-2 py-0.5 text-xs text-ink"
                          >
                            {sample === "" ? "(empty)" : sample}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-ink-muted">No values</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
