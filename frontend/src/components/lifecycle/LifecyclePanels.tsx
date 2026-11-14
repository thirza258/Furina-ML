/**
 * Small shared pieces of the lifecycle UI: leakage findings, the fit
 * diagnosis, and job progress. Each answers "what did Furina do?" in the
 * user's own terms rather than just showing a number (PRD §42).
 */
import type { Diagnosis, Job, LeakageSummary, ResourceEstimate } from "../../interface/LifecycleTypes";

const SEVERITY_STYLE: Record<string, string> = {
  info: "border-rule bg-paper-sunken text-ink",
  warning: "border-state-warn bg-paper-raised text-ink",
  critical: "border-state-bad bg-paper-raised text-ink",
};

const SEVERITY_LABEL: Record<string, string> = {
  info: "Note",
  warning: "Warning",
  critical: "Critical",
};

/** Pre-training leakage and data-quality findings. */
export function LeakagePanel({ summary }: { summary: LeakageSummary | null }) {
  if (!summary) return null;
  if (summary.count === 0) {
    return (
      <p className="text-small text-ink-muted border-l-2 border-state-good pl-3">
        No leakage or data-quality problems found in these columns.
      </p>
    );
  }
  return (
    <div className="space-y-2">
      <p className="text-small text-ink-muted">
        {summary.count} finding{summary.count === 1 ? "" : "s"} — the highest severity is{" "}
        <strong>{SEVERITY_LABEL[summary.severity] ?? summary.severity}</strong>. Training is not
        blocked; review these first.
      </p>
      <ul className="space-y-2">
        {summary.findings.map((finding, index) => (
          <li
            key={`${finding.code}-${index}`}
            className={`border-l-2 pl-3 py-1.5 text-small ${SEVERITY_STYLE[finding.severity] ?? SEVERITY_STYLE.info}`}
          >
            <span className="font-semibold">{SEVERITY_LABEL[finding.severity] ?? finding.severity}</span>
            {" — "}
            {finding.message}
            {finding.columns.length > 0 && (
              <span className="text-ink-muted"> ({finding.columns.join(", ")})</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The train-versus-test fit diagnosis, with the model's own reasoning shown. */
export function DiagnosisPanel({ diagnosis }: { diagnosis: Diagnosis | null }) {
  if (!diagnosis) return null;
  const tone =
    diagnosis.verdict === "good_fit"
      ? "border-state-good"
      : diagnosis.verdict === "potential_leakage"
        ? "border-state-bad"
        : "border-state-warn";
  const heading: Record<string, string> = {
    good_fit: "Good fit",
    overfitting: "Possible overfitting",
    underfitting: "Underfitting",
    potential_leakage: "Potential data leakage",
    unknown: "Not enough data to diagnose",
  };
  return (
    <div className={`border-l-2 pl-3 ${tone}`}>
      <p className="font-semibold">
        {heading[diagnosis.verdict] ?? diagnosis.verdict}
        {diagnosis.confidence !== "low" && (
          <span className="text-ink-muted font-normal"> — {diagnosis.confidence} confidence</span>
        )}
      </p>
      {diagnosis.train_metric !== null && diagnosis.test_metric !== null && (
        <p className="text-small text-ink-muted mt-1">
          Train {diagnosis.metric_name ?? "metric"}: {diagnosis.train_metric.toFixed(3)} · Test:{" "}
          {diagnosis.test_metric.toFixed(3)}
          {diagnosis.gap !== null && ` · Gap: ${diagnosis.gap.toFixed(3)}`}
        </p>
      )}
      <ul className="mt-2 space-y-1 text-small">
        {diagnosis.reasons.map((reason, index) => (
          <li key={index}>{reason}</li>
        ))}
      </ul>
    </div>
  );
}

/** Job progress, with the cancel affordance and failure guidance. */
export function JobProgress({
  job,
  onCancel,
}: {
  job: Job | null;
  onCancel?: () => void;
}) {
  if (!job) return null;
  const active = job.status === "queued" || job.status === "running";
  return (
    <div className="border border-rule p-3 space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <p className="font-semibold">
          {job.kind === "automl"
            ? "AutoML"
            : job.kind === "hpo"
              ? "Hyperparameter search"
              : job.kind === "reproduce"
                ? "Reproducing run"
                : "Training"}{" "}
          <span className="text-ink-muted font-normal">— {job.status}</span>
        </p>
        {active && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="text-small border border-rule px-2 py-1 hover:bg-paper-sunken"
          >
            Cancel
          </button>
        )}
      </div>
      <div className="h-2 bg-paper-sunken" role="progressbar" aria-valuenow={Math.round(job.progress)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-2 bg-primary" style={{ width: `${Math.max(0, Math.min(100, job.progress))}%` }} />
      </div>
      <p className="text-small text-ink-muted">
        {Math.round(job.progress)}%
        {job.total_units > 0 && ` · ${job.completed_units}/${job.total_units} trials`}
        {job.cancel_requested && " · cancellation requested"}
      </p>
      {job.error && (
        <div className="border-l-2 border-state-bad pl-3 text-small">
          <p className="font-semibold">{job.error}</p>
          {job.suggested_action && <p className="text-ink-muted mt-1">{job.suggested_action}</p>}
        </div>
      )}
    </div>
  );
}

/** The pre-flight resource estimate, graded so a too-big job is visible early. */
export function ResourcePanel({ estimate }: { estimate: ResourceEstimate | null }) {
  if (!estimate) return null;
  const tone =
    estimate.risk === "high"
      ? "border-state-bad"
      : estimate.risk === "medium"
        ? "border-state-warn"
        : "border-state-good";
  return (
    <div className={`border-l-2 pl-3 text-small ${tone}`}>
      <p>
        Estimated memory <strong>{estimate.estimated_gb} GB</strong> of{" "}
        {estimate.available_gb} GB available — risk{" "}
        <strong>{estimate.risk}</strong>.
      </p>
      <p className="text-ink-muted">
        {estimate.rows.toLocaleString()} rows × {estimate.columns} columns.
      </p>
    </div>
  );
}
