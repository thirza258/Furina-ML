/**
 * The top bar. Besides the product mark it answers the three questions a user
 * of this app asks most often: is the backend up, what data am I working on,
 * and did my model train?
 *
 * The backend check matters for the dataset tools. "Nothing happens when I click" is a miserable way to
 * discover it is not running — so a failed health check is stated in words,
 * with the reason and the API base URL, in a bar nobody can miss.
 */
import React, { useCallback, useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";

import services, { API_BASE_URL } from "../services/services";
import type { ApiError, HealthResponse } from "../interface/MlTypes";
import { useAuth } from "../context/authContext";
import { useWorkspace } from "../context/workspaceContext";

interface NavBarProps {
  selectedTool: string;
}

type HealthState =
  | { status: "checking" }
  | { status: "online"; info: HealthResponse }
  | { status: "offline"; reason: string };

/**
 * Service methods already throw a normalized ApiError, so the value caught
 * here is that object — not an Error. Read `detail` off it directly; passing
 * it back through the normalizer would stringify it to "[object Object]".
 */
function errorDetail(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as ApiError).detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error
    ? error.message
    : "The backend could not be reached.";
}

/** Metrics are 0..1 ratios almost everywhere; 3dp is enough to compare runs. */
function formatMetric(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(3);
}

const NavBar: React.FC<NavBarProps> = ({ selectedTool }) => {
  const { dataset, run, restoring, restoreError } = useWorkspace();
  const { session, loading: accountLoading, signOut } = useAuth();
  const { pathname, search } = useLocation();
  const [signingOut, setSigningOut] = useState<boolean>(false);
  const onAccountPage = pathname === "/login" || pathname === "/register";
  const [health, setHealth] = useState<HealthState>({ status: "checking" });
  const [rechecking, setRechecking] = useState<boolean>(false);
  const isCourse = selectedTool === "Courses";

  const checkHealth = useCallback(async (): Promise<HealthState> => {
    try {
      const info = await services.health();
      return { status: "online", info };
    } catch (error) {
      return { status: "offline", reason: errorDetail(error) };
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const next = await checkHealth();
      if (!cancelled) setHealth(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [checkHealth]);

  const handleRecheck = async () => {
    setRechecking(true);
    setHealth({ status: "checking" });
    const next = await checkHealth();
    setHealth(next);
    setRechecking(false);
  };

  const datasetLabel = restoring
    ? "Restoring…"
    : dataset
      ? `${dataset.name} · ${dataset.n_rows.toLocaleString()} rows × ${dataset.n_cols} cols`
      : "No dataset";

  const runLabel = restoring
    ? "Restoring…"
    : run
      ? run.status === "failed"
        ? `${run.algorithm_label} · training failed`
        : `${run.algorithm_label} · ${run.primary_metric_name} ${formatMetric(run.primary_metric)}`
      : "No model trained";

  return (
    <header className="flex-shrink-0 z-10">
      {/* The masthead is paper, like everything under it. A dark bar would
          read as an application chrome; this reads as the head of a page. */}
      <nav className="w-full bg-paper-raised border-b border-rule px-5 py-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
        <div className="flex items-baseline gap-3 min-w-0">
          <Link
            to="/"
            className="font-serif text-h3 font-semibold text-ink no-underline hover:text-primary transition-colors"
          >
            Furina&nbsp;ML
          </Link>
          <span aria-hidden="true" className="text-rule-strong">
            /
          </span>
          <span className="field-label truncate">{selectedTool}</span>
        </div>

        {/* Status readouts. Label over value, separated by rules rather than
            floated in pills — a masthead states facts, it does not badge them. */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-small">
          <span
            className="flex items-baseline gap-2 tabular"
            title={
              dataset
                ? `Active dataset: ${dataset.name} (${dataset.source})`
                : "Upload a CSV on the Data Preprocessing page to get started."
            }
          >
            <span className="field-label">Dataset</span>
            <span className={dataset ? "text-ink" : "text-ink-muted"}>
              {datasetLabel}
            </span>
          </span>

          <span aria-hidden="true" className="h-4 w-px bg-rule" />

          <span
            className="flex items-baseline gap-2 tabular"
            title={
              run
                ? `Run "${run.name || run.id}" on target "${run.target_column}"`
                : "Train a model on the Data Training page."
            }
          >
            <span className="field-label">Model</span>
            <span
              className={
                !run
                  ? "text-ink-muted"
                  : run.status === "failed"
                    ? "text-state-bad font-medium"
                    : "text-ink"
              }
            >
              {runLabel}
            </span>
          </span>

          <span aria-hidden="true" className="h-4 w-px bg-rule" />

          {/* Backend connectivity. Never colour alone: each state carries a
              mark and a word. */}
          {isCourse && <span className="text-ink-muted">Self-paced lessons</span>}
          {!isCourse && health.status === "checking" && (
            <span className="flex items-baseline gap-2 text-ink-muted">
              <span aria-hidden="true">◌</span> Checking backend…
            </span>
          )}
          {!isCourse && health.status === "online" && (
            <span
              className="flex items-baseline gap-2 text-state-good"
              title={`scikit-learn ${health.info.sklearn} · pandas ${health.info.pandas} · numpy ${health.info.numpy} · ${API_BASE_URL}`}
            >
              <span aria-hidden="true">●</span> Connected
              <span className="text-ink-muted tabular">
                scikit-learn {health.info.sklearn}
              </span>
            </span>
          )}
          {!isCourse && health.status === "offline" && (
            <span className="flex items-baseline gap-2 text-state-bad font-medium">
              <span aria-hidden="true">▲</span> Disconnected
            </span>
          )}

          {/* The account. Optional everywhere: it saves course progress and
              unlocks the personal notebook server. */}
          {!accountLoading && session?.authenticated && session.user && (
            <>
              <span aria-hidden="true" className="h-4 w-px bg-rule" />
              <span className="flex items-baseline gap-2">
                <span className="field-label">Account</span>
                <span className="text-ink">{session.user.username}</span>
                <button
                  type="button"
                  disabled={signingOut}
                  onClick={() => {
                    setSigningOut(true);
                    void signOut().finally(() => setSigningOut(false));
                  }}
                  className="text-primary underline cursor-pointer disabled:opacity-60"
                >
                  {signingOut ? "Signing out…" : "Sign out"}
                </button>
              </span>
            </>
          )}
          {!accountLoading && session !== null && !session.authenticated && !onAccountPage && (
            <>
              <span aria-hidden="true" className="h-4 w-px bg-rule" />
              <Link to={`/login?next=${encodeURIComponent(pathname + search)}`}>Sign in</Link>
            </>
          )}
        </div>
      </nav>

      {/* Courses can be read and completed without the API. */}
      {health.status === "offline" && !isCourse && (
        <div
          role="alert"
          className="w-full bg-state-bad text-white px-5 py-3 flex flex-wrap items-center justify-between gap-3"
        >
          <div className="text-small max-w-measure">
            <p className="font-medium">
              ▲ Backend disconnected — {health.reason}
            </p>
            <p className="opacity-90 mt-0.5">
              Nothing can be uploaded, trained or evaluated until the API
              answers at{" "}
              <code className="bg-white/20 px-1 rounded-control">
                {API_BASE_URL}
              </code>
              . Check that the backend container is running, or start the
              development server with{" "}
              <code className="bg-white/20 px-1 rounded-control">
                python manage.py runserver
              </code>
              .
            </p>
          </div>
          <button
            type="button"
            onClick={() => void handleRecheck()}
            disabled={rechecking}
            className="bg-white text-state-bad py-2 px-4 rounded-control text-small font-medium cursor-pointer hover:bg-paper-sunken transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {rechecking ? "Checking…" : "Retry connection"}
          </button>
        </div>
      )}

      {restoreError !== null && (
        <div
          role="alert"
          className="w-full bg-state-warn text-white px-5 py-2 text-small"
        >
          ▲ Could not restore your previous workspace — {restoreError}
        </div>
      )}
    </header>
  );
};

export default NavBar;
