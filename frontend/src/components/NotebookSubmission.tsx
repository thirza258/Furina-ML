/**
 * Hand a lesson notebook in for automated grading, and see how it did.
 *
 * The learner downloads their copy from wherever they worked on it (this
 * browser, their own server, Colab) and uploads the .ipynb here. The grader
 * runs the lesson's own checks against the exercise cells in a sandbox; this
 * panel polls until the result is in, then lists every check.
 */
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Link, useLocation } from "react-router-dom";

import { useAuth } from "../context/authContext";
import type { Submission, SubmissionCheck } from "../interface/AccountTypes";
import type { ApiError } from "../interface/MlTypes";
import services from "../services/services";

const POLL_MS = 3000;

const CHECK_MARK: Record<SubmissionCheck["status"], { icon: string; label: string; tone: string }> = {
  passed: { icon: "✅", label: "passed", tone: "text-state-good" },
  failed: { icon: "❌", label: "not yet", tone: "text-state-bad" },
  "not-run": { icon: "⏸", label: "did not run", tone: "text-state-warn" },
};

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function isPending(submission: Submission | null): boolean {
  return submission?.status === "queued" || submission?.status === "running";
}

function Result({ submission }: { submission: Submission }) {
  if (isPending(submission)) {
    return (
      <p role="status" className="text-small">
        {submission.status === "queued" ? "Waiting for the grader…" : "Grading: running your notebook from the top…"}
      </p>
    );
  }
  if (submission.status === "error") {
    return <p role="alert" className="text-small text-state-bad">Could not be graded: {submission.error}</p>;
  }
  const items = submission.result.items ?? [];
  const unanswered = submission.result.unanswered ?? 0;
  return (
    <div>
      <p className="text-h3 tabular" role="status">
        {submission.passed} of {submission.total} checks passed
        <span className="text-small text-ink-muted"> · {Math.round((submission.score ?? 0) * 100)}%</span>
      </p>
      {unanswered > 0 && (
        <p className="text-small text-ink-muted mt-1">
          {unanswered} exercise{unanswered === 1 ? " is" : "s are"} still as the lesson ships {unanswered === 1 ? "it" : "them"}: look for <strong>✍️ Your turn</strong>.
        </p>
      )}
      {submission.result.complete === false && (
        <p className="text-small text-state-warn mt-1">
          The notebook stopped before the end, so later checks did not run. Check for a cell that runs out of memory.
        </p>
      )}
      <ul className="mt-3 divide-y divide-rule border-y border-rule">
        {items.map((item, index) => {
          const mark = CHECK_MARK[item.status];
          return (
            <li key={`${item.name}-${index}`} className="py-2">
              <details>
                <summary className="cursor-pointer text-small">
                  <span aria-hidden="true">{mark.icon}</span> <span className="font-semibold">{item.name}</span>{" "}
                  <span className={mark.tone}>({mark.label})</span>
                </summary>
                {item.message && (
                  <pre className="mt-2 whitespace-pre-wrap break-words text-caption bg-paper-sunken p-2">{item.message}</pre>
                )}
              </details>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function NotebookSubmission({ path }: { path: string }) {
  const { session } = useAuth();
  const { pathname } = useLocation();
  const signedIn = session?.authenticated ?? false;
  const [history, setHistory] = useState<Submission[] | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!signedIn) return;
    let cancelled = false;
    services.submissions(path).then(
      (list) => { if (!cancelled) setHistory(list); },
      () => { if (!cancelled) setHistory([]); }
    );
    return () => {
      cancelled = true;
    };
  }, [signedIn, path]);

  const latest = history?.[0] ?? null;

  // Poll the newest submission until the grader has finished with it.
  useEffect(() => {
    if (!latest || !isPending(latest)) return;
    const timer = window.setTimeout(() => {
      services.submission(latest.id).then(
        (fresh) => setHistory((list) => list?.map((item) => (item.id === fresh.id ? fresh : item)) ?? list),
        (caught: unknown) => setError((caught as ApiError).detail)
      );
    }, POLL_MS);
    return () => window.clearTimeout(timer);
  }, [latest]);

  if (!signedIn) {
    return (
      <p className="text-small">
        <Link to={`/login?next=${encodeURIComponent(pathname)}`}>Sign in</Link> to hand this notebook in and have your exercises graded.
      </p>
    );
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) return;
    setSending(true);
    setError(null);
    try {
      const created = await services.submitNotebook(path, file);
      setHistory((list) => [created, ...(list ?? [])]);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (caught) {
      const apiError = caught as ApiError;
      setError(apiError.errors?.notebook?.[0] ?? apiError.errors?.path?.[0] ?? apiError.detail);
    } finally {
      setSending(false);
    }
  };

  const earlier = (history ?? []).slice(1, 6).filter((item) => item.status === "graded");

  return (
    <div>
      <p className="text-small max-w-measure mb-3">
        Finished the exercises? Download your copy of the notebook (in Jupyter: <strong>File → Download</strong>; in Colab: <strong>File → Download → .ipynb</strong>) and hand it in.
        The grader runs it from the top in a sandbox and re-runs the lesson's own checks on every <strong>✍️ Your turn</strong> cell: only those cells are graded.
      </p>
      <form onSubmit={submit} className="flex flex-wrap items-center gap-3">
        <label className="sr-only" htmlFor={`submission-${path}`}>Your notebook (.ipynb)</label>
        <input
          id={`submission-${path}`}
          ref={inputRef}
          type="file"
          accept=".ipynb,application/x-ipynb+json,application/json"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          disabled={sending || isPending(latest)}
          className="text-small"
        />
        <button type="submit" className="btn btn-primary" disabled={!file || sending || isPending(latest)}>
          {sending ? "Handing in…" : "Hand in for grading"}
        </button>
      </form>
      {error && <p role="alert" className="mt-2 text-small text-state-bad">{error}</p>}

      {latest && (
        <div className="mt-4" aria-live="polite">
          <p className="field-label">Latest submission · {when(latest.created_at)}</p>
          <Result submission={latest} />
        </div>
      )}
      {earlier.length > 0 && (
        <p className="mt-3 text-caption text-ink-muted">
          Earlier: {earlier.map((item) => `${item.passed}/${item.total} (${when(item.created_at)})`).join(" · ")}
        </p>
      )}
    </div>
  );
}
