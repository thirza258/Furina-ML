/**
 * A lesson's Jupyter notebook, opened in the page the way Colab opens one.
 *
 * The learner picks where the code runs, once:
 * - **This browser** -- JupyterLite. Python (Pyodide) runs on the learner's
 *   own device; no account needed; edits stay in this browser.
 * - **My server** -- signed in, on a site that runs JupyterHub. The learner's
 *   own copy on their isolated server, saved between visits.
 * The choice is remembered, so every later lesson opens its notebook straight
 * away. "Open in Colab" and a download are always offered beside it. When the
 * site runs the grader, the finished notebook can be handed in below it.
 */
import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";

import { useAuth } from "../context/authContext";
import { tierLabel } from "../data/courseCatalogue";
import type { Course, CourseLesson } from "../data/courseCatalogue";
import type { NotebookLaunch } from "../interface/AccountTypes";
import type { ApiError } from "../interface/MlTypes";
import services, { NOTEBOOKS_GITHUB_PATH } from "../services/services";
import NotebookSubmission from "./NotebookSubmission";

type Runtime = "browser" | "server";
const RUNTIME_KEY = "furina-ml.notebook-runtime";

function readRuntime(): Runtime | null {
  try {
    const stored = window.localStorage.getItem(RUNTIME_KEY);
    return stored === "browser" || stored === "server" ? stored : null;
  } catch {
    return null;
  }
}

function rememberRuntime(runtime: Runtime | null): void {
  try {
    if (runtime) window.localStorage.setItem(RUNTIME_KEY, runtime);
    else window.localStorage.removeItem(RUNTIME_KEY);
  } catch {
    /* Remembering the choice is a convenience only. */
  }
}

export default function LessonNotebook({ course, lesson }: { course: Course; lesson: CourseLesson }) {
  const { session } = useAuth();
  const { pathname } = useLocation();
  const [chosen, setChosen] = useState<Runtime | null>(readRuntime);
  const [server, setServer] = useState<NotebookLaunch | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fullScreen, setFullScreen] = useState(false);
  const notebook = lesson.notebook;
  const path = notebook?.path ?? "";

  const hubEnabled = session?.notebooks.hub_enabled ?? false;
  // The workbench notebook loads data; it has no exercises to grade.
  const gradable = (session?.notebooks.grading_enabled ?? false) && !path.startsWith("workbench/");
  const signedIn = session?.authenticated ?? false;
  // A remembered "server" choice only applies while it can be honoured.
  const runtime: Runtime | null = chosen === "server" && !(hubEnabled && signedIn) ? null : chosen;

  useEffect(() => {
    if (runtime !== "server" || !path) return;
    let cancelled = false;
    services.launchNotebook(path).then(
      (launch) => { if (!cancelled) setServer(launch); },
      (caught: unknown) => { if (!cancelled) setError((caught as ApiError).detail); }
    );
    return () => {
      cancelled = true;
    };
  }, [runtime, path]);

  useEffect(() => {
    if (!fullScreen) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setFullScreen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullScreen]);

  if (!notebook) return null;

  const browserUrl = `/notebooks/notebooks/index.html?path=${encodeURIComponent(path)}`;
  const downloadUrl = `/notebooks/files/${path}`;
  const colabUrl = NOTEBOOKS_GITHUB_PATH
    ? `https://colab.research.google.com/github/${NOTEBOOKS_GITHUB_PATH}/${path}`
    : null;
  const frameUrl = runtime === "browser" ? browserUrl : runtime === "server" ? server?.url ?? null : null;

  const choose = (next: Runtime | null) => {
    setError(null);
    setServer(null);
    setChosen(next);
    rememberRuntime(next);
  };

  const actions = (
    <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-small">
      {frameUrl && (
        <button type="button" className="text-primary underline cursor-pointer" onClick={() => setFullScreen((value) => !value)}>
          {fullScreen ? "Exit full screen" : "Full screen"}
        </button>
      )}
      {frameUrl && <a href={frameUrl} target="_blank" rel="noopener">New tab<span className="sr-only"> (opens in a new tab)</span></a>}
      {runtime === "server" && server && <a href={server.lab_url} target="_blank" rel="noopener">JupyterLab<span className="sr-only"> (opens in a new tab)</span></a>}
      {colabUrl && <a href={colabUrl} target="_blank" rel="noopener">Open in Colab<span className="sr-only"> (opens in a new tab)</span></a>}
      <a href={downloadUrl} download>Download .ipynb</a>
      {runtime && <button type="button" className="text-primary underline cursor-pointer" onClick={() => choose(null)}>Change runtime</button>}
    </span>
  );

  return (
    <section className="my-8 border border-rule-strong bg-paper-raised" aria-labelledby="lesson-notebook-title">
      <header className="flex flex-wrap items-baseline justify-between gap-3 px-4 py-3 border-b border-rule">
        <div className="min-w-0">
          <p className="field-label">Notebook · {tierLabel(course.dependencyTier)}</p>
          <h2 id="lesson-notebook-title" className="text-h3 truncate">📓 {notebook.title}</h2>
        </div>
        {actions}
      </header>

      {!runtime && (
        <div className="p-4 sm:p-6">
          <p className="mb-4 max-w-measure">
            Build it yourself: read a cell, run it, then type the next piece where the notebook says <strong>✍️ Your turn</strong>.
            Each check tells you when your code matches, and from then on the notebook runs on <em>your</em> version. Where should the code run?
          </p>
          <div className="grid sm:grid-cols-2 gap-4">
            <button type="button" onClick={() => choose("browser")} className="text-left border border-rule-strong bg-paper p-4 cursor-pointer hover:border-primary">
              <span className="block font-semibold mb-1">▶ Run in this browser</span>
              <span className="block text-small text-ink-muted">No account or installation. Python runs on your device; the first open downloads it once. Your edits stay in this browser.</span>
            </button>
            {hubEnabled && signedIn ? (
              <button type="button" onClick={() => choose("server")} className="text-left border border-rule-strong bg-paper p-4 cursor-pointer hover:border-primary">
                <span className="block font-semibold mb-1">▶ Open my saved copy</span>
                <span className="block text-small text-ink-muted">Runs on your own notebook server. Your work is saved to your account and is there next time.</span>
              </button>
            ) : hubEnabled ? (
              <Link to={`/login?next=${encodeURIComponent(pathname)}`} className="block border border-rule-strong bg-paper p-4 no-underline text-ink hover:border-primary">
                <span className="block font-semibold mb-1">Sign in to save your work</span>
                <span className="block text-small text-ink-muted">With an account, the notebook opens on your own server and keeps your edits between visits.</span>
              </Link>
            ) : (
              <div className="border border-rule bg-paper p-4 text-small text-ink-muted">
                Prefer your own setup? Download the notebook and open it in Jupyter{colabUrl ? ", or open it in Colab" : ""}. Every notebook is self-contained.
              </div>
            )}
          </div>
        </div>
      )}

      {runtime === "server" && !server && !error && (
        <p className="px-4 py-6 text-small" role="status">Starting your notebook server… the first start takes a few seconds.</p>
      )}
      {error && <p role="alert" className="px-4 py-4 text-small text-state-bad">{error}</p>}

      {frameUrl && (
        <div className={fullScreen ? "fixed inset-0 z-50 bg-paper flex flex-col" : ""}>
          {fullScreen && (
            <div className="flex flex-wrap items-baseline justify-between gap-3 px-4 py-2 border-b border-rule bg-paper-raised">
              <span className="font-semibold truncate">📓 {notebook.title}</span>
              {actions}
            </div>
          )}
          {/* Sandboxed so a notebook can never navigate this page away. */}
          <iframe
            key={frameUrl}
            src={frameUrl}
            title={`Notebook: ${notebook.title}`}
            className={fullScreen ? "w-full flex-grow border-0" : "w-full h-[85vh] border-0 border-t border-rule"}
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads allow-modals"
            allow="clipboard-read; clipboard-write"
          />
        </div>
      )}

      {gradable && (
        <div className="px-4 py-4 border-t border-rule">
          <h3 className="font-semibold mb-2">Hand in for grading</h3>
          <NotebookSubmission path={path} />
        </div>
      )}

      <p className="px-4 py-3 border-t border-rule text-caption text-ink-muted">
        {runtime === "server"
          ? "Saved in your workspace. An idle server is stopped after a while: your files stay, but run the cells again from the top to rebuild variables. "
          : runtime === "browser"
            ? "Running in this browser: work is kept in this browser's storage. Download the notebook to keep a copy. "
            : ""}
        Only run notebooks you trust: their code runs with your access to this site.
      </p>
    </section>
  );
}
