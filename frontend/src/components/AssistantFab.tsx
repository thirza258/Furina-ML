/**
 * The floating helper button, and the panel it opens.
 *
 * The helper is told what the user is looking at — which page, which dataset,
 * which run — so it can answer "which column should I predict?" against the
 * real column list instead of in the abstract. What it is told is the *shape*
 * of the data: column names, kinds, missing counts, and the scores of a
 * finished run. Never a row of the data itself.
 *
 * That still leaves column names going to a third party, so the panel says so
 * in as many words before the first message. Being quiet about it would be the
 * wrong call for an app whose whole pitch is that it does not hide what it does
 * with your data.
 *
 * No API key is involved here. The browser talks to `/api/assistant/` and the
 * server holds the OpenRouter key — a key shipped in a Vite bundle is a
 * published key.
 */
import { useEffect, useRef, useState } from "react";

import services, { toApiError } from "../services/services";
import type { AssistantMessage } from "../interface/MlTypes";
import { useWorkspace } from "../context/workspaceContext";
import { WARNING_MARK } from "../explainer/marks";

interface AssistantFabProps {
  /** The page the user is on, passed through to the model as context. */
  tool: string;
}

const SUGGESTIONS = [
  "What should I do on this page?",
  "Which column should I predict?",
  "Is my model overfitting?",
  "Which algorithm should I try next?",
];

export default function AssistantFab({ tool }: AssistantFabProps) {
  const { dataset, run } = useWorkspace();

  const [open, setOpen] = useState(false);
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const logRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Ask once whether the server has a key at all, so the button can explain
  // itself rather than failing on the first question.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await services.assistantStatus();
        if (!cancelled) setEnabled(status.enabled);
      } catch {
        if (!cancelled) setEnabled(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [messages, busy]);

  // Escape closes the panel, as it does for every other overlay.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const send = (text: string) => {
    const question = text.trim();
    if (!question || busy) return;

    const next: AssistantMessage[] = [...messages, { role: "user", content: question }];
    setMessages(next);
    setDraft("");
    setBusy(true);
    setError(null);

    void (async () => {
      try {
        const answer = await services.askAssistant(next, {
          tool,
          dataset_id: dataset?.id ?? null,
          run_id: run?.id ?? null,
        });
        setMessages([...next, { role: "assistant", content: answer.reply }]);
      } catch (caught) {
        setError(toApiError(caught).detail);
      } finally {
        setBusy(false);
      }
    })();
  };

  return (
    <>
      {/* The button. Square, ruled, ink on paper — a floating circle in a
          brand colour would be the one piece of this app that looked like a
          different app. */}
      <button
        type="button"
        onClick={() => setOpen((previous) => !previous)}
        aria-expanded={open}
        aria-controls="furina-assistant-panel"
        className="fixed bottom-6 right-6 z-40 flex items-center gap-2 border border-primary
                   bg-primary px-4 py-3 text-white font-sans text-small font-medium
                   cursor-pointer transition-colors hover:bg-[#471724] rounded-control"
      >
        <span aria-hidden="true" className="font-serif text-base leading-none">
          {open ? "×" : "?"}
        </span>
        {open ? "Close" : "Ask for help"}
      </button>

      {open && (
        <div
          id="furina-assistant-panel"
          role="dialog"
          aria-label="Furina ML assistant"
          className="fixed bottom-24 right-6 z-40 w-[min(26rem,calc(100vw-3rem))]
                     max-h-[min(34rem,calc(100vh-9rem))] flex flex-col
                     border border-rule bg-paper-raised"
        >
          <header className="px-4 py-3 border-b border-rule">
            <h2 className="font-serif text-h3 text-ink">Helper</h2>
            <p className="font-sans text-caption text-ink-muted">
              Knows this page
              {dataset ? `, your dataset “${dataset.name}”` : ""}
              {run ? ", and your last run" : ""}.
            </p>
          </header>

          <div ref={logRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {enabled === false ? (
              <p className="font-sans text-small text-ink-muted">
                {WARNING_MARK} The helper is switched off on this server. Set{" "}
                <code className="font-mono text-caption">OPENROUTER_API_KEY</code> in
                the environment and restart the backend to turn it on.
              </p>
            ) : messages.length === 0 ? (
              <>
                <p className="font-serif text-body text-ink">
                  Ask about anything on screen — what a control does, which
                  column to predict, whether a score is any good.
                </p>
                <p className="font-sans text-caption text-ink-muted border-l-2 border-rule pl-3">
                  Your question is answered by a language model run by OpenRouter.
                  It is told your column <em>names</em>, their types and your
                  scores — never the rows themselves.
                </p>
                <ul className="space-y-1.5 pt-1">
                  {SUGGESTIONS.map((suggestion) => (
                    <li key={suggestion}>
                      <button
                        type="button"
                        onClick={() => send(suggestion)}
                        className="text-left font-sans text-small text-primary hover:underline cursor-pointer"
                      >
                        {suggestion}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={
                    message.role === "user"
                      ? "border-l-2 border-primary pl-3"
                      : "border-l-2 border-rule pl-3"
                  }
                >
                  <p className="field-label mb-1">
                    {message.role === "user" ? "You" : "Helper"}
                  </p>
                  <p className="font-serif text-body text-ink whitespace-pre-wrap">
                    {message.content}
                  </p>
                </div>
              ))
            )}

            {busy && (
              <p className="font-sans text-small text-ink-muted">Thinking…</p>
            )}
            {error !== null && (
              <p className="font-sans text-small text-state-bad">
                {WARNING_MARK} {error}
              </p>
            )}
          </div>

          {enabled !== false && (
            <form
              className="border-t border-rule p-3 flex items-end gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                send(draft);
              }}
            >
              <textarea
                ref={inputRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  // Enter sends; Shift+Enter is a newline, as in every chat box.
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    send(draft);
                  }
                }}
                rows={2}
                placeholder="Ask a question…"
                className="flex-1 resize-none border border-rule-strong rounded-control
                           bg-paper-raised px-2 py-1.5 font-sans text-small text-ink"
              />
              <button
                type="submit"
                disabled={busy || draft.trim().length === 0}
                className="btn btn-primary"
              >
                Send
              </button>
            </form>
          )}
        </div>
      )}
    </>
  );
}
