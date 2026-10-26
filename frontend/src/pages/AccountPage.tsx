/**
 * Sign in and sign up.
 *
 * An account is optional for the workbench and the course text. It adds two
 * things: course progress saved with the account rather than in one browser,
 * and -- where the deployment runs JupyterHub -- a personal notebook server
 * whose files persist between visits.
 *
 * `?next=` carries where to go afterwards. It is often not a page of this app
 * at all but the OAuth authorize endpoint, which sent a signed-out student
 * here on their way to their notebook; that is why success is a full page
 * load, not a client-side navigation.
 */
import React, { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { safeNextPath, useAuth } from "../context/authContext";
import type { ApiError } from "../interface/MlTypes";

type Mode = "login" | "register";

function fieldError(error: ApiError | null, field: string): string | undefined {
  return error?.errors?.[field]?.[0];
}

export default function AccountPage({ mode }: { mode: Mode }) {
  const { session, loading, signIn, signUp } = useAuth();
  const [search] = useSearchParams();
  const next = safeNextPath(search.get("next"));
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const isLogin = mode === "login";
  const otherQuery = next ? `?next=${encodeURIComponent(next)}` : "";

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true });
  }, [mode]);

  const finish = () => {
    window.location.assign(next ?? "/app/courses");
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (isLogin) await signIn({ username, password });
      else await signUp({ username, email, password });
      finish();
    } catch (caught) {
      setError(caught as ApiError);
      setBusy(false);
    }
  };

  if (!loading && session?.authenticated && session.user) {
    return (
      <div className="p-5 sm:p-8 max-w-measure">
        <h1 ref={headingRef} tabIndex={-1} className="text-h1 mb-4 focus:outline-none">You are signed in</h1>
        <p className="mb-6">Signed in as <strong>{session.user.username}</strong>.</p>
        <button type="button" className="btn btn-primary" onClick={finish}>{next ? "Continue" : "Go to courses"}</button>
      </div>
    );
  }

  const registrationClosed = !isLogin && session !== null && !session.registration_open;

  return (
    <div className="p-5 sm:p-8">
      <div className="max-w-md">
        <p className="field-label mb-3">Furina ML · Account</p>
        <h1 ref={headingRef} tabIndex={-1} className="text-h1 mb-4 focus:outline-none">
          {isLogin ? "Sign in" : "Create an account"}
        </h1>
        <p className="mb-6 text-small text-ink-muted">
          An account saves your course progress and, where this site runs a notebook server, gives you your own
          Jupyter workspace whose files are kept between visits. The workbench and lessons work without one.
        </p>
        {next?.startsWith("/api/oauth/") && (
          <p className="callout mb-6 text-small" role="status">Sign in to open your notebook. You will go straight to it afterwards.</p>
        )}
        {registrationClosed ? (
          <p className="callout">New accounts are created by an administrator on this site. <Link to={`/login${otherQuery}`}>Sign in</Link> if you have one.</p>
        ) : (
          <form onSubmit={(event) => void handleSubmit(event)} noValidate className="space-y-5">
            <label className="block">
              <span className="field-label block mb-2">Username</span>
              <input
                value={username} onChange={(event) => setUsername(event.target.value)} required
                autoComplete="username" autoCapitalize="none" spellCheck={false}
                aria-invalid={Boolean(fieldError(error, "username"))}
                aria-describedby={fieldError(error, "username") ? "username-error" : undefined}
                className="w-full min-h-11 border border-rule-strong bg-paper-raised px-3 rounded-control"
              />
              {fieldError(error, "username") && <span id="username-error" className="block text-small text-state-bad mt-1">{fieldError(error, "username")}</span>}
            </label>
            {!isLogin && (
              <label className="block">
                <span className="field-label block mb-2">Email (optional)</span>
                <input
                  type="email" value={email} onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  aria-invalid={Boolean(fieldError(error, "email"))}
                  aria-describedby={fieldError(error, "email") ? "email-error" : undefined}
                  className="w-full min-h-11 border border-rule-strong bg-paper-raised px-3 rounded-control"
                />
                {fieldError(error, "email") && <span id="email-error" className="block text-small text-state-bad mt-1">{fieldError(error, "email")}</span>}
              </label>
            )}
            <label className="block">
              <span className="field-label block mb-2">Password</span>
              <input
                type="password" value={password} onChange={(event) => setPassword(event.target.value)} required
                autoComplete={isLogin ? "current-password" : "new-password"}
                aria-invalid={Boolean(fieldError(error, "password"))}
                aria-describedby={fieldError(error, "password") ? "password-error" : !isLogin ? "password-help" : undefined}
                className="w-full min-h-11 border border-rule-strong bg-paper-raised px-3 rounded-control"
              />
              {!isLogin && !fieldError(error, "password") && <span id="password-help" className="block text-small text-ink-muted mt-1">At least 8 characters, not only numbers, and not a common password.</span>}
              {fieldError(error, "password") && (
                <span id="password-error" className="block text-small text-state-bad mt-1">{error?.errors?.password?.join(" ")}</span>
              )}
            </label>
            {error && !error.errors && <p role="alert" className="text-small text-state-bad">{error.detail}</p>}
            <button type="submit" className="btn btn-primary w-full" disabled={busy || !username || !password}>
              {busy ? (isLogin ? "Signing in…" : "Creating account…") : isLogin ? "Sign in" : "Create account"}
            </button>
          </form>
        )}
        <p className="mt-6 text-small">
          {isLogin
            ? <>New here? <Link to={`/register${otherQuery}`}>Create an account</Link></>
            : <>Already have an account? <Link to={`/login${otherQuery}`}>Sign in</Link></>}
        </p>
      </div>
    </div>
  );
}
