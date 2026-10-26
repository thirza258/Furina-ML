/**
 * Who is signed in, for the parts of the app that care: saved course
 * progress, the per-student notebook server and the account controls.
 *
 * The workbench itself stays open to anonymous visitors, so nothing waits on
 * this: `loading` is true only until the first session check answers, and a
 * failed check simply leaves everyone signed out.
 *
 * Split from AuthProvider.tsx so neither file mixes component and
 * non-component exports (React Fast Refresh).
 */
import { createContext, useContext } from "react";

import type { Credentials, Registration, SessionInfo } from "../interface/AccountTypes";

export interface AuthState {
  session: SessionInfo | null;
  loading: boolean;
  signIn: (credentials: Credentials) => Promise<SessionInfo>;
  signUp: (registration: Registration) => Promise<SessionInfo>;
  signOut: () => Promise<void>;
}

export const AuthContext = createContext<AuthState | null>(null);

export function useAuth(): AuthState {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used inside an <AuthProvider>.");
  }
  return value;
}

/**
 * Only same-site paths may follow sign-in. `next` arrives in the URL -- the
 * OAuth authorize endpoint sends signed-out students here with it -- so an
 * absolute or protocol-relative value would make this an open redirect.
 */
export function safeNextPath(next: string | null): string | null {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) {
    return null;
  }
  return next;
}
