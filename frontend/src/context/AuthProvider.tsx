import React, { useCallback, useEffect, useMemo, useState } from "react";

import services from "../services/services";
import type { Credentials, Registration, SessionInfo } from "../interface/AccountTypes";
import { AuthContext } from "./authContext";
import type { AuthState } from "./authContext";

const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const current = await services.session();
        if (!cancelled) setSession(current);
      } catch {
        // No backend: everyone is anonymous, and the courses still work.
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(async (credentials: Credentials) => {
    const next = await services.login(credentials);
    setSession(next);
    return next;
  }, []);

  const signUp = useCallback(async (registration: Registration) => {
    const next = await services.register(registration);
    setSession(next);
    return next;
  }, []);

  const signOut = useCallback(async () => {
    const hubUrl = session?.notebooks.hub_enabled ? session.notebooks.hub_url : null;
    setSession(await services.logout());
    // Then end the notebook server's session too: the hub deletes the tokens
    // of this browser session, stops the server and sends the student back.
    // Without it, the next person on this browser could reopen the notebook.
    if (hubUrl) window.location.assign(`${hubUrl}/hub/logout`);
  }, [session]);

  const value = useMemo<AuthState>(
    () => ({ session, loading, signIn, signUp, signOut }),
    [session, loading, signIn, signUp, signOut]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthProvider;
