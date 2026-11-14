/**
 * Polling for a lifecycle job.
 *
 * Training, search and AutoML are asynchronous: POST returns a job id and the
 * worker runs it out of band. This hook polls `services.getJob` until the job
 * reaches a terminal state, stops on unmount, and surfaces the job's own
 * `error`/`suggested_action` rather than inventing a message.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import services, { toApiError } from "../services/services";
import type { Job } from "../interface/LifecycleTypes";

/** How often to ask, in ms. Fast enough to feel live, slow enough not to hammer. */
const POLL_INTERVAL_MS = 1200;

const TERMINAL: Job["status"][] = ["completed", "failed", "cancelled"];

export interface JobPolling {
  job: Job | null;
  /** A transport/HTTP failure while polling, not the job's own error. */
  error: string | null;
  polling: boolean;
  /** Begin polling an existing job id. */
  watch: (jobId: string) => void;
  /** Stop polling and forget the job. */
  reset: () => void;
  /** Ask the worker to cancel; the poller keeps running until it lands. */
  cancel: () => Promise<void>;
}

export function useJobPolling(): JobPolling {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const cancelled = useRef(false);
  const timer = useRef<number | null>(null);
  const watchedId = useRef<string | null>(null);

  const clearTimer = useCallback(() => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  const poll = useCallback(
    async (jobId: string) => {
      try {
        const fresh = await services.getJob(jobId);
        if (cancelled.current) return;
        setJob(fresh);
        setError(null);
        if (TERMINAL.includes(fresh.status)) {
          setPolling(false);
          return;
        }
        timer.current = window.setTimeout(() => void poll(jobId), POLL_INTERVAL_MS);
      } catch (caught) {
        if (cancelled.current) return;
        const apiError = toApiError(caught);
        // A 404 means the job vanished; stop rather than poll forever.
        if (apiError.status === 404) {
          setPolling(false);
          setError(apiError.detail);
          return;
        }
        setError(apiError.detail);
        timer.current = window.setTimeout(() => void poll(jobId), POLL_INTERVAL_MS);
      }
    },
    []
  );

  const watch = useCallback(
    (jobId: string) => {
      clearTimer();
      cancelled.current = false;
      watchedId.current = jobId;
      setPolling(true);
      setJob(null);
      setError(null);
      void poll(jobId);
    },
    [clearTimer, poll]
  );

  const reset = useCallback(() => {
    clearTimer();
    cancelled.current = true;
    watchedId.current = null;
    setJob(null);
    setError(null);
    setPolling(false);
  }, [clearTimer]);

  const cancel = useCallback(async () => {
    if (!watchedId.current) return;
    try {
      const fresh = await services.cancelJob(watchedId.current);
      setJob(fresh);
    } catch (caught) {
      setError(toApiError(caught).detail);
    }
  }, []);

  useEffect(() => {
    return () => {
      cancelled.current = true;
      clearTimer();
    };
  }, [clearTimer]);

  return { job, error, polling, watch, reset, cancel };
}
