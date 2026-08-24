import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import services, { toApiError } from "../services/services";
import type { DatasetDetail, TrainingRunDetail } from "../interface/MlTypes";
import {
  DATASET_STORAGE_KEY,
  RUN_STORAGE_KEY,
  WorkspaceContext,
} from "./workspaceContext";
import type { WorkspaceState } from "./workspaceContext";

/** localStorage throws in private-mode Safari and when site data is blocked. */
function readStored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    /* Persistence is a convenience; losing it must never break the app. */
  }
}

interface WorkspaceProviderProps {
  children: React.ReactNode;
  /** Wired to App.tsx's page switcher so pages can advance the user. */
  onNavigate: (tool: string) => void;
}

const WorkspaceProvider: React.FC<WorkspaceProviderProps> = ({
  children,
  onNavigate,
}) => {
  const [dataset, setDatasetState] = useState<DatasetDetail | null>(null);
  const [run, setRunState] = useState<TrainingRunDetail | null>(null);
  const [restoring, setRestoring] = useState<boolean>(true);
  const [restoreError, setRestoreError] = useState<string | null>(null);

  // Keep the latest navigate callback without making it a dependency of the
  // memoized context value (App.tsx re-creates it on every render).
  const navigateRef = useRef(onNavigate);
  navigateRef.current = onNavigate;

  const setDataset = useCallback((next: DatasetDetail | null) => {
    setDatasetState(next);
    writeStored(DATASET_STORAGE_KEY, next?.id ?? null);
    if (!next) {
      setRunState(null);
      writeStored(RUN_STORAGE_KEY, null);
    }
  }, []);

  const setRun = useCallback((next: TrainingRunDetail | null) => {
    setRunState(next);
    writeStored(RUN_STORAGE_KEY, next?.id ?? null);
  }, []);

  const refreshDataset = useCallback(
    async (previewRows = 10) => {
      if (!dataset?.id) return;
      const fresh = await services.getDataset(dataset.id, previewRows);
      setDatasetState(fresh);
    },
    [dataset?.id]
  );

  const clearWorkspace = useCallback(() => {
    setDatasetState(null);
    setRunState(null);
    writeStored(DATASET_STORAGE_KEY, null);
    writeStored(RUN_STORAGE_KEY, null);
  }, []);

  const goToTool = useCallback((tool: string) => {
    navigateRef.current(tool);
  }, []);

  // Rehydrate from the ids saved on the last visit. A stale id (dataset or run
  // deleted) is a 404: drop it silently, because there is nothing the user can
  // do about it. Anything else — server down, 500 — is a real failure and is
  // surfaced through `restoreError`, which the pages render in a banner.
  useEffect(() => {
    let cancelled = false;

    const restore = async () => {
      const datasetId = readStored(DATASET_STORAGE_KEY);
      const runId = readStored(RUN_STORAGE_KEY);

      if (!datasetId && !runId) {
        if (!cancelled) setRestoring(false);
        return;
      }

      try {
        if (datasetId) {
          const fresh = await services.getDataset(datasetId);
          if (!cancelled) setDatasetState(fresh);
        }
      } catch (error) {
        const apiError = toApiError(error);
        if (apiError.status === 404) writeStored(DATASET_STORAGE_KEY, null);
        else if (!cancelled) setRestoreError(apiError.detail);
      }

      try {
        if (runId) {
          const fresh = await services.getRun(runId);
          if (!cancelled) setRunState(fresh);
        }
      } catch (error) {
        const apiError = toApiError(error);
        if (apiError.status === 404) writeStored(RUN_STORAGE_KEY, null);
        else if (!cancelled) setRestoreError(apiError.detail);
      }

      if (!cancelled) setRestoring(false);
    };

    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo<WorkspaceState>(
    () => ({
      dataset,
      run,
      restoring,
      restoreError,
      setDataset,
      setRun,
      refreshDataset,
      clearWorkspace,
      goToTool,
    }),
    [
      dataset,
      run,
      restoring,
      restoreError,
      setDataset,
      setRun,
      refreshDataset,
      clearWorkspace,
      goToTool,
    ]
  );

  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
};

export default WorkspaceProvider;
