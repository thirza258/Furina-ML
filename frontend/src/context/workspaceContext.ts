/**
 * Shared workspace state: the dataset the user is working on and the model
 * they just trained.
 *
 * Pages are swapped by `selectedTool` in App.tsx rather than by route params,
 * so there is nowhere in the URL to carry "which dataset am I on". This
 * context is what makes "upload on Preprocessing -> train on Training ->
 * evaluate on Evaluation" a single continuous flow.
 *
 * The context object and the hook live here (and the provider component lives
 * in WorkspaceProvider.tsx) so that neither file mixes component and
 * non-component exports, which would break React Fast Refresh.
 */
import { createContext, useContext } from "react";

import type { DatasetDetail, TrainingRunDetail } from "../interface/MlTypes";

export interface WorkspaceState {
  /** The dataset currently loaded, or null before the first upload. */
  dataset: DatasetDetail | null;
  /** The most recently trained model, or null before the first training run. */
  run: TrainingRunDetail | null;
  /** True while ids saved in localStorage are being re-fetched on a page load. */
  restoring: boolean;
  /** Set when the API could not be reached during restore. */
  restoreError: string | null;

  setDataset: (dataset: DatasetDetail | null) => void;
  setRun: (run: TrainingRunDetail | null) => void;
  /** Re-fetch the active dataset from the API (after a preprocessing step). */
  refreshDataset: (previewRows?: number) => Promise<void>;
  /** Forget the dataset and run, including the persisted ids. */
  clearWorkspace: () => void;
  /** Switch the visible page — the sidebar's tool names, e.g. "Data Training". */
  goToTool: (tool: string) => void;
}

export const WorkspaceContext = createContext<WorkspaceState | null>(null);

export const DATASET_STORAGE_KEY = "furina-ml.dataset-id";
export const RUN_STORAGE_KEY = "furina-ml.run-id";

export function useWorkspace(): WorkspaceState {
  const value = useContext(WorkspaceContext);
  if (!value) {
    throw new Error("useWorkspace must be used inside a <WorkspaceProvider>.");
  }
  return value;
}
