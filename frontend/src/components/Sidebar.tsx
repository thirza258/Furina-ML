/**
 * Tool navigation, plus the progress spine of the app.
 *
 * The three Traditional ML tools are not siblings — they are a pipeline, and
 * each one needs the previous one's output. The sidebar says so: a finished
 * step is ticked, a step whose input is missing says what it is waiting for.
 * It never disables a step, though. Being told why a page will be empty and
 * then being allowed to look anyway is far less frustrating than a dead
 * button, and every page renders its own empty state with a way forward.
 */
import React from "react";

import { useWorkspace } from "../context/workspaceContext";

interface SidebarProps {
  selectedTool: string;
  setSelectedTool: (tool: string) => void;
}

/** Must match the strings App.tsx switches on — a typo here is a dead link. */
const APPLICATION_TOOLS = ["Landing Page", "About Us"] as const;

type StepState = "done" | "ready" | "locked";

interface PipelineStep {
  tool: string;
  state: StepState;
  /** Shown under a locked step: what has to happen first. */
  hint: string;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedTool, setSelectedTool }) => {
  const { dataset, run, restoring } = useWorkspace();

  const steps: PipelineStep[] = [
    {
      tool: "Data Preprocessing",
      state: dataset ? "done" : "ready",
      hint: dataset
        ? `${dataset.name} loaded`
        : "Start here — upload a CSV",
    },
    {
      tool: "Data Training",
      state: run ? "done" : dataset ? "ready" : "locked",
      hint: run
        ? `${run.algorithm_label} trained`
        : dataset
          ? "Pick a target and train"
          : "Locked — upload a dataset first",
    },
    {
      tool: "Model Evaluation",
      state: run ? "ready" : "locked",
      hint: run
        ? "Metrics and predictions ready"
        : "Locked — train a model first",
    },
  ];

  const renderToolButton = (
    tool: string,
    options: { index?: number; step?: PipelineStep } = {}
  ) => {
    const { index, step } = options;
    const isSelected = selectedTool === tool;

    return (
      <li key={tool} className="mb-2">
        <button
          type="button"
          onClick={() => setSelectedTool(tool)}
          aria-current={isSelected ? "page" : undefined}
          className={`w-full text-left p-2 rounded border-b border-gray-300 hover:bg-gray-300 transition-colors ${
            isSelected ? "font-bold bg-gray-300" : ""
          }`}
        >
          <span className="flex items-center gap-2">
            {step && (
              <span
                aria-hidden="true"
                className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                  step.state === "done"
                    ? "bg-state-good text-white"
                    : step.state === "ready"
                      ? "bg-white text-secondary border border-secondary"
                      : "bg-gray-300 text-ink-muted"
                }`}
              >
                {step.state === "done" ? "✓" : index}
              </span>
            )}
            <span className="flex-grow">{tool}</span>
            {step && (
              <span className="sr-only">
                {step.state === "done"
                  ? " — step complete"
                  : step.state === "locked"
                    ? " — locked"
                    : " — ready"}
              </span>
            )}
          </span>
          {step && (
            <span
              className={`block pl-7 text-xs ${
                step.state === "locked" ? "text-ink-muted" : "text-gray-600"
              }`}
            >
              {restoring ? "Restoring…" : step.hint}
            </span>
          )}
        </button>
      </li>
    );
  };

  return (
    <div className="w-1/4 bg-gray-200 h-full p-4 border-r-2 border-white z-10 shadow-lg flex-shrink-0 flex flex-col">
      <div className="flex-shrink-0">
        <h2 className="mb-4 text-xl font-semibold text-secondary">Tools</h2>
        <div className="overflow-y-auto">
          <div className="mb-4">
            <details className="group" open>
              <summary className="flex justify-between items-center cursor-pointer font-medium text-secondary mb-2 hover:bg-gray-300 p-2 rounded border-b border-gray-300">
                Applications
                <span aria-hidden="true" className="transition group-open:rotate-180">
                  ▼
                </span>
              </summary>
              <ul className="pl-4 text-sm text-secondary">
                {APPLICATION_TOOLS.map((tool) => renderToolButton(tool))}
              </ul>
            </details>
          </div>

          <div className="mb-4">
            <details className="group" open>
              <summary className="flex justify-between items-center cursor-pointer font-medium text-secondary mb-2 hover:bg-gray-300 p-2 rounded border-b border-gray-300">
                Traditional Machine Learning
                <span aria-hidden="true" className="transition group-open:rotate-180">
                  ▼
                </span>
              </summary>
              <ol className="pl-4 text-sm text-secondary">
                {steps.map((step, i) =>
                  renderToolButton(step.tool, { index: i + 1, step })
                )}
              </ol>
              <p className="pl-4 pr-2 pt-1 text-xs text-ink-muted">
                Each step feeds the next. A locked step still opens — it will
                tell you what it needs.
              </p>
            </details>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
