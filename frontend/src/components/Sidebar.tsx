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
import React, { useState } from "react";
import { Link } from "react-router-dom";

import { useWorkspace } from "../context/workspaceContext";

interface SidebarProps {
  selectedTool: string;
  setSelectedTool: (tool: string) => void;
}

/** Paths for each tool so search crawlers and users have real hyperlinks. */
const TOOL_PATHS: Record<string, string> = {
  "Landing Page": "/app",
  "Dashboard": "/app/dashboard",
  "Data Preprocessing": "/app/preprocessing",
  "Data Training": "/app/training",
  "Model Evaluation": "/app/evaluation",
  "Experiments": "/app/experiments",
  "Model Registry": "/app/models",
  "Deployments": "/app/deployments",
  "Courses": "/app/courses",
  "About Us": "/about",
};

/** Must match the strings App.tsx switches on — a typo here is a dead link. */
const APPLICATION_TOOLS = [
  "Landing Page",
  "Dashboard",
  "Experiments",
  "Model Registry",
  "Deployments",
  "Courses",
  "About Us",
] as const;

/**
 * What the sidebar prints, where that differs from the routing key.
 *
 * "Landing Page" is the key App.tsx and every `goToTool` call already agree
 * on, so it stays; but the page it opens is titled "Overview", and the front
 * page proper now lives outside the workbench at `/`. Showing the key here
 * would name one screen two different things.
 */
const TOOL_LABELS: Record<string, string> = {
  "Landing Page": "Overview",
};

function toolLabel(tool: string): string {
  return TOOL_LABELS[tool] ?? tool;
}

type StepState = "done" | "ready" | "locked";

interface PipelineStep {
  tool: string;
  state: StepState;
  /** Shown under a locked step: what has to happen first. */
  hint: string;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedTool, setSelectedTool }) => {
  const { dataset, run, restoring } = useWorkspace();
  const [mobileOpen, setMobileOpen] = useState(false);

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
      // A rule in the left margin marks the current tool, the way a change
      // bar marks a revised paragraph. No fill, no pill.
      <li key={tool}>
        <Link
          to={TOOL_PATHS[tool] ?? "/app"}
          onClick={() => { setSelectedTool(tool); setMobileOpen(false); }}
          aria-current={isSelected ? "page" : undefined}
          className={`block w-full text-left py-2 pl-3 pr-2 border-l-2 cursor-pointer transition-colors no-underline ${
            isSelected
              ? "border-primary bg-paper-raised"
              : "border-transparent hover:border-rule-strong hover:bg-paper-raised"
          }`}
        >
          <span className="flex items-baseline gap-2.5">
            {step && (
              <span
                aria-hidden="true"
                className={`flex-shrink-0 w-5 h-5 flex items-center justify-center font-sans text-caption font-semibold border ${
                  step.state === "done"
                    ? "bg-state-good border-state-good text-white"
                    : step.state === "ready"
                      ? "bg-paper-raised border-rule-strong text-ink"
                      : "bg-paper-sunken border-rule text-ink-muted"
                }`}
              >
                {step.state === "done" ? "✓" : index}
              </span>
            )}
            <span
              className={`flex-grow font-sans text-small ${
                isSelected ? "text-ink font-semibold" : "text-ink"
              }`}
            >
              {toolLabel(tool)}
            </span>
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
            <span className="block pl-[1.9rem] mt-0.5 font-sans text-caption text-ink-muted">
              {restoring ? "Restoring…" : step.hint}
            </span>
          )}
        </Link>
      </li>
    );
  };

  return (
    <div className="w-full md:w-64 lg:w-72 bg-paper md:h-full max-h-[45dvh] md:max-h-none border-b md:border-b-0 md:border-r border-rule z-10 flex-shrink-0 flex flex-col overflow-y-auto">
      <button
        type="button" aria-expanded={mobileOpen} aria-controls="workbench-tools"
        onClick={() => setMobileOpen((open) => !open)}
        className="md:hidden flex items-center justify-between px-5 py-3 text-small font-medium min-h-11"
      >
        <span>Tools · {toolLabel(selectedTool)}</span><span aria-hidden="true">{mobileOpen ? "−" : "+"}</span>
      </button>
      <div id="workbench-tools" className={`${mobileOpen ? "block" : "hidden"} md:block flex-shrink-0 px-4 py-5`}>
        <h2 className="field-label mb-4">Tools</h2>
        <div className="overflow-y-auto">
          <div className="mb-4">
            <details className="group" open>
              <summary className="flex justify-between items-center cursor-pointer font-serif text-body text-ink mb-2 pb-1.5 border-b border-rule list-none [&::-webkit-details-marker]:hidden">
                Applications
                <span aria-hidden="true" className="transition group-open:rotate-180">
                  ▼
                </span>
              </summary>
              <ul className="text-small">
                {APPLICATION_TOOLS.map((tool) => renderToolButton(tool))}
              </ul>
            </details>
          </div>

          <div className="mb-4">
            <details className="group" open>
              <summary className="flex justify-between items-center cursor-pointer font-serif text-body text-ink mb-2 pb-1.5 border-b border-rule list-none [&::-webkit-details-marker]:hidden">
                Traditional Machine Learning
                <span aria-hidden="true" className="transition group-open:rotate-180">
                  ▼
                </span>
              </summary>
              <ol className="text-small">
                {steps.map((step, i) =>
                  renderToolButton(step.tool, { index: i + 1, step })
                )}
              </ol>
              <p className="pl-3 pr-2 pt-3 font-sans text-caption text-ink-muted">
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
