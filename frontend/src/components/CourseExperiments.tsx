import { useState } from "react";
import { SystemArchitectureExplorer, SystemBudgetExplorer } from "./SystemDesignExperiments";
import type { CourseLesson } from "../data/courseCatalogue";
import { AttentionExplorer, AudioExplorer, TrainingBudgetExplorer } from "./AdvancedCourseExperiments";

function SplitExplorer() {
  const [testPercent, setTestPercent] = useState(20);
  const testRows = testPercent * 10;
  const trainRows = 1000 - testRows;

  return (
    <div>
      <h3 className="text-h3 mb-2">Explore a train/test split</h3>
      <p className="text-small text-ink-muted mb-5">
        A teaching example with 1,000 independent labelled rows and a 10% minority class.
      </p>
      <label htmlFor="course-test-size" className="block font-medium mb-2">
        Test set: {testPercent}%
      </label>
      <input
        id="course-test-size" type="range" min={5} max={50} step={5}
        value={testPercent} onChange={(event) => setTestPercent(Number(event.target.value))}
        className="w-full accent-primary min-h-11"
        aria-valuetext={`${testPercent}% test, ${100 - testPercent}% training`}
      />
      <div className="flex h-5 my-4" aria-hidden="true">
        <span className="bg-series-1" style={{ width: `${100 - testPercent}%` }} />
        <span className="bg-series-2" style={{ width: `${testPercent}%` }} />
      </div>
      <dl className="grid sm:grid-cols-3 gap-4 tabular" aria-live="polite">
        <div><dt className="field-label">Training rows</dt><dd className="text-h2 font-serif">{trainRows}</dd></div>
        <div><dt className="field-label">Test rows</dt><dd className="text-h2 font-serif">{testRows}</dd></div>
        <div><dt className="field-label">Minority test rows, approx.</dt><dd className="text-h2 font-serif">{testRows / 10}</dd></div>
      </dl>
      <p className="text-small mt-4">
        One changed test prediction moves accuracy by {(100 / testRows).toFixed(2)} percentage points.
        Increasing the test share gives more evaluation examples and leaves fewer for learning.
      </p>
    </div>
  );
}

const CLASSIFICATION_EXAMPLES = [
  { name: "Flag fewer parcels", tp: 10, fp: 2, fn: 10, tn: 78 },
  { name: "Middle setting", tp: 16, fp: 8, fn: 4, tn: 72 },
  { name: "Flag more parcels", tp: 19, fp: 20, fn: 1, tn: 60 },
];

function ClassificationExplorer() {
  const [setting, setSetting] = useState(1);
  const { tp, fp, fn, tn } = CLASSIFICATION_EXAMPLES[setting];
  const precision = tp / (tp + fp);
  const recall = tp / (tp + fn);
  const f1 = 2 * precision * recall / (precision + recall);

  return (
    <div>
      <h3 className="text-h3 mb-2">False alarms or missed parcels?</h3>
      <p className="text-small text-ink-muted mb-5">
        Illustrative decisions on the same 100 parcels: 20 late and 80 on time.
        “Late” is the positive class. These examples are independent of your trained model.
      </p>
      <fieldset>
        <legend className="font-medium mb-2">Choose a flagging strategy</legend>
        <div className="flex flex-wrap gap-2">
          {CLASSIFICATION_EXAMPLES.map((example, index) => (
            <label key={example.name} className={`btn gap-2 ${setting === index ? "btn-primary" : "btn-secondary"}`}>
              <input
                type="radio" name="course-strategy" value={index} checked={setting === index}
                onChange={() => setSetting(index)} className="accent-primary"
              />
              {example.name}
            </label>
          ))}
        </div>
      </fieldset>
      <div className="overflow-x-auto mt-5">
        <table className="w-full text-small border-collapse">
          <caption className="text-left text-ink-muted mb-2">Rows = actual · columns = predicted</caption>
          <thead><tr className="border-b border-rule">
            <th scope="col" className="text-left p-3">Actual label</th>
            <th scope="col" className="p-3">Predicted on time</th>
            <th scope="col" className="p-3">Predicted late</th>
          </tr></thead>
          <tbody>
            <tr className="border-b border-rule">
              <th scope="row" className="text-left p-3">On time</th>
              <td className="p-3 text-center bg-paper-sunken">{tn} correct</td>
              <td className="p-3 text-center">{fp} false alarms</td>
            </tr>
            <tr className="border-b border-rule">
              <th scope="row" className="text-left p-3">Late</th>
              <td className="p-3 text-center">{fn} missed</td>
              <td className="p-3 text-center bg-paper-sunken">{tp} found</td>
            </tr>
          </tbody>
        </table>
      </div>
      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-5 tabular" aria-live="polite">
        {[
          ["Overall accuracy", (tp + tn) / 100],
          ["Late: precision", precision],
          ["Late: recall", recall],
          ["Late: F1", f1],
        ].map(([label, value]) => (
          <div key={label}><dt className="field-label">{label}</dt><dd className="text-h2 font-serif">{(Number(value) * 100).toFixed(1)}%</dd></div>
        ))}
      </dl>
      <p className="text-small mt-4">
        Precision = found / all flagged. Recall = found / all actually late.
        Flagging more parcels here finds more late deliveries and creates more false alarms.
      </p>
    </div>
  );
}

function RegressionExplorer() {
  const [lastPrediction, setLastPrediction] = useState(36);
  const actual = [10, 20, 30];
  const predicted = [12, 18, lastPrediction];
  const errors = actual.map((value, index) => value - predicted[index]);
  const mae = errors.reduce((sum, error) => sum + Math.abs(error), 0) / errors.length;
  const mse = errors.reduce((sum, error) => sum + error * error, 0) / errors.length;

  return (
    <div>
      <h3 className="text-h3 mb-2">See the effect of a large miss</h3>
      <p className="text-small text-ink-muted mb-5">Three illustrative delivery times, measured in hours.</p>
      <label htmlFor="course-prediction" className="block font-medium mb-2">
        Third prediction: {lastPrediction} hours · actual: 30 hours
      </label>
      <input
        id="course-prediction" type="range" min={10} max={60} step={1}
        value={lastPrediction} onChange={(event) => setLastPrediction(Number(event.target.value))}
        className="w-full accent-primary min-h-11" aria-valuetext={`${lastPrediction} hours`}
      />
      <div className="overflow-x-auto my-4">
        <table className="w-full text-small text-left border-collapse">
          <caption className="text-left text-ink-muted mb-2">Residual = actual − predicted; absolute error ignores the sign.</caption>
          <thead><tr className="border-b border-rule">
            <th scope="col" className="p-3">Actual (h)</th>
            <th scope="col" className="p-3">Predicted (h)</th>
            <th scope="col" className="p-3">Residual (h)</th>
          </tr></thead>
          <tbody>{actual.map((value, index) => (
            <tr key={value} className="border-b border-rule">
              <td className="p-3">{value}</td><td className="p-3">{predicted[index]}</td><td className="p-3">{errors[index]}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <dl className="grid sm:grid-cols-3 gap-4 tabular" aria-live="polite">
        <div><dt className="field-label">MAE (hours)</dt><dd className="text-h2 font-serif">{mae.toFixed(2)}</dd></div>
        <div><dt className="field-label">RMSE (hours)</dt><dd className="text-h2 font-serif">{Math.sqrt(mse).toFixed(2)}</dd></div>
        <div><dt className="field-label">MSE (hours²)</dt><dd className="text-h2 font-serif">{mse.toFixed(2)}</dd></div>
      </dl>
      <p className="text-small mt-4">Move the prediction away from 30. Squaring errors makes a large miss contribute more strongly to RMSE than to MAE.</p>
    </div>
  );
}

export default function CourseExperiments({ kind }: { kind: NonNullable<CourseLesson["experiment"]> }) {
  return (
    <section className="callout my-8" aria-label="Interactive example">
      <p className="field-label mb-3">Try it · teaching example</p>
      {kind === "split" && <SplitExplorer />}
      {kind === "classification" && <ClassificationExplorer />}
      {kind === "regression" && <RegressionExplorer />}
      {kind === "attention" && <AttentionExplorer />}
      {kind === "training-budget" && <TrainingBudgetExplorer />}
      {kind === "audio" && <AudioExplorer />}
      {kind === "system-architecture" && <SystemArchitectureExplorer />}
      {kind === "system-budget" && <SystemBudgetExplorer />}
    </section>
  );
}
