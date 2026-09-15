import { useState } from "react";

interface SystemStep {
  id: string;
  title: string;
  contract: string;
  failure: string;
}

const OFFLINE_STEPS: SystemStep[] = [
  { id: "sources", title: "1. Source events", contract: "Stable event IDs, event time, availability time, units and schema version. Retain corrections and lineage under the data policy.", failure: "Duplicate deliveries or a unit change can corrupt every downstream row. Validate and quarantine before publishing." },
  { id: "features", title: "2. Historical features", contract: "Join each prediction to feature versions available at its original timestamp. Apply freshness and missing-value policies.", failure: "A backfilled value can leak future information. Test event time and availability time independently." },
  { id: "training", title: "3. Train & evaluate", contract: "Fit transforms on training data, select the complete decision on validation data, and keep independent evaluation separate.", failure: "A successful training job can produce an unacceptable candidate. A failed quality gate keeps the current release active." },
  { id: "release", title: "4. Release package", contract: "Publish compatible model, transformations, feature contract, threshold and evaluation metadata as one version.", failure: "Weights can load while feature semantics are incompatible. Check contracts and retain a usable previous release." },
];

export function SystemArchitectureExplorer() {
  const [mode, setMode] = useState("online");
  const [selectedId, setSelectedId] = useState("features");
  const online = mode === "online";
  const serving: SystemStep[] = [
    { id: "trigger", title: online ? "1. Live request" : "1. Scheduled job", contract: online ? "Validate an eligible request and propagate its remaining deadline." : "Choose a complete population snapshot and a publication deadline for the run.", failure: online ? "Reject malformed requests before expensive work. Bound admission when overloaded." : "Detect missing input partitions and prevent overlapping jobs from publishing partial output." },
    { id: "lookup", title: online ? "2. Feature lookup" : "2. Snapshot read", contract: online ? "Read fresh, available values using bounded dependency timeouts." : "Read one declared feature snapshot with its schema and availability cutoff.", failure: online ? "A dependency outage invokes a fallback that does not need the same dependency." : "A stale or mixed-version snapshot can invalidate the entire prediction batch." },
    { id: "inference", title: "3. Approved model", contract: "Load the approved package, apply its transforms, score inputs and use its saved decision policy.", failure: "A failed candidate can be removed from exposure while compatible baseline behavior remains available." },
    { id: "delivery", title: online ? "4. Response & action" : "4. Publish & lookup", contract: online ? "Return the score, action, versions, timestamps and fallback status before the deadline." : "Publish a complete versioned result set with prediction time and expiry; consumers look up that version.", failure: online ? "A technically successful fallback may still reduce decision quality. Monitor it separately." : "Define behavior for missing entities, expired predictions and failed publication." },
  ];
  const feedback: SystemStep[] = [
    { id: "log", title: "1. Prediction log", contract: "Record prediction ID, release, action, exposure, relevant feature metadata and fallback reason.", failure: "Missing IDs or unbounded payload logging can make outcome joins unreliable or retain unnecessary data." },
    { id: "outcomes", title: "2. Mature outcomes", contract: "Join labels available by a declared cutoff. Report coverage and a cohort old enough for the label window.", failure: "Treating pending outcomes as negatives biases evaluation toward quickly completed cases." },
    { id: "monitor", title: "3. Review & retrain", contract: "Inspect service, data and quality signals. A justified retraining trigger returns to the offline path with fresh gates.", failure: "Drift alone does not prove that retraining or promoting a candidate will help." },
  ];
  const lanes = [
    { title: "Offline · build the release", steps: OFFLINE_STEPS },
    { title: online ? "Online · each eligible request" : "Batch · each scheduled run", steps: serving },
    { title: "Feedback · observe and improve", steps: feedback },
  ];
  const selected = [...OFFLINE_STEPS, ...serving, ...feedback].find((step) => step.id === selectedId)!;

  return (
    <div>
      <h3 className="text-h3 mb-3">Trace a prediction system</h3>
      <p className="text-small text-ink-muted mb-5">Choose a serving pattern, then select a component to inspect its contract and a failure to plan for. These are design choices for the delivery case.</p>
      <label className="block mb-6">
        <span className="field-label block mb-2">Serving pattern</span>
        <select value={mode} onChange={(event) => setMode(event.target.value)} className="w-full min-h-11 border border-rule-strong bg-paper px-3">
          <option value="online">Online · decide at request time</option>
          <option value="batch">Batch · publish scores on a schedule</option>
        </select>
      </label>
      <div className="space-y-5" aria-label="Prediction system architecture">
        {lanes.map((lane) => (
          <section key={lane.title}>
            <h4 className="font-sans font-medium text-small mb-2">{lane.title}</h4>
            <ol className="grid sm:grid-cols-2 gap-2">
              {lane.steps.map((step) => (
                <li key={step.id}>
                  <button type="button" aria-pressed={selectedId === step.id} aria-controls="system-component-detail"
                    onClick={() => setSelectedId(step.id)}
                    className={"w-full min-h-12 px-3 py-3 text-left text-small border " + (selectedId === step.id ? "border-primary bg-paper-sunken font-semibold" : "border-rule bg-paper hover:border-primary")}>
                    {step.title}
                  </button>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
      <p className="text-small text-ink-muted my-5">Connections: the release package supplies serving; serving writes the prediction log; reviewed outcomes can trigger another training run. Numbers show order within each path.</p>
      <div id="system-component-detail" className="border-l-2 border-accent pl-4" aria-live="polite" aria-atomic="true">
        <h4 className="text-h3 mb-2">{selected.title}</h4>
        <p className="text-small mb-3"><strong>Contract:</strong> {selected.contract}</p>
        <p className="text-small"><strong>Failure to plan for:</strong> {selected.failure}</p>
      </div>
    </div>
  );
}

export function SystemBudgetExplorer() {
  const [peak, setPeak] = useState(1000);
  const [concurrency, setConcurrency] = useState(4);
  const [serviceMs, setServiceMs] = useState(100);
  const [utilization, setUtilization] = useState(70);
  const [requests, setRequests] = useState(1_000_000);
  const [allowedPerMillion, setAllowedPerMillion] = useState(1000);
  const [failures, setFailures] = useState(300);
  const perReplica = concurrency * 1000 / serviceMs * utilization / 100;
  const replicas = Math.ceil(peak / perReplica);
  const allowed = requests * allowedPerMillion / 1_000_000;
  const remaining = allowed - failures;

  return (
    <div>
      <h3 className="text-h3 mb-3">Plan serving capacity and an error budget</h3>
      <p className="text-small text-ink-muted mb-5">A simplified arithmetic model. Average service demand and independent replica capacity do not predict queueing, burst behavior, p99 latency or shared-dependency limits.</p>
      <div className="grid sm:grid-cols-2 gap-4">
        {[
          { label: "Peak requests / second", value: peak, set: setPeak, values: [100, 1000, 5000, 10000] },
          { label: "Concurrent slots / replica", value: concurrency, set: setConcurrency, values: [1, 2, 4, 8] },
          { label: "Mean slot occupancy (ms)", value: serviceMs, set: setServiceMs, values: [10, 50, 100, 250] },
          { label: "Target utilization (%)", value: utilization, set: setUtilization, values: [50, 70, 80, 90] },
        ].map((field) => (
          <label key={field.label}><span className="field-label block mb-2">{field.label}</span>
            <select value={field.value} onChange={(event) => field.set(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">
              {field.values.map((value) => <option key={value} value={value}>{value.toLocaleString()}</option>)}
            </select>
          </label>
        ))}
      </div>
      <dl className="grid sm:grid-cols-3 gap-4 my-6 tabular" aria-live="polite">
        <div><dt className="field-label">Target requests / second / replica</dt><dd className="text-h2">{perReplica.toFixed(1)}</dd></div>
        <div><dt className="field-label">Replicas for peak</dt><dd className="text-h2">{replicas}</dd></div>
        <div><dt className="field-label">With one spare replica</dt><dd className="text-h2">{replicas + 1}</dd></div>
      </dl>
      <p className="text-small mb-6">Capacity = concurrent slots × 1,000 / mean occupancy in ms × target utilization. Round peak / capacity up to a whole replica. One spare covers one replica loss under these assumptions; it does not cover a zone or a shared dependency.</p>
      <h4 className="text-h3 mb-3">Request-based reliability target</h4>
      <div className="grid sm:grid-cols-2 gap-4">
        <label><span className="field-label block mb-2">Eligible requests in the window</span><select value={requests} onChange={(event) => setRequests(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{[1_000_000, 5_000_000, 10_000_000].map((value) => <option key={value} value={value}>{value.toLocaleString()}</option>)}</select></label>
        <label><span className="field-label block mb-2">Deadline-success SLO</span><select value={allowedPerMillion} onChange={(event) => setAllowedPerMillion(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{[1000, 100, 10].map((value) => <option key={value} value={value}>{(100 - value / 10000).toFixed(3)}%</option>)}</select></label>
        <label><span className="field-label block mb-2">Unsuccessful requests</span><select value={failures} onChange={(event) => setFailures(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{[0, 10, 100, 300, 1500].map((value) => <option key={value} value={value}>{value.toLocaleString()}</option>)}</select></label>
      </div>
      <p className="text-small mt-5" aria-live="polite">Allowed: <strong>{allowed.toLocaleString()} unsuccessful requests</strong>. {remaining >= 0 ? <>Remaining: <strong>{remaining.toLocaleString()}</strong>.</> : <>Budget exceeded by <strong>{(-remaining).toLocaleString()}</strong>.</>} Consumed: <strong>{(failures / allowed * 100).toFixed(1)}%</strong>.</p>
      <p className="text-small text-ink-muted mt-3">Choose the eligibility rules, deadline and window in your design. This example counts deadline failures; fallback rate and prediction quality need separate indicators. Request counts are not downtime estimates.</p>
    </div>
  );
}
