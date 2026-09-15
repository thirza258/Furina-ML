import { useState } from "react";

const TOKENS = ["the", "small", "cat", "sat", "here"];

export function AttentionExplorer() {
  const [query, setQuery] = useState(2);
  const [causal, setCausal] = useState(true);
  const [temperature, setTemperature] = useState(1);
  const scores = TOKENS.map((_, index) => 2 / (1 + Math.abs(index - query)));
  const exponentials = scores.map((score, index) => causal && index > query ? 0 : Math.exp(score / temperature));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  const weights = exponentials.map((value) => value / denominator);
  return (
    <div>
      <h3 className="text-h3 mb-2">Inspect one attention row</h3>
      <p className="text-small text-ink-muted mb-5">Hand-assigned compatibility scores illustrate masking and softmax. These are teaching values, not attention extracted from a trained model.</p>
      <div className="grid sm:grid-cols-2 gap-4 mb-4">
        <label className="block"><span className="field-label block mb-2">Query position</span><select value={query} onChange={(event) => setQuery(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{TOKENS.map((token, index) => <option key={index} value={index}>{index + 1}. {token}</option>)}</select></label>
        <label className="flex gap-3 items-center min-h-11"><input type="checkbox" checked={causal} onChange={(event) => setCausal(event.target.checked)} className="accent-primary" />Apply causal mask</label>
      </div>
      <label htmlFor="attention-temperature" className="block text-small">Softmax temperature: {temperature.toFixed(1)}</label>
      <input id="attention-temperature" type="range" min={0.5} max={2} step={0.1} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} className="w-full min-h-11 accent-primary" />
      <div className="overflow-x-auto mt-4"><table className="w-full text-small text-left">
        <caption className="text-left text-ink-muted mb-2">Attention from position {query + 1}: “{TOKENS[query]}”</caption>
        <thead><tr className="border-b border-rule"><th scope="col" className="p-2">Key</th><th scope="col" className="p-2">Access</th><th scope="col" className="p-2">Weight</th><th scope="col" className="p-2">Share</th></tr></thead>
        <tbody>{TOKENS.map((token, index) => <tr key={index} className="border-b border-rule"><th scope="row" className="p-2">{index + 1}. {token}</th><td className="p-2">{causal && index > query ? "Blocked" : "Allowed"}</td><td className="p-2 tabular">{(weights[index] * 100).toFixed(1)}%</td><td className="p-2 w-1/3"><div className="h-2 bg-paper-sunken" aria-hidden="true"><div className="h-2 bg-series-1" style={{ width: `${weights[index] * 100}%` }} /></div></td></tr>)}</tbody>
      </table></div>
      <p className="text-small mt-4" aria-live="polite">Weights sum to {weights.reduce((sum, value) => sum + value, 0).toFixed(3)}. {causal ? "Future positions contribute zero." : "Future positions are visible; this would leak answers in ordinary next-token training."} Lower temperature concentrates the distribution.</p>
    </div>
  );
}

export function TrainingBudgetExplorer() {
  const [millions, setMillions] = useState(100);
  const [trainablePercent, setTrainablePercent] = useState(100);
  const [microbatch, setMicrobatch] = useState(4);
  const [accumulation, setAccumulation] = useState(4);
  const [workers, setWorkers] = useState(1);
  const [context, setContext] = useState(512);
  const weightGiB = millions * 1e6 * 2 / 2 ** 30;
  const stateGiB = millions * 1e6 * trainablePercent / 100 * 14 / 2 ** 30;
  const batch = microbatch * accumulation * workers;
  return (
    <div>
      <h3 className="text-h3 mb-2">Estimate training state and update size</h3>
      <p className="text-small text-ink-muted mb-5">An arithmetic exercise with explicit assumptions. Actual memory also includes activations, temporary buffers, caches and framework overhead.</p>
      <div className="grid sm:grid-cols-2 gap-4">
        <label><span className="field-label block mb-2">Total parameters</span><select value={millions} onChange={(event) => setMillions(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{[1, 10, 100, 1000, 7000].map((value) => <option key={value} value={value}>{value >= 1000 ? `${value / 1000} billion` : `${value} million`}</option>)}</select></label>
        <label><span className="field-label block mb-2">Trainable share</span><select value={trainablePercent} onChange={(event) => setTrainablePercent(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{[1, 5, 10, 100].map((value) => <option key={value} value={value}>{value}%</option>)}</select></label>
      </div>
      <dl className="grid sm:grid-cols-3 gap-4 my-5 tabular" aria-live="polite">
        <div><dt className="field-label">Weights (GiB)</dt><dd className="text-h2">{weightGiB.toFixed(2)}</dd></div>
        <div><dt className="field-label">Additional training state (GiB)</dt><dd className="text-h2">{stateGiB.toFixed(2)}</dd></div>
        <div><dt className="field-label">Subtotal (GiB)</dt><dd className="text-h2">{(weightGiB + stateGiB).toFixed(2)}</dd></div>
      </dl>
      <p className="text-small mb-5">Assumes 2 bytes per stored weight and, per trainable parameter, 2 for gradients, 4 for a master weight and 8 for Adam moments. No sharding or quantization is assumed. A smaller trainable share is hypothetical and does not measure a particular LoRA configuration.</p>
      <div className="grid sm:grid-cols-2 gap-4">
        {[
          { label: "Microbatch per worker", value: microbatch, set: setMicrobatch, values: [1, 2, 4, 8, 16] },
          { label: "Accumulation steps", value: accumulation, set: setAccumulation, values: [1, 2, 4, 8, 16] },
          { label: "Workers", value: workers, set: setWorkers, values: [1, 2, 4, 8] },
          { label: "Valid tokens per sequence", value: context, set: setContext, values: [128, 512, 1024, 4096] },
        ].map((field) => <label key={field.label}><span className="field-label block mb-2">{field.label}</span><select value={field.value} onChange={(event) => field.set(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{field.values.map((value) => <option key={value}>{value}</option>)}</select></label>)}
      </div>
      <p className="mt-5 text-small tabular" aria-live="polite">Effective global batch: <strong>{batch} sequences</strong> · <strong>{(batch * context).toLocaleString()} valid tokens per update</strong>, assuming full equally sized microbatches. This does not account for padding or a partial final accumulation group.</p>
    </div>
  );
}

export function AudioExplorer() {
  const [sampleRate, setSampleRate] = useState(16000);
  const [duration, setDuration] = useState(3);
  const [hop, setHop] = useState(160);
  const [path, setPath] = useState("a a _ a b b _");
  const labels = path.trim().split(/\s+/).filter(Boolean);
  const decoded = labels.filter((label, index) => label !== labels[index - 1]).filter((label) => label !== "_");
  return (
    <div>
      <h3 className="text-h3 mb-2">Connect samples, time and CTC labels</h3>
      <p className="text-small text-ink-muted mb-5">Explore audio arithmetic without uploading or recording sound.</p>
      <div className="grid sm:grid-cols-2 gap-4">
        <label><span className="field-label block mb-2">Sample rate (Hz)</span><select value={sampleRate} onChange={(event) => setSampleRate(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{[8000, 16000, 24000, 48000].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label><span className="field-label block mb-2">Hop length (samples)</span><select value={hop} onChange={(event) => setHop(Number(event.target.value))} className="w-full min-h-11 border border-rule-strong bg-paper px-3">{[80, 160, 320, 480].map((value) => <option key={value}>{value}</option>)}</select></label>
      </div>
      <label htmlFor="audio-duration" className="block text-small mt-4">Duration: {duration} seconds</label>
      <input id="audio-duration" type="range" min={1} max={10} value={duration} onChange={(event) => setDuration(Number(event.target.value))} className="min-h-11 w-full accent-primary" />
      <dl className="grid sm:grid-cols-3 gap-4 my-5 tabular" aria-live="polite">
        <div><dt className="field-label">Samples</dt><dd className="text-h2">{(duration * sampleRate).toLocaleString()}</dd></div>
        <div><dt className="field-label">Hop duration (ms)</dt><dd className="text-h2">{(hop / sampleRate * 1000).toFixed(2)}</dd></div>
        <div><dt className="field-label">Frame starts / second, approx.</dt><dd className="text-h2">{(sampleRate / hop).toFixed(1)}</dd></div>
      </dl>
      <p className="text-small mb-5">Mono 16-bit PCM payload: {(duration * sampleRate * 2 / 1000).toFixed(1)} kB, excluding file headers. Exact feature-frame counts depend on window size and padding.</p>
      <label htmlFor="ctc-path" className="block font-medium mb-2">CTC path · separate labels with spaces; _ means blank</label>
      <input id="ctc-path" value={path} onChange={(event) => setPath(event.target.value)} className="w-full min-h-11 border border-rule-strong bg-paper px-3 font-mono text-small" />
      <p className="text-small mt-3" aria-live="polite">Decoded labels: <strong className="font-mono">{decoded.length ? decoded.join(" ") : "(empty)"}</strong>. Consecutive repeats are merged before blanks are removed.</p>
    </div>
  );
}
