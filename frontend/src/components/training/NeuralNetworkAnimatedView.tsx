import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DatasetDetail, RequestedTaskType } from "../../interface/MlTypes";

interface NeuralNetworkAnimatedViewProps {
  dataset: DatasetDetail | null;
  selectedTarget?: string;
  selectedFeatures?: string[];
  requestedTask?: RequestedTaskType;
  onSwitchToWeightAnalysis?: () => void;
  onSwitchToNormal?: () => void;
}

type ActivationFunction = "relu" | "sigmoid" | "tanh" | "linear";
type TrainingPhase = "idle" | "forward" | "loss" | "backward" | "update";

// Fallback teaching dataset if no dataset is loaded yet
const TEACHING_SAMPLES = [
  { inputs: [0.85, 0.12, 0.74, 0.35], target: 0.9 },
  { inputs: [0.15, 0.88, 0.22, 0.65], target: 0.1 },
  { inputs: [0.72, 0.31, 0.81, 0.40], target: 0.8 },
  { inputs: [0.25, 0.75, 0.19, 0.85], target: 0.2 },
  { inputs: [0.90, 0.05, 0.68, 0.20], target: 0.95 },
  { inputs: [0.30, 0.80, 0.35, 0.70], target: 0.25 },
];

const TEACHING_FEATURE_NAMES = ["Feature X₁ (Signal)", "Feature X₂ (Noise)", "Feature X₃ (Velocity)", "Feature X₄ (Pressure)"];
const TEACHING_TARGET_NAME = "Target Y (Yield)";

// Activation functions & derivatives
function activate(z: number, fn: ActivationFunction): number {
  switch (fn) {
    case "relu":
      return Math.max(0, z);
    case "sigmoid":
      return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));
    case "tanh":
      return Math.tanh(z);
    case "linear":
      return z;
  }
}

function activateDerivative(z: number, a: number, fn: ActivationFunction): number {
  switch (fn) {
    case "relu":
      return z > 0 ? 1 : 0;
    case "sigmoid":
      return a * (1 - a);
    case "tanh":
      return 1 - a * a;
    case "linear":
      return 1;
  }
}

export default function NeuralNetworkAnimatedView({
  dataset,
  selectedTarget,
  selectedFeatures,
  requestedTask = "auto",
  onSwitchToWeightAnalysis,
  onSwitchToNormal,
}: NeuralNetworkAnimatedViewProps) {
  // Network configuration
  const [hiddenLayerCount, setHiddenLayerCount] = useState<1 | 2>(1);
  const [neuronsL1, setNeuronsL1] = useState<number>(4);
  const [neuronsL2, setNeuronsL2] = useState<number>(3);
  const [activationFn, setActivationFn] = useState<ActivationFunction>("sigmoid");
  const [learningRate, setLearningRate] = useState<number>(0.05);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1);

  // Playback & Animation state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [phase, setPhase] = useState<TrainingPhase>("idle");
  const [epoch, setEpoch] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [currentLoss, setCurrentLoss] = useState<number>(0.42);
  const [currentMetric, setCurrentMetric] = useState<number>(0.65);
  const [pulseT, setPulseT] = useState<number>(0); // 0 to 1 for animated traveling signal

  // Selected sample row index
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number>(0);

  // Interactive inspect selection
  const [selectedNode, setSelectedNode] = useState<{ layer: number; index: number } | null>({
    layer: 1,
    index: 0,
  });
  const [selectedSynapse, setSelectedSynapse] = useState<{
    fromLayer: number;
    fromIdx: number;
    toLayer: number;
    toIdx: number;
  } | null>(null);

  // Derive feature names and normalized samples from dataset or teaching fallback
  const { featureNames, targetName, samples, isUsingDataset } = useMemo(() => {
    if (dataset && dataset.columns && dataset.preview && dataset.preview.rows.length > 0) {
      const activeTarget = selectedTarget || (dataset.columns.find((c) => c.suitable_as_target)?.name ?? dataset.columns[dataset.columns.length - 1].name);
      const activeFeatures = (selectedFeatures && selectedFeatures.length > 0)
        ? selectedFeatures.filter((f) => f !== activeTarget)
        : dataset.columns.filter((c) => c.name !== activeTarget).map((c) => c.name);

      const targetColIdx = dataset.preview.headers.indexOf(activeTarget);
      const featColIndices = activeFeatures.map((f) => dataset.preview.headers.indexOf(f)).filter((idx) => idx !== -1);

      // Take up to 5 feature names for clean visual presentation
      const visibleFeatures = activeFeatures.slice(0, 5);
      const visibleIndices = featColIndices.slice(0, 5);

      // Extract rows and normalize numeric values
      const parsedSamples: { inputs: number[]; target: number; rawInputs: (string | number | null)[]; rawTarget: string | number | null }[] = [];
      const rows = dataset.preview.rows.slice(0, 15);

      for (const row of rows) {
        const rawTargetVal = targetColIdx !== -1 ? row[targetColIdx] : 0;
        let numTarget = Number(rawTargetVal);
        if (!Number.isFinite(numTarget)) {
          // If categorical target, map via hash
          numTarget = String(rawTargetVal).charCodeAt(0) % 2;
        } else {
          // Normalize to [0, 1] range loosely
          numTarget = Math.max(0, Math.min(1, numTarget > 10 ? numTarget / 100 : numTarget));
        }

        const inputs: number[] = [];
        const rawInputs: (string | number | null)[] = [];
        for (const fIdx of visibleIndices) {
          const val = row[fIdx];
          rawInputs.push(val);
          let numVal = Number(val);
          if (!Number.isFinite(numVal)) {
            numVal = 0.5;
          } else {
            // Min-max standard normalize
            numVal = Math.tanh(numVal / 10) * 0.5 + 0.5;
          }
          inputs.push(Number(numVal.toFixed(3)));
        }

        parsedSamples.push({
          inputs,
          target: Number(numTarget.toFixed(3)),
          rawInputs,
          rawTarget: rawTargetVal,
        });
      }

      if (parsedSamples.length > 0) {
        return {
          featureNames: visibleFeatures,
          targetName: activeTarget,
          samples: parsedSamples,
          isUsingDataset: true,
        };
      }
    }

    // Default teaching samples
    return {
      featureNames: TEACHING_FEATURE_NAMES,
      targetName: TEACHING_TARGET_NAME,
      samples: TEACHING_SAMPLES.map((s) => ({
        inputs: s.inputs,
        target: s.target,
        rawInputs: s.inputs,
        rawTarget: s.target,
      })),
      isUsingDataset: false,
    };
  }, [dataset, selectedTarget, selectedFeatures]);

  // Layer dimensions: [inputDim, hidden1, (hidden2), outputDim]
  const layerSizes = useMemo<number[]>(() => {
    const inputDim = Math.max(1, featureNames.length);
    if (hiddenLayerCount === 1) {
      return [inputDim, neuronsL1, 1];
    }
    return [inputDim, neuronsL1, neuronsL2, 1];
  }, [featureNames.length, hiddenLayerCount, neuronsL1, neuronsL2]);

  // Model parameters (weights & biases)
  // weights[layerIdx] is a 2D array [toNeuron][fromNeuron]
  const [weights, setWeights] = useState<number[][][]>(() => {
    return [
      Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => (Math.random() * 1.2 - 0.6))),
      Array.from({ length: 1 }, () => Array.from({ length: 4 }, () => (Math.random() * 1.2 - 0.6))),
    ];
  });

  const [biases, setBiases] = useState<number[][]>(() => {
    return [
      Array.from({ length: 4 }, () => (Math.random() * 0.4 - 0.2)),
      Array.from({ length: 1 }, () => (Math.random() * 0.4 - 0.2)),
    ];
  });

  // Re-seed weights when architecture changes
  const resetWeights = useCallback(() => {
    const newWeights: number[][][] = [];
    const newBiases: number[][] = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const fromDim = layerSizes[l];
      const toDim = layerSizes[l + 1];
      const std = Math.sqrt(2 / (fromDim + toDim)); // Xavier/Glorot init
      const layerW: number[][] = [];
      const layerB: number[] = [];
      for (let j = 0; j < toDim; j++) {
        const row: number[] = [];
        for (let i = 0; i < fromDim; i++) {
          row.push(Number(((Math.random() * 2 - 1) * std).toFixed(4)));
        }
        layerW.push(row);
        layerB.push(Number(((Math.random() * 0.2 - 0.1)).toFixed(4)));
      }
      newWeights.push(layerW);
      newBiases.push(layerB);
    }
    setWeights(newWeights);
    setBiases(newBiases);
    setEpoch(0);
    setLossHistory([]);
    setCurrentLoss(0.48);
    setCurrentMetric(0.52);
    setPhase("idle");
    setPulseT(0);
  }, [layerSizes]);

  // Initialize weights when layer dimensions change
  useEffect(() => {
    resetWeights();
  }, [resetWeights]);

  // Forward Pass computation on current sample
  const currentSample = samples[selectedSampleIdx % samples.length] || samples[0];

  const computation = useMemo(() => {
    const activations: number[][] = [];
    const zValues: number[][] = [];

    // Layer 0: Inputs
    const a0 = currentSample.inputs.slice(0, layerSizes[0]);
    activations.push(a0);
    zValues.push(a0);

    let currentA = a0;

    for (let l = 0; l < weights.length; l++) {
      const layerW = weights[l];
      const layerB = biases[l];
      const nextZ: number[] = [];
      const nextA: number[] = [];

      for (let j = 0; j < layerW.length; j++) {
        let sum = layerB ? (layerB[j] ?? 0) : 0;
        const wRow = layerW[j];
        for (let i = 0; i < currentA.length; i++) {
          sum += (wRow[i] ?? 0) * (currentA[i] ?? 0);
        }
        nextZ.push(sum);
        // Last layer can use sigmoid for probability/normalized output
        const isOutput = l === weights.length - 1;
        const fn = isOutput ? "sigmoid" : activationFn;
        nextA.push(activate(sum, fn));
      }

      zValues.push(nextZ);
      activations.push(nextA);
      currentA = nextA;
    }

    const predicted = currentA[0] ?? 0;
    const target = currentSample.target;
    const error = predicted - target;
    const sampleLoss = 0.5 * error * error;

    return {
      activations,
      zValues,
      predicted,
      target,
      error,
      sampleLoss,
    };
  }, [currentSample, weights, biases, layerSizes, activationFn]);

  // Backpropagation step & weight updates
  const stepTraining = useCallback(() => {
    const { activations, zValues, predicted, target } = computation;
    const numLayers = layerSizes.length;

    // Gradients for deltas
    const deltas: number[][] = [];
    for (let l = 0; l < numLayers; l++) {
      deltas.push([]);
    }

    // Output delta: delta_out = (predicted - target) * f'(z_out)
    const outZ = zValues[numLayers - 1][0] ?? 0;
    const outA = activations[numLayers - 1][0] ?? 0;
    const dOut = (predicted - target) * activateDerivative(outZ, outA, "sigmoid");
    deltas[numLayers - 1] = [dOut];

    // Hidden deltas backwards
    for (let l = numLayers - 2; l >= 1; l--) {
      const nextLayerDelta = deltas[l + 1];
      const nextWeights = weights[l]; // [toNext][fromCurrent]
      const currentZ = zValues[l];
      const currentA = activations[l];
      const layerD: number[] = [];

      for (let i = 0; i < layerSizes[l]; i++) {
        let errorSum = 0;
        for (let j = 0; j < nextLayerDelta.length; j++) {
          errorSum += (nextWeights[j]?.[i] ?? 0) * nextLayerDelta[j];
        }
        const deriv = activateDerivative(currentZ[i] ?? 0, currentA[i] ?? 0, activationFn);
        layerD.push(errorSum * deriv);
      }
      deltas[l] = layerD;
    }

    // Update weights and biases: W -= lr * delta * a_prev
    const updatedWeights: number[][][] = [];
    const updatedBiases: number[][] = [];

    for (let l = 0; l < weights.length; l++) {
      const layerW = weights[l];
      const layerB = biases[l];
      const layerDelta = deltas[l + 1];
      const prevA = activations[l];

      const newWRow: number[][] = [];
      const newBRow: number[] = [];

      for (let j = 0; j < layerW.length; j++) {
        const row: number[] = [];
        const dj = layerDelta[j] ?? 0;
        for (let i = 0; i < prevA.length; i++) {
          const grad = dj * (prevA[i] ?? 0);
          const oldW = layerW[j][i] ?? 0;
          const newW = oldW - learningRate * grad;
          row.push(Number(newW.toFixed(5)));
        }
        newWRow.push(row);
        const oldB = layerB[j] ?? 0;
        const newB = oldB - learningRate * dj;
        newBRow.push(Number(newB.toFixed(5)));
      }
      updatedWeights.push(newWRow);
      updatedBiases.push(newBRow);
    }

    setWeights(updatedWeights);
    setBiases(updatedBiases);

    // Update loss history
    const newLoss = Math.max(0.001, Number(computation.sampleLoss.toFixed(4)));
    setCurrentLoss(newLoss);
    setLossHistory((prev) => {
      const next = [...prev, newLoss];
      return next.length > 30 ? next.slice(next.length - 30) : next;
    });

    const accuracyEstimate = Math.max(0.1, Math.min(0.99, 1 - Math.abs(computation.error)));
    setCurrentMetric(Number(accuracyEstimate.toFixed(3)));
    setEpoch((prev) => prev + 1);

    // Advance sample row cyclically
    setSelectedSampleIdx((prev) => (prev + 1) % samples.length);
  }, [computation, layerSizes, weights, biases, activationFn, learningRate, samples.length]);

  // Animation loop when playing
  const animFrameRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number>(0);

  useEffect(() => {
    if (!isPlaying) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      setPhase("idle");
      setPulseT(0);
      return;
    }

    let localT = 0;
    const intervalMs = Math.max(80, 700 / speedMultiplier);

    const loop = (timestamp: number) => {
      if (!lastStepTimeRef.current) lastStepTimeRef.current = timestamp;
      const elapsed = timestamp - lastStepTimeRef.current;
      localT = (elapsed % intervalMs) / intervalMs;
      setPulseT(localT);

      // Determine phase based on sub-cycle
      if (localT < 0.4) {
        setPhase("forward");
      } else if (localT < 0.55) {
        setPhase("loss");
      } else if (localT < 0.85) {
        setPhase("backward");
      } else {
        setPhase("update");
      }

      if (elapsed >= intervalMs) {
        stepTraining();
        lastStepTimeRef.current = timestamp;
      }

      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, speedMultiplier, stepTraining]);

  // Layout positions for SVG
  const svgWidth = 760;
  const svgHeight = 360;
  const numLayers = layerSizes.length;
  const layerSpacing = (svgWidth - 140) / (numLayers - 1);

  // Precompute neuron coordinates: { layer, index, x, y, a, z, label }
  const neuronLayout = useMemo(() => {
    const layout: {
      layer: number;
      index: number;
      x: number;
      y: number;
      a: number;
      z: number;
      label: string;
      color: string;
    }[][] = [];

    for (let l = 0; l < numLayers; l++) {
      const count = layerSizes[l];
      const layerX = 70 + l * layerSpacing;
      const layerNodes: {
        layer: number;
        index: number;
        x: number;
        y: number;
        a: number;
        z: number;
        label: string;
        color: string;
      }[] = [];

      const totalH = (count - 1) * 58;
      const startY = (svgHeight - totalH) / 2;

      for (let j = 0; j < count; j++) {
        const y = count === 1 ? svgHeight / 2 : startY + j * 58;
        const aVal = computation.activations[l]?.[j] ?? 0;
        const zVal = computation.zValues[l]?.[j] ?? 0;

        let label = "";
        let color = "#5A1D2E"; // burgundy default

        if (l === 0) {
          label = featureNames[j] ? featureNames[j].slice(0, 10) : `X_${j + 1}`;
          color = "#1F3352"; // Oxford navy
        } else if (l === numLayers - 1) {
          label = targetName ? targetName.slice(0, 10) : "Target Ŷ";
          color = "#1F7A5A"; // state-good
        } else {
          label = `h_${l},${j + 1}`;
          color = "#8C6A1F"; // brass accent
        }

        layerNodes.push({
          layer: l,
          index: j,
          x: layerX,
          y,
          a: aVal,
          z: zVal,
          label,
          color,
        });
      }
      layout.push(layerNodes);
    }
    return layout;
  }, [layerSizes, numLayers, layerSpacing, svgHeight, computation.activations, computation.zValues, featureNames, targetName]);

  // Synaptic connections layout: from (x1, y1) to (x2, y2) with weight
  const synapseLayout = useMemo(() => {
    const connections: {
      fromLayer: number;
      fromIdx: number;
      toLayer: number;
      toIdx: number;
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      weight: number;
    }[] = [];

    for (let l = 0; l < weights.length; l++) {
      const fromNodes = neuronLayout[l];
      const toNodes = neuronLayout[l + 1];
      const layerW = weights[l];

      if (!fromNodes || !toNodes || !layerW) continue;

      for (let j = 0; j < toNodes.length; j++) {
        for (let i = 0; i < fromNodes.length; i++) {
          const w = layerW[j]?.[i] ?? 0;
          connections.push({
            fromLayer: l,
            fromIdx: i,
            toLayer: l + 1,
            toIdx: j,
            x1: fromNodes[i].x,
            y1: fromNodes[i].y,
            x2: toNodes[j].x,
            y2: toNodes[j].y,
            weight: w,
          });
        }
      }
    }
    return connections;
  }, [weights, neuronLayout]);

  // Selected item information for the inspector panel
  const inspectedNeuron = useMemo(() => {
    if (!selectedNode) return null;
    const { layer, index } = selectedNode;
    const node = neuronLayout[layer]?.[index];
    if (!node) return null;

    const layerName = layer === 0 ? "Input Layer" : layer === numLayers - 1 ? "Output Layer" : `Hidden Layer ${layer}`;
    const biasVal = layer > 0 ? (biases[layer - 1]?.[index] ?? 0) : null;
    const derivVal = layer > 0 ? activateDerivative(node.z, node.a, layer === numLayers - 1 ? "sigmoid" : activationFn) : null;

    // Incoming weights for this neuron
    const incoming = layer > 0 ? (weights[layer - 1]?.[index] ?? []) : [];
    const prevActivations = layer > 0 ? (computation.activations[layer - 1] ?? []) : [];

    return {
      layerName,
      node,
      biasVal,
      derivVal,
      incoming,
      prevActivations,
    };
  }, [selectedNode, neuronLayout, numLayers, biases, weights, computation.activations, activationFn]);

  const inspectedSynapse = useMemo(() => {
    if (!selectedSynapse) return null;
    const { fromLayer, fromIdx, toLayer, toIdx } = selectedSynapse;
    const w = weights[fromLayer]?.[toIdx]?.[fromIdx] ?? 0;
    const fromNode = neuronLayout[fromLayer]?.[fromIdx];
    const toNode = neuronLayout[toLayer]?.[toIdx];
    const aIn = computation.activations[fromLayer]?.[fromIdx] ?? 0;
    const aOut = computation.activations[toLayer]?.[toIdx] ?? 0;
    const approxGrad = (computation.error * aIn).toFixed(5);
    const approxDelta = (-learningRate * Number(approxGrad)).toFixed(5);

    return {
      fromLayer,
      fromIdx,
      toLayer,
      toIdx,
      w,
      fromLabel: fromNode?.label ?? `N_${fromIdx}`,
      toLabel: toNode?.label ?? `N_${toIdx}`,
      aIn,
      aOut,
      approxGrad,
      approxDelta,
    };
  }, [selectedSynapse, weights, neuronLayout, computation.activations, computation.error, learningRate]);

  return (
    <div className="space-y-6">
      {/* Top Banner: Dataset context & mode helper */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-control border border-rule bg-paper-sunken p-4">
        <div>
          <span className="field-label block mb-1">Architecture & Model</span>
          <h2 className="font-serif text-h3 text-ink">
            Animated Neural Network — Multi-Layer Perceptron (MLP)
          </h2>
          <p className="text-sm text-ink-muted mt-1">
            {isUsingDataset ? (
              <>
                Connected to active dataset: <span className="font-semibold text-ink">{dataset?.name}</span> ({featureNames.length} features fed into input neurons · Task: {requestedTask === "auto" ? "Auto" : requestedTask}).
              </>
            ) : (
              <>
                Using educational reference data ({samples.length} rows, {featureNames.length} features · Task: {requestedTask === "auto" ? "Classification/Regression" : requestedTask}). Upload a dataset in Preprocessing to train on your own data.
              </>
            )}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {onSwitchToWeightAnalysis && (
            <button
              type="button"
              onClick={onSwitchToWeightAnalysis}
              className="btn btn-secondary text-sm"
              title="Inspect step-by-step mathematical weight calculations"
            >
              Weight Analysis Mode →
            </button>
          )}
          {onSwitchToNormal && (
            <button
              type="button"
              onClick={onSwitchToNormal}
              className="btn btn-secondary text-sm"
            >
              Normal Training Form
            </button>
          )}
        </div>
      </div>

      {/* Control Strip & Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Playback Controls */}
        <div className="callout md:col-span-2 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="field-label">Simulation Controls</span>
              <span className={`text-xs px-2.5 py-1 rounded-control font-mono font-medium border ${
                phase === "forward" ? "border-secondary text-secondary bg-paper-sunken" :
                phase === "loss" ? "border-state-warn text-state-warn bg-paper-sunken" :
                phase === "backward" ? "border-primary text-primary bg-paper-sunken" :
                phase === "update" ? "border-state-good text-state-good bg-paper-sunken" :
                "border-rule text-ink-muted bg-paper-sunken"
              }`}>
                {phase === "forward" && "● Forward Signal Propagation"}
                {phase === "loss" && "● Loss Computation"}
                {phase === "backward" && "● Backward Gradient Flow"}
                {phase === "update" && "● Updating Synaptic Weights"}
                {phase === "idle" && (isPlaying ? "Running..." : "Paused / Ready")}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setIsPlaying((prev) => !prev)}
                className={`btn min-h-10 px-5 text-sm font-semibold transition-colors ${
                  isPlaying ? "bg-state-bad text-white border-state-bad" : "btn-primary"
                }`}
              >
                {isPlaying ? "⏸ Pause Training" : "▶ Start Animated Training"}
              </button>

              <button
                type="button"
                disabled={isPlaying}
                onClick={stepTraining}
                className="btn btn-secondary min-h-10 text-sm"
                title="Perform one single training iteration"
              >
                Step 1 Iteration →
              </button>

              <button
                type="button"
                onClick={resetWeights}
                className="btn btn-secondary min-h-10 text-sm"
                title="Reinitialize random weights and reset epoch count"
              >
                ↺ Reset Weights
              </button>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-rule flex flex-wrap items-center justify-between gap-2 text-xs text-ink-muted">
            <div className="flex items-center gap-2">
              <span>Speed:</span>
              {[0.5, 1, 2, 4].map((mult) => (
                <button
                  key={mult}
                  type="button"
                  onClick={() => setSpeedMultiplier(mult)}
                  className={`px-2 py-1 rounded-control font-mono ${
                    speedMultiplier === mult ? "bg-primary text-white" : "bg-paper-sunken text-ink hover:bg-rule"
                  }`}
                >
                  {mult}x
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <span>Learning Rate (η):</span>
              <input
                type="range"
                min="0.01"
                max="0.2"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-20 accent-primary"
              />
              <span className="font-mono text-ink font-semibold">{learningRate.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Real-time Loss & Epoch Readout */}
        <div className="callout flex flex-col justify-between">
          <div>
            <span className="field-label">Training Progress</span>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="font-serif text-h2 text-ink tabular">{epoch}</span>
              <span className="text-xs text-ink-muted">epochs passed</span>
            </div>
            <div className="mt-2 flex items-baseline justify-between border-t border-rule pt-2">
              <div>
                <span className="field-label block">MSE Loss (L)</span>
                <span className="font-serif text-h3 text-primary tabular font-mono">
                  {currentLoss.toFixed(4)}
                </span>
              </div>
              <div className="text-right">
                <span className="field-label block">Fit Accuracy</span>
                <span className="font-serif text-h3 text-state-good tabular font-mono">
                  {(currentMetric * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {/* Mini Loss Sparkline */}
          <div className="mt-3 h-10 w-full bg-paper-sunken relative overflow-hidden rounded-control border border-rule">
            {lossHistory.length > 1 ? (
              <svg className="w-full h-full" preserveAspectRatio="none" viewBox={`0 0 ${Math.max(1, lossHistory.length - 1)} 100`}>
                <polyline
                  fill="none"
                  stroke="#5A1D2E"
                  strokeWidth="2.5"
                  points={lossHistory
                    .map((val, idx) => {
                      const maxL = Math.max(...lossHistory, 0.5);
                      const minL = Math.min(...lossHistory, 0.01);
                      const normY = 90 - ((val - minL) / (maxL - minL || 1)) * 80;
                      return `${idx},${normY}`;
                    })
                    .join(" ")}
                />
              </svg>
            ) : (
              <div className="flex items-center justify-center h-full text-xs text-ink-muted">
                Run epochs to draw loss curve
              </div>
            )}
          </div>
        </div>

        {/* Hyperparameters & Topology Controls */}
        <div className="callout space-y-3">
          <span className="field-label">Topology Settings</span>
          <div>
            <label className="text-xs font-medium text-ink block mb-1">
              Hidden Layers: <span className="font-semibold">{hiddenLayerCount}</span>
            </label>
            <div className="grid grid-cols-2 gap-1">
              <button
                type="button"
                onClick={() => setHiddenLayerCount(1)}
                className={`text-xs py-1.5 px-2 rounded-control font-medium ${
                  hiddenLayerCount === 1 ? "bg-primary text-white" : "bg-paper-sunken text-ink"
                }`}
              >
                1 Layer
              </button>
              <button
                type="button"
                onClick={() => setHiddenLayerCount(2)}
                className={`text-xs py-1.5 px-2 rounded-control font-medium ${
                  hiddenLayerCount === 2 ? "bg-primary text-white" : "bg-paper-sunken text-ink"
                }`}
              >
                2 Layers
              </button>
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-ink block mb-1">
              Hidden 1 Neurons: <span className="font-mono font-semibold">{neuronsL1}</span>
            </label>
            <input
              type="range"
              min="2"
              max="6"
              value={neuronsL1}
              onChange={(e) => setNeuronsL1(Number(e.target.value))}
              className="w-full accent-primary"
            />
          </div>

          {hiddenLayerCount === 2 && (
            <div>
              <label className="text-xs font-medium text-ink block mb-1">
                Hidden 2 Neurons: <span className="font-mono font-semibold">{neuronsL2}</span>
              </label>
              <input
                type="range"
                min="2"
                max="6"
                value={neuronsL2}
                onChange={(e) => setNeuronsL2(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
          )}

          <div>
            <label className="text-xs font-medium text-ink block mb-1">Activation Function</label>
            <select
              value={activationFn}
              onChange={(e) => setActivationFn(e.target.value as ActivationFunction)}
              className="w-full text-xs p-1.5 bg-paper-sunken border border-rule rounded-control"
            >
              <option value="sigmoid">Sigmoid (σ)</option>
              <option value="relu">ReLU (max(0, z))</option>
              <option value="tanh">Tanh (hyperbolic tan)</option>
              <option value="linear">Linear (identity)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Neural Network Animation Canvas */}
      <div className="callout p-0 overflow-hidden relative border border-rule bg-paper-raised">
        <div className="p-3 bg-paper-sunken border-b border-rule flex flex-wrap items-center justify-between text-xs gap-2">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-ink">Interactive Neural Architecture Visualizer</span>
            <span className="text-ink-muted">· Click any node or synapse line to inspect</span>
          </div>

          <div className="flex items-center gap-4 text-ink-muted">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-1 bg-[#1F3352]" /> Positive Weight (W &gt; 0)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-1 bg-[#5A1D2E]" /> Negative Weight (W &lt; 0)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-accent" /> Active Signal
            </span>
          </div>
        </div>

        {/* SVG Network Canvas */}
        <div className="w-full overflow-x-auto p-4 flex justify-center bg-paper">
          <svg
            width={svgWidth}
            height={svgHeight}
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="select-none"
          >
            <defs>
              {/* Glow filter for active firing neurons */}
              <filter id="furina-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* Layer Column Headers */}
            {neuronLayout.map((layerNodes, l) => {
              const firstNode = layerNodes[0];
              if (!firstNode) return null;
              const headerText =
                l === 0
                  ? "Input Layer"
                  : l === numLayers - 1
                  ? "Output Layer"
                  : `Hidden Layer ${l}`;
              return (
                <text
                  key={`header-${l}`}
                  x={firstNode.x}
                  y={22}
                  textAnchor="middle"
                  className="font-sans text-caption uppercase tracking-wider font-semibold fill-[#5A544B]"
                >
                  {headerText}
                </text>
              );
            })}

            {/* Synaptic Connection Lines */}
            {synapseLayout.map((syn, idx) => {
              const isSelected =
                selectedSynapse?.fromLayer === syn.fromLayer &&
                selectedSynapse?.fromIdx === syn.fromIdx &&
                selectedSynapse?.toLayer === syn.toLayer &&
                selectedSynapse?.toIdx === syn.toIdx;

              const isPositive = syn.weight >= 0;
              const strokeColor = isSelected
                ? "#8C6A1F"
                : isPositive
                ? "#1F3352"
                : "#5A1D2E";

              const strokeWidth = Math.min(6, Math.max(1, Math.abs(syn.weight) * 2.5));
              const strokeOpacity = isSelected ? 1 : Math.min(0.85, Math.max(0.2, Math.abs(syn.weight)));

              return (
                <g key={`syn-${idx}`}>
                  <line
                    x1={syn.x1}
                    y1={syn.y1}
                    x2={syn.x2}
                    y2={syn.y2}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeOpacity={strokeOpacity}
                    strokeLinecap="round"
                    className="cursor-pointer transition-all duration-150 hover:stroke-accent"
                    onClick={() => {
                      setSelectedSynapse({
                        fromLayer: syn.fromLayer,
                        fromIdx: syn.fromIdx,
                        toLayer: syn.toLayer,
                        toIdx: syn.toIdx,
                      });
                      setSelectedNode(null);
                    }}
                  />

                  {/* Traveling Pulse Particle when training animation is active */}
                  {isPlaying && (phase === "forward" || phase === "backward") && (
                    <circle
                      cx={
                        phase === "forward"
                          ? syn.x1 + (syn.x2 - syn.x1) * pulseT
                          : syn.x2 - (syn.x2 - syn.x1) * pulseT
                      }
                      cy={
                        phase === "forward"
                          ? syn.y1 + (syn.y2 - syn.y1) * pulseT
                          : syn.y2 - (syn.y2 - syn.y1) * pulseT
                      }
                      r={Math.max(2, Math.min(4.5, Math.abs(syn.weight) * 2))}
                      fill={phase === "forward" ? "#8C6A1F" : "#5A1D2E"}
                      filter="url(#furina-glow)"
                      opacity={0.9}
                    />
                  )}
                </g>
              );
            })}

            {/* Neuron Nodes */}
            {neuronLayout.map((layerNodes) =>
              layerNodes.map((node) => {
                const isSelected =
                  selectedNode?.layer === node.layer && selectedNode?.index === node.index;
                const radius = 22;

                // Color fill according to activation value [0, 1]
                const clampedA = Math.max(0, Math.min(1, node.a));
                const isInput = node.layer === 0;
                const isOutput = node.layer === numLayers - 1;

                const fillColor = isSelected
                  ? "#F2F0EA"
                  : isInput
                  ? `rgba(31, 51, 82, ${0.15 + clampedA * 0.7})`
                  : isOutput
                  ? `rgba(31, 122, 90, ${0.15 + clampedA * 0.7})`
                  : `rgba(140, 106, 31, ${0.15 + clampedA * 0.7})`;

                const strokeColor = isSelected
                  ? "#5A1D2E"
                  : isInput
                  ? "#1F3352"
                  : isOutput
                  ? "#1F7A5A"
                  : "#8C6A1F";

                return (
                  <g
                    key={`node-${node.layer}-${node.index}`}
                    className="cursor-pointer group"
                    onClick={() => {
                      setSelectedNode({ layer: node.layer, index: node.index });
                      setSelectedSynapse(null);
                    }}
                  >
                    {/* Pulsing ring during forward pass */}
                    {isPlaying && phase === "forward" && (
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={radius + 4}
                        fill="none"
                        stroke="#8C6A1F"
                        strokeWidth="1.5"
                        strokeDasharray="3 3"
                        opacity={clampedA}
                      />
                    )}

                    {/* Main Node Body */}
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={radius}
                      fill={fillColor}
                      stroke={strokeColor}
                      strokeWidth={isSelected ? 3 : 2}
                      className="transition-colors duration-150 group-hover:stroke-primary"
                    />

                    {/* Activation Value readout inside circle */}
                    <text
                      x={node.x}
                      y={node.y + 4}
                      textAnchor="middle"
                      className="font-mono text-xs font-semibold fill-[#1A1815]"
                    >
                      {node.a.toFixed(2)}
                    </text>

                    {/* Label below / next to node */}
                    <text
                      x={node.x}
                      y={node.y + radius + 14}
                      textAnchor="middle"
                      className="font-sans text-[11px] font-medium fill-[#1A1815]"
                    >
                      {node.label}
                    </text>
                  </g>
                );
              })
            )}
          </svg>
        </div>
      </div>

      {/* Bottom Section: Active Sample Selector + Detailed Inspector Panel */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Sample Input Row Feeder */}
        <div className="callout md:col-span-1 space-y-4">
          <div className="flex items-center justify-between">
            <span className="field-label">Dataset Sample Feeder</span>
            <span className="text-xs text-ink-muted">
              Sample #{selectedSampleIdx + 1} of {samples.length}
            </span>
          </div>

          <p className="text-xs text-ink-muted">
            The input values currently lighting up the network:
          </p>

          <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {featureNames.map((feat, idx) => {
              const val = currentSample.inputs[idx] ?? 0;
              const raw = currentSample.rawInputs?.[idx];
              return (
                <div key={feat} className="flex items-center justify-between p-2 rounded-control bg-paper-sunken text-xs">
                  <div className="truncate max-w-[140px]" title={feat}>
                    <span className="font-medium text-ink">{feat}</span>
                    {raw !== undefined && raw !== null && (
                      <span className="block text-[10px] text-ink-muted">raw: {String(raw)}</span>
                    )}
                  </div>
                  <div className="text-right">
                    <span className="font-mono font-semibold text-secondary">{val.toFixed(2)}</span>
                    <span className="block text-[10px] text-ink-muted">norm (0-1)</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="border-t border-rule pt-3 flex items-center justify-between text-xs">
            <div>
              <span className="font-semibold text-ink block">{targetName}</span>
              <span className="text-ink-muted">Ground Truth Y:</span>
            </div>
            <div className="text-right">
              <span className="font-mono font-bold text-state-good">{currentSample.target}</span>
              <span className="block text-[10px] text-ink-muted">
                pred: {computation.predicted.toFixed(3)}
              </span>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              disabled={selectedSampleIdx === 0}
              onClick={() => setSelectedSampleIdx((prev) => Math.max(0, prev - 1))}
              className="btn btn-secondary text-xs flex-1 min-h-8 py-1"
            >
              ← Prev Sample
            </button>
            <button
              type="button"
              disabled={selectedSampleIdx >= samples.length - 1}
              onClick={() => setSelectedSampleIdx((prev) => Math.min(samples.length - 1, prev + 1))}
              className="btn btn-secondary text-xs flex-1 min-h-8 py-1"
            >
              Next Sample →
            </button>
          </div>
        </div>

        {/* Detailed Inspector Panel for Clicked Node or Synapse */}
        <div className="callout md:col-span-2">
          {inspectedNeuron && (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-rule pb-2">
                <div>
                  <span className="field-label">{inspectedNeuron.layerName}</span>
                  <h3 className="font-serif text-h3 text-ink">
                    Neuron {inspectedNeuron.node.label}
                  </h3>
                </div>
                <div className="text-right">
                  <span className="field-label block">Activation a = f(z)</span>
                  <span className="font-serif text-h2 text-primary font-mono tabular">
                    {inspectedNeuron.node.a.toFixed(4)}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Pre-activation (z)</span>
                  <span className="font-mono font-semibold text-ink text-sm">
                    {inspectedNeuron.node.z.toFixed(4)}
                  </span>
                </div>
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Bias (b)</span>
                  <span className="font-mono font-semibold text-ink text-sm">
                    {inspectedNeuron.biasVal !== null ? inspectedNeuron.biasVal.toFixed(4) : "—"}
                  </span>
                </div>
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Derivative f&apos;(z)</span>
                  <span className="font-mono font-semibold text-ink text-sm">
                    {inspectedNeuron.derivVal !== null ? inspectedNeuron.derivVal.toFixed(4) : "—"}
                  </span>
                </div>
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Activation Type</span>
                  <span className="font-mono font-semibold text-ink text-sm uppercase">
                    {inspectedNeuron.node.layer === 0 ? "Raw Input" : inspectedNeuron.node.layer === numLayers - 1 ? "Sigmoid" : activationFn}
                  </span>
                </div>
              </div>

              {/* Mathematical formula breakdown */}
              {inspectedNeuron.node.layer > 0 && (
                <div className="p-3 bg-paper-sunken rounded-control border border-rule text-xs space-y-2">
                  <p className="font-mono font-medium text-ink">
                    z = ∑(wᵢ · xᵢ) + b ={" "}
                    {inspectedNeuron.incoming.map((w, i) => {
                      const prevA = inspectedNeuron.prevActivations[i] ?? 0;
                      return `(${w.toFixed(2)} × ${prevA.toFixed(2)})`;
                    }).join(" + ")}{" "}
                    + ({inspectedNeuron.biasVal?.toFixed(2)}) ={" "}
                    <span className="font-bold text-primary">{inspectedNeuron.node.z.toFixed(3)}</span>
                  </p>
                  <p className="font-mono text-ink-muted">
                    a = {activationFn}({inspectedNeuron.node.z.toFixed(3)}) ={" "}
                    <span className="font-bold text-state-good">{inspectedNeuron.node.a.toFixed(4)}</span>
                  </p>
                </div>
              )}
            </div>
          )}

          {inspectedSynapse && (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-rule pb-2">
                <div>
                  <span className="field-label">Synaptic Connection</span>
                  <h3 className="font-serif text-h3 text-ink">
                    Weight {inspectedSynapse.fromLabel} → {inspectedSynapse.toLabel}
                  </h3>
                </div>
                <div className="text-right">
                  <span className="field-label block">Weight Value (W)</span>
                  <span className={`font-serif text-h2 font-mono tabular ${inspectedSynapse.w >= 0 ? "text-secondary" : "text-primary"}`}>
                    {inspectedSynapse.w.toFixed(4)}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Input Activation (a_in)</span>
                  <span className="font-mono font-semibold text-ink text-sm">
                    {inspectedSynapse.aIn.toFixed(4)}
                  </span>
                </div>
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Output Activation (a_out)</span>
                  <span className="font-mono font-semibold text-ink text-sm">
                    {inspectedSynapse.aOut.toFixed(4)}
                  </span>
                </div>
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Gradient (∂L/∂w)</span>
                  <span className="font-mono font-semibold text-ink text-sm">
                    {inspectedSynapse.approxGrad}
                  </span>
                </div>
                <div className="p-2.5 rounded-control bg-paper-sunken">
                  <span className="text-ink-muted block">Weight Step (Δw)</span>
                  <span className="font-mono font-semibold text-ink text-sm">
                    {inspectedSynapse.approxDelta}
                  </span>
                </div>
              </div>

              <div className="p-3 bg-paper-sunken rounded-control border border-rule text-xs">
                <p className="font-semibold text-ink mb-1">Weight Update Formula:</p>
                <p className="font-mono text-ink">
                  W_new = W_old - η · (∂L/∂w) = {inspectedSynapse.w.toFixed(4)} - ({learningRate} × {inspectedSynapse.approxGrad}) ={" "}
                  <span className="font-bold text-state-good">
                    {(inspectedSynapse.w + Number(inspectedSynapse.approxDelta)).toFixed(4)}
                  </span>
                </p>
              </div>
            </div>
          )}

          {!inspectedNeuron && !inspectedSynapse && (
            <div className="flex items-center justify-center h-48 text-ink-muted text-sm text-center">
              Click any neuron circle or connection line above to inspect its math, weights, and gradients.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
