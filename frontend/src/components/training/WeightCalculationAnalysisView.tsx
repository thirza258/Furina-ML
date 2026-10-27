import { useMemo, useState } from "react";
import type { DatasetDetail, RequestedTaskType } from "../../interface/MlTypes";

interface WeightCalculationAnalysisViewProps {
  dataset: DatasetDetail | null;
  selectedTarget?: string;
  selectedFeatures?: string[];
  requestedTask?: RequestedTaskType;
  onSwitchToAnimated?: () => void;
  onSwitchToNormal?: () => void;
}

type ActivationFunction = "sigmoid" | "relu" | "tanh" | "linear";

const DEFAULT_INPUTS = [
  { name: "Feature X₁ (Income / Scaled)", value: 0.72 },
  { name: "Feature X₂ (Age / Scaled)", value: 0.45 },
  { name: "Feature X₃ (Credit Score / Scaled)", value: 0.88 },
  { name: "Feature X₄ (Debt Ratio / Scaled)", value: 0.31 },
];

const DEFAULT_WEIGHTS = [0.65, -0.42, 0.78, -0.55];

export default function WeightCalculationAnalysisView({
  dataset,
  selectedTarget,
  selectedFeatures,
  requestedTask = "auto",
  onSwitchToAnimated,
  onSwitchToNormal,
}: WeightCalculationAnalysisViewProps) {
  // Extract features and values from dataset if available
  const { featureList, targetName, initialTargetY, isUsingDataset } = useMemo(() => {
    if (dataset && dataset.columns && dataset.preview && dataset.preview.rows.length > 0) {
      const activeTarget =
        selectedTarget ||
        (dataset.columns.find((c) => c.suitable_as_target)?.name ??
          dataset.columns[dataset.columns.length - 1].name);

      const activeFeatures =
        selectedFeatures && selectedFeatures.length > 0
          ? selectedFeatures.filter((f) => f !== activeTarget)
          : dataset.columns.filter((c) => c.name !== activeTarget).map((c) => c.name);

      const visibleFeatures = activeFeatures.slice(0, 4);
      const targetColIdx = dataset.preview.headers.indexOf(activeTarget);
      const firstRow = dataset.preview.rows[0];

      let initialTarget = 1.0;
      if (firstRow && targetColIdx !== -1) {
        const rawTarget = Number(firstRow[targetColIdx]);
        if (Number.isFinite(rawTarget)) {
          initialTarget = Math.max(0, Math.min(1, rawTarget > 10 ? rawTarget / 100 : rawTarget));
        }
      }

      const list = visibleFeatures.map((f) => {
        const idx = dataset.preview.headers.indexOf(f);
        const rawVal = firstRow && idx !== -1 ? firstRow[idx] : null;
        let numVal = Number(rawVal);
        if (!Number.isFinite(numVal)) {
          numVal = 0.5;
        } else {
          numVal = Number((Math.tanh(numVal / 10) * 0.5 + 0.5).toFixed(2));
        }
        return {
          name: f,
          value: numVal,
        };
      });

      return {
        featureList: list.length > 0 ? list : DEFAULT_INPUTS,
        targetName: activeTarget,
        initialTargetY: initialTarget,
        isUsingDataset: true,
      };
    }

    return {
      featureList: DEFAULT_INPUTS,
      targetName: "Target Label Y",
      initialTargetY: 1.0,
      isUsingDataset: false,
    };
  }, [dataset, selectedTarget, selectedFeatures]);

  // Interactive weights, inputs, bias, and parameters
  const [weights, setWeights] = useState<number[]>(() =>
    featureList.map((_, i) => DEFAULT_WEIGHTS[i % DEFAULT_WEIGHTS.length] ?? 0.5)
  );

  const [inputs, setInputs] = useState<number[]>(() =>
    featureList.map((f) => f.value)
  );

  const [bias, setBias] = useState<number>(0.15);
  const [learningRate, setLearningRate] = useState<number>(0.05);
  const [targetY, setTargetY] = useState<number>(initialTargetY);
  const [activationFn, setActivationFn] = useState<ActivationFunction>("sigmoid");

  // Additional hidden neurons for matrix view
  const [matrixWeights, setMatrixWeights] = useState<number[][]>([
    [0.65, -0.42, 0.78, -0.55],
    [0.32, 0.81, -0.25, 0.44],
    [-0.58, 0.19, 0.62, -0.33],
  ]);

  // Keep weights aligned if featureList length changes
  useMemo(() => {
    if (weights.length !== featureList.length) {
      setWeights(featureList.map((_, i) => DEFAULT_WEIGHTS[i % DEFAULT_WEIGHTS.length] ?? 0.5));
      setInputs(featureList.map((f) => f.value));
    }
  }, [featureList, weights.length]);

  // Forward Pass Calculations
  const partialProducts = useMemo(() => {
    return inputs.map((x, i) => {
      const w = weights[i] ?? 0;
      return Number((x * w).toFixed(4));
    });
  }, [inputs, weights]);

  const sumOfProducts = useMemo(() => {
    return partialProducts.reduce((sum, p) => sum + p, 0);
  }, [partialProducts]);

  const preActivationZ = useMemo(() => {
    return Number((sumOfProducts + bias).toFixed(4));
  }, [sumOfProducts, bias]);

  // Activation & Derivative
  const { activationA, derivativeFPrime, activationFormula } = useMemo(() => {
    const z = preActivationZ;
    let a = 0;
    let fPrime = 0;
    let formula = "";

    switch (activationFn) {
      case "sigmoid":
        a = 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));
        fPrime = a * (1 - a);
        formula = "σ(z) = 1 / (1 + e^(-z))";
        break;
      case "relu":
        a = Math.max(0, z);
        fPrime = z > 0 ? 1 : 0;
        formula = "ReLU(z) = max(0, z)";
        break;
      case "tanh":
        a = Math.tanh(z);
        fPrime = 1 - a * a;
        formula = "tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))";
        break;
      case "linear":
        a = z;
        fPrime = 1;
        formula = "f(z) = z (Identity)";
        break;
    }

    return {
      activationA: Number(a.toFixed(4)),
      derivativeFPrime: Number(fPrime.toFixed(4)),
      activationFormula: formula,
    };
  }, [preActivationZ, activationFn]);

  // Loss & Gradient Sensitivity
  const { errorDiff, lossMSE, deltaError } = useMemo(() => {
    const error = Number((activationA - targetY).toFixed(4));
    const loss = Number((0.5 * error * error).toFixed(5));
    // Chain rule: delta = ∂L/∂z = (a - y) * f'(z)
    const delta = Number((error * derivativeFPrime).toFixed(5));
    return {
      errorDiff: error,
      lossMSE: loss,
      deltaError: delta,
    };
  }, [activationA, targetY, derivativeFPrime]);

  // Gradients for each individual weight & update steps
  const weightGradients = useMemo(() => {
    return weights.map((w, i) => {
      const x = inputs[i] ?? 0;
      // ∂L/∂w_i = delta * x_i
      const grad = Number((deltaError * x).toFixed(5));
      // Δw_i = -η * ∂L/∂w_i
      const deltaW = Number((-learningRate * grad).toFixed(5));
      const newW = Number((w + deltaW).toFixed(5));
      const pctChange = w !== 0 ? Number(((deltaW / Math.abs(w)) * 100).toFixed(2)) : 0;

      return {
        feature: featureList[i]?.name ?? `Input X_${i + 1}`,
        x,
        oldW: w,
        grad,
        deltaW,
        newW,
        pctChange,
      };
    });
  }, [weights, inputs, deltaError, learningRate, featureList]);

  // Bias gradient & update
  const biasUpdate = useMemo(() => {
    // ∂L/∂b = delta * 1
    const gradB = deltaError;
    const deltaB = Number((-learningRate * gradB).toFixed(5));
    const newB = Number((bias + deltaB).toFixed(5));
    return {
      oldB: bias,
      gradB,
      deltaB,
      newB,
    };
  }, [bias, deltaError, learningRate]);

  // Weight Statistics & Diagnostics across all weights
  const diagnostics = useMemo(() => {
    const allW = [...weights, ...matrixWeights.flat()];
    const count = allW.length;
    const mean = allW.reduce((sum, v) => sum + v, 0) / count;
    const variance = allW.reduce((sum, v) => sum + (v - mean) ** 2, 0) / count;
    const std = Math.sqrt(variance);
    const minW = Math.min(...allW);
    const maxW = Math.max(...allW);
    const l1Norm = allW.reduce((sum, v) => sum + Math.abs(v), 0);
    const l2Norm = Math.sqrt(allW.reduce((sum, v) => sum + v * v, 0));
    const sparseCount = allW.filter((v) => Math.abs(v) < 0.05).length;
    const sparsityPct = (sparseCount / count) * 100;

    // Gradient norm ||∇W||
    const allGrads = weightGradients.map((g) => g.grad);
    const gradNorm = Math.sqrt(allGrads.reduce((sum, g) => sum + g * g, 0));

    let healthStatus: "healthy" | "vanishing" | "exploding" = "healthy";
    let healthMessage = "Gradient flow is in a balanced, healthy range.";

    if (gradNorm < 0.0005) {
      healthStatus = "vanishing";
      healthMessage = "Vanishing gradient detected: updates are near zero. Weights will change very slowly.";
    } else if (gradNorm > 5.0) {
      healthStatus = "exploding";
      healthMessage = "Exploding gradient detected: updates are very large, risking numerical instability.";
    }

    return {
      mean: Number(mean.toFixed(4)),
      std: Number(std.toFixed(4)),
      minW: Number(minW.toFixed(4)),
      maxW: Number(maxW.toFixed(4)),
      l1Norm: Number(l1Norm.toFixed(4)),
      l2Norm: Number(l2Norm.toFixed(4)),
      sparsityPct: Number(sparsityPct.toFixed(1)),
      gradNorm: Number(gradNorm.toFixed(5)),
      healthStatus,
      healthMessage,
    };
  }, [weights, matrixWeights, weightGradients]);

  // Handler to apply the calculated weight updates
  const applyUpdates = () => {
    setWeights(weightGradients.map((g) => g.newW));
    setBias(biasUpdate.newB);
  };

  // Reset to default weights
  const resetWeights = () => {
    setWeights(featureList.map((_, i) => DEFAULT_WEIGHTS[i % DEFAULT_WEIGHTS.length] ?? 0.5));
    setBias(0.15);
    setMatrixWeights([
      [0.65, -0.42, 0.78, -0.55],
      [0.32, 0.81, -0.25, 0.44],
      [-0.58, 0.19, 0.62, -0.33],
    ]);
  };

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-control border border-rule bg-paper-sunken p-4">
        <div>
          <span className="field-label block mb-1">Mathematical Deep Dive</span>
          <h2 className="font-serif text-h3 text-ink">
            Weight Calculation &amp; Update Analysis
          </h2>
          <p className="text-sm text-ink-muted mt-1">
            Step-by-step breakdown of dot products, activations, loss sensitivity (∂L/∂z), and gradient updates (Δw = -η·∂L/∂w) · Task: {requestedTask === "auto" ? "Classification/Regression" : requestedTask}.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {onSwitchToAnimated && (
            <button
              type="button"
              onClick={onSwitchToAnimated}
              className="btn btn-secondary text-sm"
            >
              ← Animated Neural Network
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

      {/* Diagnostics & Health Summary Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <div className="callout p-3">
          <span className="field-label">Mean Weight (μ)</span>
          <p className="font-serif text-h3 text-ink tabular font-mono mt-1">{diagnostics.mean}</p>
          <span className="text-xs text-ink-muted">center of distribution</span>
        </div>

        <div className="callout p-3">
          <span className="field-label">Std Dev (σ)</span>
          <p className="font-serif text-h3 text-ink tabular font-mono mt-1">{diagnostics.std}</p>
          <span className="text-xs text-ink-muted">spread of weights</span>
        </div>

        <div className="callout p-3">
          <span className="field-label">Weight Range</span>
          <p className="font-serif text-h3 text-ink tabular font-mono mt-1">
            [{diagnostics.minW}, {diagnostics.maxW}]
          </p>
          <span className="text-xs text-ink-muted">[min, max] bounds</span>
        </div>

        <div className="callout p-3">
          <span className="field-label">L2 Regularization Norm</span>
          <p className="font-serif text-h3 text-secondary tabular font-mono mt-1">{diagnostics.l2Norm}</p>
          <span className="text-xs text-ink-muted">||W||₂ weight decay</span>
        </div>

        <div className="callout p-3">
          <span className="field-label">Sparsity</span>
          <p className="font-serif text-h3 text-ink tabular font-mono mt-1">{diagnostics.sparsityPct}%</p>
          <span className="text-xs text-ink-muted">weights near zero</span>
        </div>

        <div className="callout p-3">
          <span className="field-label">Gradient Norm ||∇W||</span>
          <p className={`font-serif text-h3 tabular font-mono mt-1 ${
            diagnostics.healthStatus === "healthy" ? "text-state-good" :
            diagnostics.healthStatus === "vanishing" ? "text-state-warn" : "text-state-bad"
          }`}>
            {diagnostics.gradNorm}
          </p>
          <span className={`text-[11px] font-semibold block ${
            diagnostics.healthStatus === "healthy" ? "text-state-good" :
            diagnostics.healthStatus === "vanishing" ? "text-state-warn" : "text-state-bad"
          }`}>
            {diagnostics.healthStatus === "healthy" ? "✓ Balanced" :
             diagnostics.healthStatus === "vanishing" ? "⚠ Vanishing" : "✕ Exploding"}
          </span>
        </div>
      </div>

      {/* Main 4-Step Analytical Walkthrough */}
      <div className="space-y-6">
        {/* Step 1: Linear Combination / Dot Product */}
        <div className="callout space-y-4">
          <div className="flex flex-wrap items-center justify-between border-b border-rule pb-3 gap-2">
            <div>
              <span className="field-label">Step 1 — Linear Combination</span>
              <h3 className="font-serif text-h3 text-ink">
                Dot Product &amp; Weighted Sum: z = ∑(wᵢ · xᵢ) + b = wᵀx + b
              </h3>
            </div>
            <div className="text-right">
              <span className="field-label block">Pre-activation (z)</span>
              <span className="font-serif text-h2 font-mono text-primary tabular">
                {preActivationZ}
              </span>
            </div>
          </div>

          <p className="text-sm text-ink-muted">
            Each input feature xᵢ is multiplied by its corresponding synaptic connection weight wᵢ.
            The partial products are summed together with the neuron bias b to produce the raw net input z.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-rule text-caption uppercase text-ink-muted font-sans">
                  <th scope="col" className="p-3">Input Feature (xᵢ)</th>
                  <th scope="col" className="p-3">Input Value (xᵢ)</th>
                  <th scope="col" className="p-3">Weight (wᵢ)</th>
                  <th scope="col" className="p-3">Product (wᵢ · xᵢ)</th>
                  <th scope="col" className="p-3">Share of Net Sum</th>
                  <th scope="col" className="p-3">Adjust Weight</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-rule">
                {featureList.map((feat, i) => {
                  const x = inputs[i] ?? 0;
                  const w = weights[i] ?? 0;
                  const prod = partialProducts[i] ?? 0;
                  const totalAbs = partialProducts.reduce((s, p) => s + Math.abs(p), 0) || 1;
                  const share = ((Math.abs(prod) / totalAbs) * 100).toFixed(1);

                  return (
                    <tr key={feat.name} className="hover:bg-paper-sunken">
                      <td className="p-3 font-medium text-ink">
                        {feat.name}
                        {isUsingDataset && (
                          <span className="block text-[11px] text-ink-muted font-normal">
                            from dataset column
                          </span>
                        )}
                      </td>
                      <td className="p-3 font-mono tabular">
                        <input
                          type="number"
                          step="0.05"
                          value={x}
                          onChange={(e) => {
                            const val = Number(e.target.value);
                            setInputs((prev) => {
                              const next = [...prev];
                              next[i] = Number.isFinite(val) ? val : 0;
                              return next;
                            });
                          }}
                          className="w-20 p-1 border border-rule rounded-control bg-paper text-ink font-mono"
                        />
                      </td>
                      <td className="p-3 font-mono font-semibold tabular">
                        <span className={w >= 0 ? "text-secondary" : "text-primary"}>
                          {w.toFixed(3)}
                        </span>
                      </td>
                      <td className="p-3 font-mono tabular font-semibold text-ink">
                        {prod.toFixed(4)}
                      </td>
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-paper-sunken rounded-control overflow-hidden">
                            <div
                              className="h-full bg-accent"
                              style={{ width: `${Math.min(100, Number(share))}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs tabular text-ink-muted">{share}%</span>
                        </div>
                      </td>
                      <td className="p-3">
                        <input
                          type="range"
                          min="-1.5"
                          max="1.5"
                          step="0.02"
                          value={w}
                          onChange={(e) => {
                            const val = Number(e.target.value);
                            setWeights((prev) => {
                              const next = [...prev];
                              next[i] = val;
                              return next;
                            });
                          }}
                          className="w-28 accent-primary"
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr className="border-t-2 border-rule font-semibold bg-paper-sunken">
                  <td className="p-3 text-ink">Bias (b)</td>
                  <td className="p-3 text-ink-muted font-mono font-normal">constant = 1.0</td>
                  <td className="p-3">
                    <input
                      type="number"
                      step="0.05"
                      value={bias}
                      onChange={(e) => setBias(Number(e.target.value) || 0)}
                      className="w-20 p-1 border border-rule rounded-control bg-paper text-ink font-mono"
                    />
                  </td>
                  <td className="p-3 font-mono text-primary font-bold">{bias.toFixed(4)}</td>
                  <td colSpan={2} className="p-3 text-xs text-ink-muted">
                    Total Pre-activation z = {sumOfProducts.toFixed(4)} + {bias.toFixed(4)} ={" "}
                    <span className="font-bold text-ink">{preActivationZ}</span>
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>

        {/* Step 2: Activation Function */}
        <div className="callout space-y-4">
          <div className="flex flex-wrap items-center justify-between border-b border-rule pb-3 gap-2">
            <div>
              <span className="field-label">Step 2 — Activation Transformation</span>
              <h3 className="font-serif text-h3 text-ink">
                Non-Linear Activation: a = f(z)
              </h3>
            </div>
            <div className="text-right">
              <span className="field-label block">Activated Output (a)</span>
              <span className="font-serif text-h2 font-mono text-state-good tabular">
                {activationA}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-3">
              <label className="text-xs font-semibold text-ink block">
                Choose Activation Function f(z):
              </label>
              <div className="grid grid-cols-2 gap-2">
                {(["sigmoid", "relu", "tanh", "linear"] as ActivationFunction[]).map((fn) => (
                  <button
                    key={fn}
                    type="button"
                    onClick={() => setActivationFn(fn)}
                    className={`py-2 px-3 text-xs font-semibold rounded-control transition-colors ${
                      activationFn === fn
                        ? "bg-primary text-white"
                        : "bg-paper-sunken text-ink hover:bg-rule"
                    }`}
                  >
                    {fn.toUpperCase()}
                  </button>
                ))}
              </div>

              <div className="p-3 bg-paper-sunken rounded-control border border-rule text-xs space-y-1.5">
                <span className="field-label block">Formula &amp; Derivative</span>
                <p className="font-mono text-ink font-semibold">{activationFormula}</p>
                <p className="font-mono text-ink-muted">
                  Derivative: f&apos;(z) ={" "}
                  <span className="font-bold text-secondary">{derivativeFPrime}</span>
                </p>
              </div>
            </div>

            {/* Interactive Transfer Curve Visualization SVG */}
            <div className="md:col-span-2 p-3 bg-paper-raised rounded-control border border-rule flex flex-col justify-between">
              <div className="flex items-center justify-between text-xs text-ink-muted mb-2">
                <span>Activation Curve Transfer Map: z ∈ [-4, +4]</span>
                <span className="font-mono text-ink font-semibold">
                  z = {preActivationZ} → a = {activationA}
                </span>
              </div>

              <div className="w-full h-36 relative">
                <svg className="w-full h-full" viewBox="0 0 320 120" preserveAspectRatio="none">
                  {/* Grid Lines */}
                  <line x1="0" y1="60" x2="320" y2="60" stroke="#D8D3C7" strokeWidth="1" strokeDasharray="3 3" />
                  <line x1="160" y1="0" x2="160" y2="120" stroke="#D8D3C7" strokeWidth="1" strokeDasharray="3 3" />

                  {/* Activation Curve */}
                  <path
                    d={(() => {
                      const points: string[] = [];
                      for (let px = 0; px <= 320; px += 4) {
                        const localZ = -4 + (px / 320) * 8;
                        let localA = 0;
                        if (activationFn === "sigmoid") localA = 1 / (1 + Math.exp(-localZ));
                        else if (activationFn === "relu") localA = Math.max(0, localZ) / 4;
                        else if (activationFn === "tanh") localA = (Math.tanh(localZ) + 1) / 2;
                        else localA = (localZ + 4) / 8;

                        const py = 110 - Math.max(0, Math.min(1, localA)) * 100;
                        points.push(`${px},${py}`);
                      }
                      return `M ${points.join(" L ")}`;
                    })()}
                    fill="none"
                    stroke="#1F3352"
                    strokeWidth="2.5"
                  />

                  {/* Current Operating Point Dot */}
                  {(() => {
                    const normX = Math.max(0, Math.min(320, 160 + (preActivationZ / 4) * 160));
                    let currentNormA = activationA;
                    if (activationFn === "relu") currentNormA = activationA / 4;
                    if (activationFn === "tanh") currentNormA = (activationA + 1) / 2;
                    if (activationFn === "linear") currentNormA = (activationA + 4) / 8;
                    const normY = 110 - Math.max(0, Math.min(1, currentNormA)) * 100;

                    return (
                      <g>
                        <circle cx={normX} cy={normY} r="6" fill="#8C6A1F" stroke="#FFFFFF" strokeWidth="2" />
                        <text x={Math.min(270, Math.max(40, normX))} y={Math.max(16, normY - 10)} textAnchor="middle" className="font-mono text-[10px] font-bold fill-[#1A1815]">
                          ({preActivationZ}, {activationA})
                        </text>
                      </g>
                    );
                  })()}
                </svg>
              </div>

              <div className="flex justify-between text-[10px] text-ink-muted mt-1">
                <span>z = -4.0 (Inhibition)</span>
                <span>z = 0.0</span>
                <span>z = +4.0 (Saturation / Firing)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Step 3: Loss & Error Sensitivity */}
        <div className="callout space-y-4">
          <div className="flex flex-wrap items-center justify-between border-b border-rule pb-3 gap-2">
            <div>
              <span className="field-label">Step 3 — Loss &amp; Error Sensitivity</span>
              <h3 className="font-serif text-h3 text-ink">
                Error Gradient: δ = ∂L/∂z = (a - y) · f&apos;(z)
              </h3>
            </div>
            <div className="text-right">
              <span className="field-label block">MSE Loss L = 1/2 (a - y)²</span>
              <span className="font-serif text-h2 font-mono text-primary tabular">
                {lossMSE}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 bg-paper-sunken rounded-control border border-rule space-y-2 text-xs">
              <span className="field-label block">Target Value (y)</span>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={targetY}
                  onChange={(e) => setTargetY(Number(e.target.value))}
                  className="w-full accent-primary"
                />
                <span className="font-mono text-sm font-bold text-ink">{targetY.toFixed(2)}</span>
              </div>
              <p className="text-ink-muted">
                Target column: <span className="font-semibold text-ink">{targetName}</span>
              </p>
            </div>

            <div className="p-3 bg-paper-sunken rounded-control border border-rule space-y-2 text-xs">
              <span className="field-label block">Residual Difference (a - y)</span>
              <p className="font-mono text-base font-semibold text-ink">
                {activationA} - {targetY.toFixed(2)} ={" "}
                <span className={errorDiff >= 0 ? "text-primary" : "text-secondary"}>
                  {errorDiff >= 0 ? `+${errorDiff}` : errorDiff}
                </span>
              </p>
              <p className="text-ink-muted">
                {errorDiff > 0 ? "Neuron is over-predicting the target." : "Neuron is under-predicting the target."}
              </p>
            </div>

            <div className="p-3 bg-paper-sunken rounded-control border border-rule space-y-2 text-xs">
              <span className="field-label block">Sensitivity Error Term (δ)</span>
              <p className="font-mono text-base font-bold text-secondary">
                δ = {errorDiff} × {derivativeFPrime} = {deltaError}
              </p>
              <p className="text-ink-muted">
                Scales how strongly each incoming weight must change.
              </p>
            </div>
          </div>
        </div>

        {/* Step 4: Gradient Calculation & Weight Update */}
        <div className="callout space-y-4">
          <div className="flex flex-wrap items-center justify-between border-b border-rule pb-3 gap-2">
            <div>
              <span className="field-label">Step 4 — Gradient &amp; Update Rule</span>
              <h3 className="font-serif text-h3 text-ink">
                Backpropagation Rule: Δwᵢ = -η · (∂L/∂wᵢ) = -η · (δ · xᵢ)
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={applyUpdates}
                className="btn btn-primary text-xs min-h-9 py-1 px-3"
                title="Apply calculated updates to the current weights"
              >
                Apply Updates (W ← W + ΔW)
              </button>
              <button
                type="button"
                onClick={resetWeights}
                className="btn btn-secondary text-xs min-h-9 py-1 px-3"
              >
                Reset Weights
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs text-ink-muted">
            <span>Learning Rate (η):</span>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className="w-32 accent-primary"
            />
            <span className="font-mono text-ink font-bold">{learningRate.toFixed(2)}</span>
            <span className="ml-auto">
              Formula: <code className="font-mono bg-paper-sunken px-1.5 py-0.5 rounded-control">W_new = W_old + ΔW</code>
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-rule text-caption uppercase text-ink-muted font-sans">
                  <th scope="col" className="p-3">Weight Path</th>
                  <th scope="col" className="p-3">Input (xᵢ)</th>
                  <th scope="col" className="p-3">Current W (w_old)</th>
                  <th scope="col" className="p-3">Gradient (∂L/∂wᵢ)</th>
                  <th scope="col" className="p-3">Step (Δw = -η·grad)</th>
                  <th scope="col" className="p-3">New Weight (w_new)</th>
                  <th scope="col" className="p-3">Direction</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-rule font-mono">
                {weightGradients.map((g) => (
                  <tr key={g.feature} className="hover:bg-paper-sunken">
                    <td className="p-3 font-sans font-medium text-ink">{g.feature}</td>
                    <td className="p-3 tabular">{g.x.toFixed(3)}</td>
                    <td className="p-3 tabular font-semibold text-ink">{g.oldW.toFixed(4)}</td>
                    <td className="p-3 tabular text-secondary">{g.grad.toFixed(5)}</td>
                    <td className="p-3 tabular font-semibold text-accent">
                      {g.deltaW >= 0 ? `+${g.deltaW.toFixed(5)}` : g.deltaW.toFixed(5)}
                    </td>
                    <td className="p-3 tabular font-bold text-state-good">{g.newW.toFixed(4)}</td>
                    <td className="p-3 font-sans text-xs">
                      {g.deltaW > 0 ? (
                        <span className="text-state-good font-semibold">▲ Increase (+{g.pctChange}%)</span>
                      ) : g.deltaW < 0 ? (
                        <span className="text-primary font-semibold">▼ Decrease ({g.pctChange}%)</span>
                      ) : (
                        <span className="text-ink-muted">◌ No change</span>
                      )}
                    </td>
                  </tr>
                ))}
                {/* Bias Row */}
                <tr className="border-t-2 border-rule bg-paper-sunken font-mono">
                  <td className="p-3 font-sans font-semibold text-ink">Bias (b)</td>
                  <td className="p-3 tabular text-ink-muted">1.000</td>
                  <td className="p-3 tabular font-semibold text-ink">{biasUpdate.oldB.toFixed(4)}</td>
                  <td className="p-3 tabular text-secondary">{biasUpdate.gradB.toFixed(5)}</td>
                  <td className="p-3 tabular font-semibold text-accent">
                    {biasUpdate.deltaB >= 0 ? `+${biasUpdate.deltaB.toFixed(5)}` : biasUpdate.deltaB.toFixed(5)}
                  </td>
                  <td className="p-3 tabular font-bold text-state-good">{biasUpdate.newB.toFixed(4)}</td>
                  <td className="p-3 font-sans text-xs">
                    {biasUpdate.deltaB > 0 ? (
                      <span className="text-state-good font-semibold">▲ Increase</span>
                    ) : (
                      <span className="text-primary font-semibold">▼ Decrease</span>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Section 5: Weight Matrix Heatmap View */}
        <div className="callout space-y-4">
          <div className="flex flex-wrap items-center justify-between border-b border-rule pb-3 gap-2">
            <div>
              <span className="field-label">Weight Matrix Grid</span>
              <h3 className="font-serif text-h3 text-ink">
                Layer-to-Layer Synaptic Matrix Heatmap (W ∈ ℝᵐˣⁿ)
              </h3>
            </div>
            <div className="flex items-center gap-3 text-xs text-ink-muted">
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 bg-[#5A1D2E] rounded-control" /> Negative Weight
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 bg-paper-sunken border border-rule rounded-control" /> Near Zero
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 bg-[#1F3352] rounded-control" /> Positive Weight
              </span>
            </div>
          </div>

          <p className="text-sm text-ink-muted">
            The full weight matrix connecting Input Features to Hidden Neurons.
            Click on any cell to set that weight as the active inspection target above.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-center border-collapse">
              <thead>
                <tr className="border-b border-rule text-caption uppercase text-ink-muted">
                  <th className="p-2 text-left">Input Feature (Row)</th>
                  {matrixWeights.map((_, hIdx) => (
                    <th key={`h-${hIdx}`} className="p-2">
                      Neuron h₁,{hIdx + 1}
                    </th>
                  ))}
                  <th className="p-2 bg-paper-sunken font-bold">Active Weight (w)</th>
                </tr>
              </thead>
              <tbody>
                {featureList.map((feat, fIdx) => (
                  <tr key={feat.name} className="border-b border-rule">
                    <td className="p-3 text-left font-medium text-xs text-ink">
                      {feat.name}
                    </td>
                    {matrixWeights.map((hRow, hIdx) => {
                      const val = hRow[fIdx] ?? 0;
                      const isPos = val >= 0;
                      const intensity = Math.min(1, Math.abs(val));
                      const bg = isPos
                        ? `rgba(31, 51, 82, ${0.1 + intensity * 0.7})`
                        : `rgba(90, 29, 46, ${0.1 + intensity * 0.7})`;

                      return (
                        <td
                          key={`cell-${fIdx}-${hIdx}`}
                          style={{ backgroundColor: bg }}
                          className="p-3 font-mono text-xs font-semibold tabular cursor-pointer transition-transform hover:scale-105 border border-rule"
                          onClick={() => {
                            setWeights((prev) => {
                              const next = [...prev];
                              next[fIdx] = val;
                              return next;
                            });
                          }}
                          title={`Click to copy weight ${val} to active inspector`}
                        >
                          <span className={intensity > 0.5 ? "text-white" : "text-ink"}>
                            {val.toFixed(3)}
                          </span>
                        </td>
                      );
                    })}

                    {/* Active Inspector Weight column */}
                    <td className="p-3 font-mono text-xs font-bold bg-paper-sunken border-l-2 border-rule">
                      <span className={weights[fIdx] >= 0 ? "text-secondary" : "text-primary"}>
                        {weights[fIdx]?.toFixed(3)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
