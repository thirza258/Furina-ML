# Furina ML — Python course labs

These are complete, standalone CPU teaching scripts. They generate small local
datasets, fit real models, make their selections using validation data, and
evaluate on test examples kept apart from those decisions. The neural-model
labs generate their test examples separately; the forecasting lab tests on the
final, later segment of its series. They require no API key, model download,
microphone access or paid service.

The system-design lab also replays **simulated request timings and dependency
incidents**. Its model computation is real; the replay is not a serving benchmark
or an experiment on product outcomes.

The symbolic, visual and audio datasets are intentionally small. Their metrics
describe these teaching tasks; they do not establish real-world language,
vision or human-speech capability. The advanced course lessons describe the
additional data, architectures and evaluation needed for those applications.

## Setup

### System-design lab — Python standard library

Download `ml_system_design.py` and optionally `ml_system_design_workbook.md`.
Use Python 3.10 or later; no package installation is required. This lab was
verified with Python 3.14.4.

```sh
python3 ml_system_design.py --output-dir runs/normal
python3 ml_system_design.py --load-model runs/normal/model.json --candidate-latency-ms 300 --output-dir runs/slow
```

On Windows, use `python` in place of `python3`. The workbook covers feature
outages, input shifts, zero-canary runs and a full design review. The lab writes
`model.json`, `manifest.json`, `report.json` and `predictions.jsonl`. It uses
content-derived model versions, validates feature availability, and distinguishes
assigned releases from releases that actually served after fallback.

### Neural-model labs — PyTorch

Download the desired `.py` file and `requirements.txt` into a folder. Use a
Python version supported by the pinned PyTorch release. These scripts were
verified with Python 3.14.4 and PyTorch 2.14.0 on macOS ARM64.

```sh
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell activation: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python base_model.py
```

Each script defaults to one CPU thread and a small fixed update budget. The
initial package installation needs internet access; the scripts themselves
make no network requests. Run `python SCRIPT.py --help` for supported options.

### Time-series lab — NumPy and scikit-learn

Download `time_series_forecasting.py` and `requirements-sklearn.txt` into a
folder. This lab does not need PyTorch or the shared `requirements.txt`: its
own requirements file pins NumPy 2.3.4 and scikit-learn 1.9.0, the versions
used by Furina's backend, and pip installs joblib as a scikit-learn
dependency. Both pinned packages require Python 3.11 or later. The lab was
verified with Python 3.14.4 on macOS ARM64.

```sh
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell activation: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-sklearn.txt
python time_series_forecasting.py --output-dir time_series_results
```

Unlike the other labs, this one generates a single daily series and divides it
into consecutive training, validation and test periods, so its test partition
is the final period of that series rather than a separately generated sample.
Rolling-origin backtests on the validation period select the model, which is
refitted once on training and validation days before the test period is
scored. `--trend-per-day` changes all three periods, while `--seed` changes
only the gradient-boosting randomness. The script writes `model.joblib`,
`report.json` and `forecasts.csv`. `model.joblib` is a pickle file, so only
load copies you trust. Unless you set `OMP_NUM_THREADS` yourself, the script
uses one OpenMP thread, because extra threads made these small fits much
slower in testing.

## Labs

| Script | What it trains | Evaluation and checks |
|---|---|---|
| `time_series_forecasting.py` | Ridge and gradient-boosting forecasters on a generated daily series | Rolling-origin backtests, seasonal baselines, test MAE/RMSE/MASE by horizon step, calibrated interval coverage, leakage and reload checks |
| `base_model.py` | MLP on a nonlinear classification boundary | Validation selection, independent test accuracy, training-only scaling, checkpoint round trip |
| `intermediate_model.py` | Imbalanced binary classifier | Validation-selected threshold, test precision/recall/F1 and confusion counts |
| `transformer_model.py` | Small causal transformer on symbol transitions | Future-token invariance, test token loss, perplexity and generation |
| `llm_finetuning.py` | Small causal symbol model, followed by an output-head LoRA adapter | Response-only masking, frozen-base invariant, test response loss, checkpoint round trip |
| `vision_language.py` | Image and text encoders on color/orientation pairs | Multi-positive contrastive loss and held-out retrieval among six descriptions |
| `voice_model.py` | Spectral features and a recurrent CTC recognizer on tones | CTC collapse checks, held-out symbol error rate, generated WAV sample |
| `ml_system_design.py` | Logistic delivery-risk model and a constant baseline | Historical features, validation gates, release reload, canary rollback, fallbacks and mature-label evaluation |

Run each script independently; no script imports another lab file. For example:

```sh
python intermediate_model.py --unweighted --output-dir comparison_unweighted
python transformer_model.py --steps 150 --output-dir transformer_run
python llm_finetuning.py --rank 8 --output-dir adapter_rank_8
python voice_model.py --noise 0.05 --output-dir noisier_audio
python time_series_forecasting.py --horizon 28 --output-dir horizon_28
```

Changing `--noise` changes all three tone-data partitions; a fixed-model noise
robustness test requires changing evaluation noise separately in the source.
`--seed` controls training randomness. The data generators
use separate fixed seeds so changing the training seed preserves the split.

## Outputs and interpretation

Scripts write a model artifact and `report.json` inside their output directory.
The vision lab also writes `sample.ppm`; the audio lab writes `sample.wav`.
Rerunning with the same output directory replaces these generated artifacts.
Use a different `--output-dir` for each comparison you intend to retain.

Reports contain real measured values and library versions. Results may vary
with software or hardware. The scripts select on validation data and do not
assert a target test score. Inspect the baseline, loss history and errors, and
avoid repeatedly choosing settings by the test result.

Checkpoints include the model information used by the lab's reload check.
They are teaching inference artifacts, not exact-resume training snapshots:
to resume identically, also save optimizer/scheduler state, data position and
random-generator states. Only load checkpoint files you trust.

The LLM lab deliberately saves the complete tiny base-plus-adapter system.
An adapter-only release for a pretrained model would additionally need the
exact compatible base revision, tokenizer and prompt template.

## Capstone workflow

1. Complete the linked course lessons and run the baseline lab.
2. Write one hypothesis and a fixed validation-selection protocol.
3. Keep an experiment ledger, including failed attempts.
4. Choose a checkpoint using development data, then evaluate independently.
5. Reload the artifact in a fresh process and reproduce its input pipeline.
6. Submit the course's deliverables and review them against its rubric.

For natural-language, image-question-answering or human-speech projects, use
the primary-source recipes linked in the course with a suitably licensed model,
permitted data and appropriate hardware. Furina's web training controls remain
the tabular scikit-learn workbench; these Python exercises run separately.
