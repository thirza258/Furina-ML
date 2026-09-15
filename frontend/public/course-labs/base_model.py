"""Furina ML: train, select, test and reload a small base model on CPU.

Run: python base_model.py --steps 200 --output-dir base_model_results
Uses generated teaching data, not a real-world performance benchmark.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def dataset(count, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 2, generator=generator)
    y = (x[:, 0] * x[:, 1] > 0).long()
    return x, y


def make_model(width):
    return nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("base_model_results"))
    args = parser.parse_args()
    if args.steps < 1 or args.width < 1 or args.learning_rate <= 0:
        parser.error("steps, width and learning-rate must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_x, train_y = dataset(768, 101)
    val_x, val_y = dataset(256, 202)
    test_x, test_y = dataset(256, 303)
    # Only training inputs determine the fitted transformation.
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(1e-6)
    train_x, val_x, test_x = [(x - mean) / scale for x in (train_x, val_x, test_x)]
    model = make_model(args.width)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)
    history, best_loss, best_state, best_step = [], float("inf"), None, None
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(len(train_x), (64,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x[indices]), train_y[indices])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; inspect the learning rate and data")
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(model(val_x), val_y).item()
            history.append({"step": step, "training_loss": loss.item(), "validation_loss": val_loss})
            if val_loss < best_loss:
                best_loss, best_state, best_step = val_loss, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    model.eval()
    # Test data enters only after checkpoint selection has finished.
    with torch.no_grad():
        logits = model(test_x)
        accuracy = (logits.argmax(-1) == test_y).float().mean().item()
        baseline_class = train_y.bincount(minlength=2).argmax()
        baseline = (test_y == baseline_class).float().mean().item()
    assert any(not torch.equal(initial[key], value) for key, value in best_state.items())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    torch.save({"model": best_state, "width": args.width, "mean": mean, "scale": scale}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded = make_model(saved["width"])
    reloaded.load_state_dict(saved["model"])
    reloaded.eval()
    raw_test_x, _ = dataset(256, 303)
    with torch.no_grad():
        restored_logits = reloaded((raw_test_x - saved["mean"]) / saved["scale"])
    torch.testing.assert_close(logits, restored_logits)
    report = {
        "task": "generated nonlinear classification; CPU teaching lab",
        "torch_version": torch.__version__, "seed": args.seed,
        "steps": args.steps, "selected_step": best_step,
        "validation_loss": best_loss, "test_accuracy": accuracy,
        "majority_baseline_test_accuracy": baseline, "checkpoint_reload_verified": True,
        "history": history,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
