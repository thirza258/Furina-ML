"""Furina ML: validation-selected thresholds on an imbalanced teaching task.

Run: python intermediate_model.py --steps 180
Try --unweighted as a controlled comparison. Never choose settings on test data.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def dataset(count, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 6, generator=g)
    score = 1.5 * x[:, 0] - x[:, 1] + x[:, 2] * x[:, 3] + 0.2 * torch.randn(count, generator=g)
    return x, (score > 2.2).float()


def make_model():
    return nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))


def metrics(probabilities, targets, threshold):
    predicted, actual = probabilities >= threshold, targets.bool()
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("intermediate_model_results"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_x, train_y = dataset(1024, 111)
    val_x, val_y = dataset(512, 222)
    test_x, test_y = dataset(512, 333)
    mean, scale = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    train_x, val_x, test_x = [(x - mean) / scale for x in (train_x, val_x, test_x)]
    positive_weight = torch.tensor(1.0) if args.unweighted else (len(train_y) - train_y.sum()) / train_y.sum()
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    g = torch.Generator().manual_seed(args.seed + 1)
    best_loss, best_state, history = float("inf"), None, []
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(train_x), (128,), generator=g)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(train_x[ids]).squeeze(-1), train_y[ids], pos_weight=positive_weight)
        assert torch.isfinite(loss), "Training loss is not finite"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                value = F.binary_cross_entropy_with_logits(model(val_x).squeeze(-1), val_y).item()
            history.append({"step": step, "validation_loss": value})
            if value < best_loss:
                best_loss, best_state = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probability = model(val_x).squeeze(-1).sigmoid()
    candidates = [metrics(val_probability, val_y, value / 100) for value in range(5, 96, 5)]
    selected = max(candidates, key=lambda item: (item["f1"], -abs(item["threshold"] - 0.5)))
    # The threshold is now fixed. Test labels do not select it.
    with torch.no_grad():
        test_probability = model(test_x).squeeze(-1).sigmoid()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    torch.save({"model": best_state, "threshold": selected["threshold"], "mean": mean, "scale": scale}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded = make_model()
    reloaded.load_state_dict(saved["model"])
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(test_probability, reloaded(test_x).squeeze(-1).sigmoid())
    report = {"task": "generated rare-class classification", "torch_version": torch.__version__,
              "seed": args.seed, "steps": args.steps, "positive_weight": positive_weight.item(),
              "validation_selection": selected, "validation_threshold_candidates": candidates,
              "test_default_threshold": metrics(test_probability, test_y, 0.5),
              "test_selected_threshold": metrics(test_probability, test_y, saved["threshold"]),
              "checkpoint_reload_verified": True, "history": history}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"history", "validation_threshold_candidates"}}, indent=2))


if __name__ == "__main__":
    main()
