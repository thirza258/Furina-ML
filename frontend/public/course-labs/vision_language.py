"""Furina ML: multi-positive image/text alignment on generated images.

Run: python vision_language.py --steps 150
Learns six color/orientation descriptions; it does not generate captions or chat.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DESCRIPTIONS = [f"{color} {direction} stripe" for color in ("red", "green", "blue")
                for direction in ("horizontal", "vertical")]
CAPTIONS = torch.tensor([[color, 3 + direction, 5] for color in range(3) for direction in range(2)])


def dataset(count, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(6, (count,), generator=g)
    images = torch.rand(count, 3, 16, 16, generator=g) * 0.08
    for index, label in enumerate(labels.tolist()):
        color, direction = divmod(label, 2)
        start = int(torch.randint(3, 8, (), generator=g))
        if direction == 0:
            images[index, color, start:start + 5, 2:14] += 0.8
        else:
            images[index, color, 2:14, start:start + 5] += 0.8
    return images, labels


class AlignmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Flatten(), nn.Linear(8 * 8 * 8, 16))
        self.words = nn.Embedding(6, 16)
        self.text = nn.Linear(16, 16)

    def image_features(self, images):
        return F.normalize(self.image(images), dim=-1)

    def text_features(self, captions):
        return F.normalize(self.text(self.words(captions).mean(1)), dim=-1)


def alignment_loss(model, images, labels):
    logits = model.image_features(images) @ model.text_features(CAPTIONS[labels]).T / 0.1
    valid_matches = labels[:, None] == labels[None, :]
    # Repeated descriptions are valid positives, not false negatives.
    positives = logits.masked_fill(~valid_matches, float("-inf"))
    image_to_text = logits.logsumexp(1) - positives.logsumexp(1)
    text_to_image = logits.logsumexp(0) - positives.logsumexp(0)
    return (image_to_text.mean() + text_to_image.mean()) / 2


def retrieval_accuracy(model, images, labels):
    scores = model.image_features(images) @ model.text_features(CAPTIONS).T
    return (scores.argmax(1) == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("vision_language_results"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train = dataset(384, 101)
    validation = dataset(128, 202)
    test = dataset(128, 303)
    model = AlignmentModel()
    untrained = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    g = torch.Generator().manual_seed(args.seed + 1)
    best_loss, best_state, history = float("inf"), None, []
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(train[0]), (48,), generator=g)
        optimizer.zero_grad(set_to_none=True)
        loss = alignment_loss(model, train[0][ids], train[1][ids])
        assert torch.isfinite(loss), "Non-finite alignment loss"
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                value = alignment_loss(model, *validation).item()
            history.append({"step": step, "validation_loss": value})
            if value < best_loss:
                best_loss, best_state = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        trained_accuracy = retrieval_accuracy(model, *test)
        initial_accuracy = retrieval_accuracy(untrained, *test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    torch.save({"model": best_state, "descriptions": DESCRIPTIONS, "caption_ids": CAPTIONS}, checkpoint)
    restored = AlignmentModel()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.image_features(test[0][:4]), restored.image_features(test[0][:4]))
        torch.testing.assert_close(model.text_features(CAPTIONS), restored.text_features(CAPTIONS))
    # A portable image sample; many image viewers can open PPM files.
    pixels = test[0][0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).reshape(-1).tolist()
    (args.output_dir / "sample.ppm").write_bytes(b"P6\n16 16\n255\n" + bytes(pixels))
    report = {"task": "six generated color/orientation descriptions", "torch_version": torch.__version__,
              "seed": args.seed, "steps": args.steps, "candidate_descriptions": DESCRIPTIONS,
              "untrained_test_recall_at_1": initial_accuracy, "trained_test_recall_at_1": trained_accuracy,
              "validation_loss": best_loss, "sample_description": DESCRIPTIONS[int(test[1][0])],
              "multi_positive_training": True, "checkpoint_reload_verified": True, "history": history}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
