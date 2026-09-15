"""Furina ML: miniature response-only SFT and LoRA on a causal language model.

Run: python llm_finetuning.py --steps 160 --rank 4
The base learns to copy a requested symbol; the adapter learns its successor.
Prompts contain irrelevant filler symbols. This is a mechanics lab, not an LLM.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VOCABULARY, CONTEXT = 12, 6
BOS, TASK, EOS = 8, 9, 10


def dataset(count, seed, offset):
    g = torch.Generator().manual_seed(seed)
    symbols = torch.randint(8, (count, 3), generator=g)
    tokens = torch.cat([torch.full((count, 1), BOS), torch.full((count, 1), TASK),
                        symbols, (symbols[:, -1:] + offset) % 8,
                        torch.full((count, 1), EOS)], dim=1)
    inputs, targets = tokens[:, :-1], tokens[:, 1:].clone()
    targets[:, :4] = -100  # Only answer and EOS targets are supervised.
    assert (targets != -100).sum().item() == count * 2
    return inputs, targets


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, 32)
        self.position = nn.Embedding(CONTEXT, 32)
        block = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, VOCABULARY)

    def forward(self, tokens):
        length = tokens.shape[1]
        hidden = self.embedding(tokens) + self.position(torch.arange(length))
        blocked = torch.ones(length, length, dtype=torch.bool).triu(1)
        return self.head(self.blocks(hidden, mask=blocked))


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = 1.0  # alpha / rank, with alpha = rank in this lab.

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def response_loss(model, x, y):
    return F.cross_entropy(model(x).reshape(-1, VOCABULARY), y.reshape(-1), ignore_index=-100)


def train_stage(model, train, validation, steps, learning_rate, seed):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    g = torch.Generator().manual_seed(seed)
    best_loss, best_state = float("inf"), None
    history = []
    for step in range(1, steps + 1):
        model.train()
        ids = torch.randint(len(train[0]), (32,), generator=g)
        optimizer.zero_grad(set_to_none=True)
        loss = response_loss(model, train[0][ids], train[1][ids])
        assert torch.isfinite(loss), "Non-finite response loss"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                value = response_loss(model, *validation).item()
            history.append({"step": step, "validation_response_loss": value})
            if value < best_loss:
                best_loss, best_state = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=160, help="Updates per stage")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("llm_finetuning_results"))
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.rank <= VOCABULARY:
        parser.error("steps must be positive and rank must be between 1 and 12")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = TinyLM()
    pretraining_history = train_stage(model, dataset(512, 101, 0), dataset(128, 202, 0), args.steps, 0.004, args.seed)
    base_reference = copy.deepcopy(model)
    model.requires_grad_(False)
    model.head = LoRALinear(model.head, args.rank)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    train = dataset(512, 404, 1)
    history = train_stage(model, train, dataset(128, 505, 1), args.steps, 0.03, args.seed + 1)
    for name, p in model.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name]), f"Frozen base weight changed: {name}"
    test = dataset(128, 606, 1)
    with torch.no_grad():
        adapted_loss = response_loss(model, *test).item()
        unchanged_base_loss = response_loss(base_reference, *test).item()
        predictions = model(test[0])[:, 4].argmax(-1)
        answer_accuracy = (predictions == test[1][:, 4]).float().mean().item()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "base_plus_adapter.pt"
    # Save the complete tiny system for a self-contained teaching artifact.
    torch.save({"model": model.state_dict(), "rank": args.rank, "vocabulary": VOCABULARY}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    restored = TinyLM()
    restored.head = LoRALinear(restored.head, saved["rank"])
    restored.load_state_dict(saved["model"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(test[0][:4]), restored(test[0][:4]))
    report = {"task": "symbol-instruction adaptation; independent draws from the same finite task",
              "torch_version": torch.__version__, "seed": args.seed, "steps_per_stage": args.steps,
              "rank": args.rank, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "base_test_response_loss": unchanged_base_loss, "adapted_test_response_loss": adapted_loss,
              "test_answer_accuracy": answer_accuracy, "frozen_base_verified": True,
              "checkpoint_reload_verified": True,
              "example_inputs": train[0][0].tolist(), "example_labels_minus_100_ignored": train[1][0].tolist(),
              "pretraining_history": pretraining_history, "adaptation_history": history}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("history")}, indent=2))


if __name__ == "__main__":
    main()
