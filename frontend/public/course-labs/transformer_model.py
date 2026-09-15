"""Furina ML: a tested causal transformer on a generated symbol grammar.

Run: python transformer_model.py --steps 150
The grammar usually advances by one symbol, occasionally by two. This is a
local-transition teaching task, not evidence of natural-language understanding.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VOCABULARY = 8
CONTEXT = 16


def dataset(count, seed):
    g = torch.Generator().manual_seed(seed)
    first = torch.randint(VOCABULARY, (count, 1), generator=g)
    increments = 1 + (torch.randint(5, (count, CONTEXT), generator=g) == 0).long()
    tokens = torch.cat([first, (first + increments.cumsum(1)) % VOCABULARY], dim=1)
    return tokens[:, :-1], tokens[:, 1:]


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, 32)
        self.position = nn.Embedding(CONTEXT, 32)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, VOCABULARY)

    def forward(self, tokens):
        length = tokens.shape[1]
        assert 0 < length <= CONTEXT
        hidden = self.embedding(tokens) + self.position(torch.arange(length, device=tokens.device))
        # TransformerEncoder uses True for BLOCKED positions.
        blocked = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        return self.head(self.blocks(hidden, mask=blocked))


def token_loss(model, x, y):
    return F.cross_entropy(model(x).reshape(-1, VOCABULARY), y.reshape(-1))


def check_causality(model):
    model.eval()
    original, _ = dataset(4, 909)
    changed = original.clone()
    changed[:, 8:] = (changed[:, 8:] + 3) % VOCABULARY
    with torch.no_grad():
        torch.testing.assert_close(model(original)[:, :8], model(changed)[:, :8], atol=1e-5, rtol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("transformer_model_results"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_x, train_y = dataset(512, 101)
    val_x, val_y = dataset(128, 202)
    test_x, test_y = dataset(128, 303)
    model = TinyTransformer()
    check_causality(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    g = torch.Generator().manual_seed(args.seed + 1)
    best_loss, best_state, history = float("inf"), None, []
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(train_x), (32,), generator=g)
        optimizer.zero_grad(set_to_none=True)
        loss = token_loss(model, train_x[ids], train_y[ids])
        assert torch.isfinite(loss), "Training loss is not finite"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                value = token_loss(model, val_x, val_y).item()
            history.append({"step": step, "training_loss": loss.item(), "validation_loss": value})
            if value < best_loss:
                best_loss, best_state = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    check_causality(model)
    with torch.no_grad():
        test_loss = token_loss(model, test_x, test_y).item()
        sequence = torch.tensor([[0, 1, 2]])
        for _ in range(20):
            next_token = model(sequence[:, -CONTEXT:])[:, -1].argmax(-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    torch.save({"model": best_state, "context": CONTEXT, "vocabulary": VOCABULARY}, checkpoint)
    fresh = TinyTransformer()
    fresh.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"])
    fresh.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(test_x[:4]), fresh(test_x[:4]))
    report = {"task": "generated local-transition grammar", "torch_version": torch.__version__,
              "seed": args.seed, "steps": args.steps, "validation_loss": best_loss,
              "test_token_loss": test_loss, "test_perplexity": math.exp(test_loss),
              "uniform_token_loss": math.log(VOCABULARY), "causal_invariance_verified": True,
              "checkpoint_reload_verified": True, "generated_tokens": sequence[0].tolist(), "history": history}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
