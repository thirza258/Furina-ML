"""Furina ML: CTC recognition of generated tone sequences, on CPU.

Run: python voice_model.py --steps 180 --noise 0.02
This is synthetic audio-to-symbol learning, not human speech recognition.
"""
import argparse
import array
import copy
from itertools import groupby
import json
import math
from pathlib import Path
import sys
import wave

import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE, FFT_SIZE, HOP = 8000, 128, 80


def collapse_ctc(path, blank=0):
    return [label for label, _ in groupby(path) if label != blank]


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, actual in enumerate(reference, 1):
        row = [i]
        for j, predicted in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (actual != predicted)))
        previous = row
    return previous[-1]


def dataset(count, seed, noise):
    g = torch.Generator().manual_seed(seed)
    features, targets, waveforms = [], [], []
    for _ in range(count):
        labels = torch.randint(1, 4, (int(torch.randint(2, 6, (), generator=g)),), generator=g)
        pieces = [torch.zeros(160)]
        for label in labels:
            length = int(torch.randint(400, 641, (), generator=g))
            time = torch.arange(length) / SAMPLE_RATE
            tone = 0.6 * torch.sin(2 * math.pi * (250 + 400 * int(label)) * time)
            pieces.extend([tone, torch.zeros(160)])
        waveform = torch.cat(pieces)
        waveform += noise * torch.randn(waveform.shape, generator=g)
        spectrum = torch.stft(waveform, n_fft=FFT_SIZE, hop_length=HOP,
                              window=torch.hann_window(FFT_SIZE), center=False, return_complex=True)
        features.append(spectrum.abs().log1p().T)
        targets.append(labels)
        waveforms.append(waveform)
    lengths = torch.tensor([len(value) for value in features], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    return padded, lengths, targets, waveforms


class AudioRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(FFT_SIZE // 2 + 1, 32)
        self.sequence = nn.GRU(32, 48, batch_first=True)
        self.output = nn.Linear(48, 4)  # blank plus three tone labels

    def forward(self, features):
        hidden, _ = self.sequence(F.relu(self.input(features)))
        return self.output(hidden).log_softmax(-1).transpose(0, 1)


def ctc_loss(model, features, lengths, targets):
    target_lengths = torch.tensor([len(value) for value in targets], dtype=torch.long)
    assert (lengths >= target_lengths * 2).all(), "Not enough frames for repeated-label alignment"
    return F.ctc_loss(model(features), torch.cat(targets), lengths, target_lengths,
                      blank=0, reduction="mean", zero_infinity=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("voice_model_results"))
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.noise) or args.noise < 0:
        parser.error("steps must be positive and noise must be finite and nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    assert collapse_ctc([1, 1, 0, 2, 2]) == [1, 2]
    assert collapse_ctc([1, 0, 1]) == [1, 1]
    assert edit_distance([1, 2], [1, 3, 2]) == 1
    train, validation, test = [dataset(count, seed, args.noise) for count, seed in [(128, 101), (40, 202), (40, 303)]]
    model = AudioRecognizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    g = torch.Generator().manual_seed(args.seed + 1)
    best_loss, best_state, history = float("inf"), None, []
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(train[0]), (16,), generator=g)
        optimizer.zero_grad(set_to_none=True)
        loss = ctc_loss(model, train[0][ids], train[1][ids], [train[2][i] for i in ids.tolist()])
        assert torch.isfinite(loss), "Non-finite CTC loss; inspect lengths and features"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                value = ctc_loss(model, *validation[:3]).item()
            history.append({"step": step, "validation_ctc_loss": value})
            if value < best_loss:
                best_loss, best_state = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        paths = model(test[0]).argmax(-1).T
    references = [value.tolist() for value in test[2]]
    hypotheses = [collapse_ctc(path[:int(length)].tolist()) for path, length in zip(paths, test[1])]
    edits = sum(edit_distance(actual, predicted) for actual, predicted in zip(references, hypotheses))
    reference_symbols = sum(len(value) for value in references)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    torch.save({"model": best_state, "sample_rate": SAMPLE_RATE, "n_fft": FFT_SIZE, "hop": HOP}, checkpoint)
    restored = AudioRecognizer()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(test[0][:2]), restored(test[0][:2]))
    pcm = array.array("h", test[3][0].clamp(-1, 1).mul(32767).short().tolist())
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(args.output_dir / "sample.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())
    report = {"task": "generated tones, not human speech", "torch_version": torch.__version__,
              "seed": args.seed, "steps": args.steps, "noise_std": args.noise,
              "sample_rate": SAMPLE_RATE, "validation_ctc_loss": best_loss,
              "test_symbol_error_rate": edits / reference_symbols, "test_edit_count": edits,
              "test_reference_symbols": reference_symbols, "ctc_decoder_checks_verified": True,
              "checkpoint_reload_verified": True, "sample_reference": references[0],
              "sample_hypothesis": hypotheses[0], "history": history}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
