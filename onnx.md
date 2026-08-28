# ONNX Export

Unblend can export its models to ONNX for deployment in browsers, mobile apps, or other runtimes. This is how the [un/blend web app](https://demucs.app) runs separation in-browser.

The CLI exports any single-checkpoint model, including your own: names resolve through the same registry as `separate`, so an entry in your `UNBLEND_EXTRA_MODELS` file exports exactly like a shipped one, and every `architecture` Unblend accepts belongs to one of the three families below. Multi-checkpoint entries — the `htdemucs_ft` bag, the two ensembles — have no single graph to trace and are rejected up front, before anything downloads.

```bash
# FP32, written to htdemucs_fp32.onnx
unblend export-onnx --model htdemucs

# Weight-only FP16, roughly half the size, near-identical output
unblend export-onnx --model bs_roformer_sw --fp16
```

| Flag | Meaning |
|---|---|
| `-m/--model` | Model name (default `htdemucs`) |
| `-o/--output` | Output path (default `{model}_fp32.onnx`, `_fp16` with `--fp16`, `_static` with `--static-batch`) |
| `--fp16` | Store weights as fp16; compute and IO stay fp32 |
| `--opset` | Opset version (raised to 18 for RoFormer and SCNet) |
| `--static-batch` | Trace a fixed batch of 1 instead of a dynamic batch axis |

`--fp16` is weight-only: the trained weights are rounded to fp16 so downloads are roughly half the size, but every operation still computes in fp32. This matters for browser runtimes, where pure-fp16 accumulation in Conv/MatMul kernels produces audible quantization noise; keeping compute in fp32 sidesteps that with no measurable quality loss.

## Shared contract

The graph contains only the neural network. You compute the STFT before it and the iSTFT after it, with parameters that must match the model exactly. Every export embeds its own parameters under the same keys, so you never have to guess:

```python
import onnx, json

metadata = {p.key: p.value for p in onnx.load("model.onnx").metadata_props}
sources = json.loads(metadata["sources"])      # e.g. ["drums", "bass", "other", "vocals"]
sample_rate = int(metadata["sample_rate"])     # 44100
segment = int(metadata["segment_samples"])     # feed exactly this many samples per call
family = metadata["model_family"]              # "demucs", "roformer" or "scnet"
n_fft = int(metadata["stft_n_fft"])
hop = int(metadata["stft_hop_length"])
win = int(metadata["stft_win_length"])
normalized = metadata["stft_normalized"] == "true"
window = metadata["stft_window"]               # "hann", or "none" for plain SCNet
```

Booleans are always `"true"`/`"false"` and `sources` is always JSON. Each family adds a few keys of its own, listed below.

Only the batch axis is dynamic unless you pass `--static-batch`, which traces a fixed batch of 1 for single-stream consumers such as browsers. The graph is traced at the model's training chunk length, so shorter or longer inputs are rejected.

## HTDemucs

Inputs `spec_real` and `spec_imag` (`[B, 2, 2048, T]`) plus the raw waveform `audio` (`[B, 2, samples]`). Outputs `out_spec_real` / `out_spec_imag` (`[B, S, 2, 2048, T]`) and the time-branch `out_wave` (`[B, S, 2, samples]`), where `S` is the stem count.

Segment length is 343980 samples (~7.8 s at 44.1 kHz). The STFT is centered with reflect padding, plus Demucs-style pre-padding of `stft_pad_samples` (1536), a `stft_frame_trim` of two frames on each side, and the top frequency bin dropped. One caveat if you reimplement the iSTFT with a raw FFT library: apply an extra `sqrt(n_fft)` factor yourself. PyTorch's `torch.istft(normalized=True)` already includes it. Finally, sum the frequency branch with `out_wave` per stem.

## RoFormer

Inputs `spec_real` / `spec_imag` (`[B, C, F, T]`). Outputs `out_spec_real` / `out_spec_imag` (`[B, S, C, F, T]`). There is no audio input and no time-domain branch, so skip the combine step entirely.

A plain centered Hann STFT, with no extra scaling factor and no pre-padding. Geometry differs per checkpoint — `melband_roformer_kim` hops 441 where the BS-RoFormers hop 512 — so read the metadata rather than assuming. Single-mask checkpoints (`output_complement: "true"`) emit one stem; compute the second client-side as `mixture - stem`.

## SCNet

Same spectrogram-in, spectrograms-out shape as RoFormer, with two differences:

1. **Pad first.** SCNet's trunk needs an even FFT frame count, so pad the audio with `unblend.scnet.stft_padding(samples, hop_length)` before computing the STFT, then trim that many samples off after the iSTFT. `segment_samples` is the padded length the graph expects; `logical_segment_samples` is the audio you actually feed it (`scnet_small`: 485100 becomes 486400).
2. **Window varies by variant.** `stft_window` is `"hann"` for masked variants and `"none"` for plain SCNet, where adding a window would silently change the result.
