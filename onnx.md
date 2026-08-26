# ONNX Export

Unblend can export its models to ONNX for deployment in browsers, mobile apps, or other runtimes. This is how the [un/blend web app](https://demucs.app) runs separation in-browser.

## Command

```bash
# FP32, written to htdemucs_fp32.onnx
unblend export-onnx --model htdemucs

# Weight-only FP16, roughly half the size, near-identical output
unblend export-onnx --model bs_roformer_sw --fp16
```

| Flag | Meaning |
|---|---|
| `-m/--model` | Model name (default `htdemucs`) |
| `-o/--output` | Output path (default `{model}_fp32.onnx` or `{model}_fp16.onnx`) |
| `--fp16` | Store weights as fp16; compute and IO stay fp32 |
| `--opset` | Opset version (raised to 18 for RoFormer and SCNet) |
| `--static-batch` | Trace a fixed batch of 1 instead of a dynamic batch axis |

`--fp16` is weight-only: the trained weights are rounded to fp16 so downloads are roughly half the size, but every operation still computes in fp32. This matters for browser runtimes, where pure-fp16 accumulation in Conv/MatMul kernels produces audible quantization noise; keeping compute in fp32 sidesteps that with no measurable quality loss.

## Shared contract

The graph contains only the neural network. You compute the STFT before it and the iSTFT after it, with parameters that must match the model exactly. Every export embeds its own parameters as metadata so you never have to guess:

```python
import onnx, json

metadata = {p.key: p.value for p in onnx.load("model.onnx").metadata_props}
sources = json.loads(metadata["sources"])      # e.g. ["drums", "bass", "other", "vocals"]
sample_rate = int(metadata["sample_rate"])     # 44100
segment = int(metadata["segment_samples"])     # feed exactly this many samples per call
n_fft = int(metadata["stft_n_fft"])
hop = int(metadata["stft_hop_length"])
win = int(metadata["stft_win_length"])
normalized = metadata["stft_normalized"] == "true"
window = metadata.get("unblend.stft_window")   # SCNet only: "hann" or "none"
```

Only the batch axis is dynamic unless you pass `--static-batch`, which traces a fixed batch of 1 for single-stream consumers such as browsers. The graph is traced at the model's training chunk length, so shorter or longer inputs are rejected.

## HTDemucs

Inputs `spec_real` and `spec_imag` (`[B, 2, 2048, T]`) plus the raw waveform `audio` (`[B, 2, samples]`). Outputs `out_spec_real` / `out_spec_imag` (`[B, S, 2, 2048, T]`) and the time-branch `out_wave` (`[B, S, 2, samples]`), where `S` is the stem count.

Segment length is 343980 samples (~7.8 s at 44.1 kHz). The STFT uses `n_fft=4096`, `hop_length=1024`, Hann window, `normalized=True`, `center=True`, reflect padding, with Demucs-style pre-padding of `1536` samples and a two-frame trim on each side. One caveat if you reimplement the iSTFT with a raw FFT library: apply an extra `sqrt(n_fft)` factor yourself. PyTorch's `torch.istft(normalized=True)` already includes it. Finally, sum the frequency branch with `out_wave` per stem.

## RoFormer

Inputs `spec_real` / `spec_imag` (`[B, C, F, T]`). Outputs `out_spec_real` / `out_spec_imag` (`[B, S, C, F, T]`). There is no audio input and no time-domain branch, so skip the combine step entirely.

The shipped checkpoints use a plain centered Hann STFT at 2048/512/2048 with `stft_normalized=false`. No extra scaling factor and no pre-padding applies here; read the embedded metadata rather than assuming. Single-mask checkpoints (metadata key `output_complement: "true"`, e.g. `melband_roformer_kim`) emit one stem; compute the second client-side as `mixture - stem`.

## SCNet

Same spectrogram-in, spectrograms-out shape as RoFormer, with two differences:

1. **Pad first.** SCNet's trunk needs an even FFT frame count, so pad the audio with `unblend.scnet.stft_padding(samples, hop_length)` before computing the STFT, then trim that many samples off after the iSTFT (`scnet_small`: 485100 becomes 486400).
2. **Window varies by variant.** Read `unblend.stft_window` from the metadata: `"hann"` for masked variants, `"none"` for plain SCNet, where adding a window would silently change the result.
