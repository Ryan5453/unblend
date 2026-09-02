# <img src="/web/app/public/favicon.svg" width="30"> ONNX Export

Unblend can export its models to ONNX for deployment in browsers, mobile apps, or other runtimes. This is how the [un/blend web app](https://demucs.app) runs separation in-browser.

The CLI can export any single-checkpoint model (ensembles are not currently supported) like this:

```bash
$ unblend export-onnx --model htdemucs
```

| Flag | Meaning |
|---|---|
| `-m/--model` | Model name (default `htdemucs`) |
| `-o/--output` | Output path (default `{model}_{precision}.onnx`, `_static` with `--static-batch`) |
| `--precision` | `native` (default), `fp32`, `bf16`, `fp16`, `fp8_e5m2`, `fp8_e4m3` (see [Precision](#precision)) |
| `--opset` | Opset version (raised to 18 for RoFormer and SCNet) |
| `--static-batch` | Trace a fixed batch of 1 instead of a dynamic batch axis |

## Precision

`--precision` sets the dtype the weights are *stored* at, not the dtype the graph computes in. It exists to shrink downloads for browsers, which cache far worse than the Python path. The default `native` follows the checkpoint's Safetensors header.

The exception is RoFormer at `fp16`, which converts the arithmetic too. Every other family keeps fp32 math whatever the weights are stored as. Both are labelled `"fp16"`, so read `weight_precision` and `compute_precision` from the metadata rather than `precision`, which reports storage only.

## Shared contract

The graph is the neural network alone. You compute the STFT before it and the iSTFT after it, with parameters that must match the model exactly — so every export carries its own parameters, under the same keys for every family:

```python
import onnx

meta = {p.key: p.value for p in onnx.load("model.onnx").metadata_props}
```

| Key | Value |
|---|---|
| `sources` | JSON list of stem names, e.g. `["drums", "bass", "other", "vocals"]` |
| `sample_rate` | Rate the checkpoint runs at; `44100` across the current registry |
| `audio_channels` | `2` across the current registry |
| `segment_samples` | Samples per call — feed exactly this many |
| `model_family` | `demucs`, `roformer`, or `scnet` |
| `architecture` | Registry architecture name |
| `stft_n_fft`, `stft_hop_length`, `stft_win_length` | STFT geometry, spelled as the `torch.stft` keywords |
| `stft_normalized` | `true` or `false` |
| `stft_window` | `hann`, or `none` for plain SCNet |
| `weight_precision` | `fp16` or `fp32` |
| `compute_precision` | `fp16` only for mixed-precision RoFormer |
| `external_normalization` | `true` if you must normalize the track yourself |
| `batch_mode` | `static` or `dynamic` |
| `license` | Weight license, when the registry declares one |

Every value is a string: booleans are `"true"`/`"false"`, numbers need `int()`, and `sources` is JSON. Each family adds a few keys of its own, listed below.

Shapes below are written with `B` batch, `S` stems, `C` channels, `F` frequency bins, and `T` frames.

Only the batch axis is dynamic unless you pass `--static-batch`, which traces a fixed batch of 1 for single-stream consumers such as browsers. The graph is traced at the model's training chunk length, so shorter or longer inputs are rejected.

## HTDemucs

| Tensor | Shape |
|---|---|
| `spec_real`, `spec_imag` | `[B, 2, 2048, T]` |
| `audio` | `[B, 2, samples]` |
| `out_spec_real`, `out_spec_imag` | `[B, S, 2, 2048, T]` |
| `out_wave` | `[B, S, 2, samples]` |

- **Normalize the track yourself.** `external_normalization` is `true` here, and only here. Subtract the mean and divide by `1e-5 + std` — both taken over the channel-mean reference signal, std unbiased — before the STFT, then reverse it after the iSTFT.
- Segment is 343980 samples, ~7.8 s at 44.1 kHz.
- The STFT is centered with reflect padding, plus `stft_pad_samples` (1536) of Demucs pre-padding, a `stft_frame_trim` of two frames each side, and the top frequency bin dropped.
- `torch.istft(normalized=True)` is already correct. A raw FFT library needs an extra `sqrt(n_fft)` factor applied yourself.
- Sum the frequency branch with `out_wave` per stem.

## RoFormer

| Tensor | Shape |
|---|---|
| `spec_real`, `spec_imag` | `[B, C, F, T]` |
| `out_spec_real`, `out_spec_imag` | `[B, S, C, F, T]` |

- A plain centered Hann STFT: no extra scaling, no pre-padding.
- Geometry is per-checkpoint — `melband_roformer_kim` hops 441 where the BS-RoFormers hop 512 — so read the metadata rather than assuming.
- There is no audio input and no time-domain branch, so skip the combine step entirely.
- `output_complement: "true"` checkpoints emit one stem. Compute the second client-side as `mixture - stem`.

## SCNet

Spectrogram in, spectrograms out, same shapes as RoFormer. Two differences:

- **Pad first.** SCNet's trunk needs an even FFT frame count. Pad the audio with `unblend.scnet.stft_padding(samples, hop_length)` before the STFT, then trim that many samples after the iSTFT. `segment_samples` is the padded length the graph expects; `logical_segment_samples` is what you actually feed it (`scnet_small`: 485100 becomes 486400).
- **Window varies by variant.** `stft_window` is `"hann"` for masked variants and `"none"` for plain SCNet, where adding a window would silently change the result.
