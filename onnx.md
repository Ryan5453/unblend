# <img src="https://raw.githubusercontent.com/Ryan5453/unblend/main/web/app/public/favicon.svg" width="30"> ONNX Export

Unblend can export its models to ONNX for deployment in browsers, mobile apps, or other runtimes. This is how the [un/blend web app](https://unblend.dev) runs separation in-browser.

Export needs the `onnx` extra: `uv pip install 'unblend[onnx]'`.

The CLI can export any single-checkpoint model (ensembles are not currently supported) like this:

```bash
$ unblend export-onnx --model htdemucs
```

| Flag | Meaning |
|---|---|
| `-m/--model` | Model name (default `htdemucs`) |
| `-o/--output` | Output path (default `{model}_{precision}.onnx`, `_static` with `--static-batch`; for `native`, `{precision}` is the precision the checkpoint resolves to, e.g. `htdemucs_fp16.onnx`) |
| `--precision` | `native` (default), `fp32`, `bf16`, `fp16`, `fp8_e5m2`, `fp8_e4m3` (see [Precision](#precision)) |
| `--opset` | Opset version (default 17; raised to 18 for RoFormer and SCNet, and to 19 for fp8 storage, which needs opset 19's fp8 types). HTDemucs goes through PyTorch's TorchScript exporter, which stops at 20 |
| `--static-batch` | Trace a fixed batch of 1 instead of a dynamic batch axis |

From Python, `export_to_onnx` takes the same options and returns the path it wrote:

```python
from unblend.onnx import export_to_onnx

path = export_to_onnx(
    model_name="htdemucs",
    output_path=None,        # default: {model}_{precision}.onnx, native resolved
    opset_version=17,
    precision="native",
    static_batch=False,
)
```

## Precision

`--precision` sets the dtype the weights are *stored* at, not the dtype the graph computes in. It exists to shrink downloads for browsers, which cache far worse than the Python path. The default `native` follows the checkpoint's Safetensors header.

The exception is RoFormer at `fp16`, which converts the arithmetic too. Every other family keeps fp32 math whatever the weights are stored as. Both have `weight_precision: fp16`; `compute_precision` tells them apart.

fp8 costs real accuracy, even with each tensor stored at its own scale. Across MUSDB18-HQ tracks, HTDemucs at `fp8_e4m3` lands about 5–31 dB SNR from the fp32 output per stem, against 33–55 dB for `bf16`; `other` and drums suffer most, and the spread depends on the track. `fp8_e5m2`, with only two mantissa bits, falls to about −7–25 dB, which audibly breaks separation, so the export warns about it. Use `fp16` or `bf16`; the fp8 formats are for when download size matters more than quality.

## Shared contract

The graph is the neural network alone. You compute the STFT before it and the iSTFT after it, with parameters that must match the model exactly — so every export carries its own parameters, under the same keys for every family:

```python
import onnx

meta = {p.key: p.value for p in onnx.load("model.onnx").metadata_props}
```

| Key | Value |
|---|---|
| `sources` | JSON list of stem names, e.g. `["drums", "bass", "other", "vocals"]`. For RoFormer, the graph emits the first `num_stems` of them (see below) |
| `sample_rate` | Rate the checkpoint runs at; `44100` across the current registry |
| `audio_channels` | `2` across the current registry |
| `segment_samples` | Samples per call, i.e. the graph's input length (for SCNet, the padded length; see below) |
| `model_family` | `demucs`, `roformer`, or `scnet` |
| `architecture` | Registry architecture name |
| `stft_n_fft`, `stft_hop_length`, `stft_win_length` | STFT geometry, spelled as the `torch.stft` keywords |
| `stft_normalized` | `true` or `false` |
| `stft_window` | `hann`, or `none` for plain SCNet |
| `weight_precision` | `fp32`, `fp16`, `bf16`, `fp8_e5m2` or `fp8_e4m3` |
| `compute_precision` | Precision the graph computes in: `fp16` for mixed-precision RoFormer, `fp32` otherwise |
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
- The STFT follows Demucs' own framing. With `hop = stft_hop_length`, `le = ceil(samples / hop)` and `pad = stft_pad_samples` (1536): reflect-pad `pad` samples on the left and `pad + le * hop - samples` on the right (the extra part rounds the length up to a whole number of hops; 84 samples for the 343980-sample segment), run a centered, normalized STFT, drop the top frequency bin, then keep frames `2 : 2 + le` (`stft_frame_trim` is 2). That leaves `T = le` = 336 frames. `unblend.onnx.compute_stft_for_export` is the reference implementation.
- The inverse mirrors it. Put back a zero top frequency bin, zero-pad two frames on each side (`T + 4` frames), run a centered, normalized Hann iSTFT with output length `le * hop + 2 * pad`, and keep samples `pad : pad + samples`. `HTDemucs._ispec` is the reference. `torch.istft(normalized=True)` is already correct; a raw FFT library needs an extra `sqrt(n_fft)` factor applied yourself.
- Sum the frequency branch with `out_wave` per stem.

## RoFormer

| Tensor | Shape |
|---|---|
| `spec_real`, `spec_imag` | `[B, C, F, T]` |
| `out_spec_real`, `out_spec_imag` | `[B, S, C, F, T]` |

- A centered Hann STFT with no pre-padding, normalized only when `stft_normalized` says so (it doesn't for any registered RoFormer).
- Geometry is per-checkpoint — `melband_roformer_kim` hops 441 where the BS-RoFormers hop 512 — so read the metadata rather than assuming.
- There is no audio input and no time-domain branch, so skip the combine step entirely.
- RoFormer adds `num_stems`, the `S` the graph emits, and `output_complement`. When `output_complement` is `"true"`, `sources` names two stems but the graph emits only the first (`num_stems` is 1). Compute the second client-side as `mixture - stem`.

## SCNet

Spectrogram in, spectrograms out, same shapes as RoFormer. Both registered SCNets use a normalized STFT (`stft_normalized` is `true`, where the registered RoFormers' is `false`), so read the key rather than reusing RoFormer's settings. Two more differences:

- **Pad first.** SCNet's trunk needs an even FFT frame count. Pad the audio with `unblend.scnet.stft_padding(samples, hop_length)` before the STFT, then trim that many samples after the iSTFT. `segment_samples` is the padded length the graph expects and `logical_segment_samples` the audio length before padding (`scnet_small`: 485100 becomes 486400); `stft_pad_samples` here is that difference, the trailing zeros appended, unlike HTDemucs' key of the same name.
- **Window varies by variant.** `stft_window` is `"hann"` for masked variants and `"none"` for plain SCNet, where adding a window would silently change the result.

## Full tracks

The graph runs one segment. To match `Separator.separate()` on a whole track, with `L` the chunk length (`segment_samples`, or for SCNet `logical_segment_samples`, since SCNet pads each chunk only after chunking):

- Start a chunk every `int(L * (1 - 0.25))` samples (the default `split_overlap` is 0.25), from sample 0 while the start is inside the track. Each chunk covers `L` samples, or fewer at the end.
- A short last chunk is widened to `L` around its center, taking real audio from before it and zeros past the end, and its output is cut back to the chunk.
- HTDemucs normalizes the whole track once, before chunking, as described above.
- Weight each chunk's output by a triangular window over `L` (`1..L/2` rising, then falling to 1, divided by its maximum; a short chunk uses the window's start). Sum the weighted outputs and divide by the summed weights.
- With `shifts` rounds (the default is 1), for each round: zero-pad the track by `S` = half a second on each side, draw a random offset `o` in `0..S`, and treat the stretch from `o` to the end of the real audio (`n + S`) as the track above. The zero padding past it is context for the last chunk, not part of the chunk grid. Drop the first `S - o` output samples, keep `n`, and average the rounds.

`unblend.apply.apply_model` is the reference, and `web/unblend/src/pipeline.ts` is a JavaScript one.
