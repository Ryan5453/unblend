# ONNX Export

`unblend` includes the ability to export its HTDemucs, RoFormer, and SCNet models to the ONNX format for deployment in browsers, mobile, or other runtimes.
This is how the [un/blend web app](https://demucs.app) runs source separation in-browser. HTDemucs specifics are below; the other architectures are covered in [RoFormer models](#roformer-models) and [SCNet](#scnet).

## Export

ONNX export is an internal developer tool, exposed as a hidden CLI command (it won't show in `unblend --help`):

```bash
# FP32 (default). Output defaults to {model}_fp32.onnx
unblend export-onnx --model htdemucs

# Weight-only FP16 — roughly halves file size; weights are rounded to fp16
# but compute and IO stay fp32, so output is near-identical (not bit-exact)
unblend export-onnx --model htdemucs --fp16 --output htdemucs_fp16.onnx
```

Flags: `-m/--model` (default `htdemucs`), `-o/--output`, `--opset` (default `17`), `--fp16`.

## Model Interface

**Inputs:**
- `spec_real`: Real part of STFT `[B, 2, 2048, T]`
- `spec_imag`: Imaginary part of STFT `[B, 2, 2048, T]`
- `audio`: Raw waveform `[B, 2, samples]`

**Outputs:**
- `out_spec_real`: Separated spectrograms (real) `[B, S, 2, 2048, T]`
- `out_spec_imag`: Separated spectrograms (imag) `[B, S, 2, 2048, T]`
- `out_wave`: Time-domain branch output `[B, S, 2, samples]`

Where `S` = number of sources (4 for htdemucs, 6 for htdemucs_6s).

## Inference Steps

The ONNX model contains only the core neural network - STFT and iSTFT are not included. You'll need to implement these yourself or use an existing FFT library.

The graph is traced at the model's training length, so feed exactly that many samples per call: `max_allowed_segment * sample_rate` = **343980 samples (~7.8s @ 44.1kHz)** for HTDemucs. (Only the batch axis is declared dynamic; the time/sample axes are fixed at the training length, since the cross-transformer's positional embeddings are only valid there — shorter or longer segments would degrade quality.)

The STFT/iSTFT parameters are fixed and must match exactly:

| Parameter | Value |
|---|---|
| `n_fft` | 4096 |
| `hop_length` | 1024 |
| `win_length` | 4096 |
| `window` | Hann, length 4096 |
| `normalized` | `True` |
| `center` | `True` |
| `pad_mode` | `"reflect"` |

> **Normalization caveat:** Demucs scales the iSTFT output by an extra `n_fft ** 0.5`. PyTorch's `torch.istft(normalized=True)` already folds this in, but if you reimplement the iSTFT with a raw FFT library (common in JS/WASM), apply the `sqrt(n_fft)` factor yourself or the output level will be wrong.

### 1. Preprocessing (STFT)

```python
NFFT = 4096
HOP = 1024
SEGMENT = 343980  # max_allowed_segment * sample_rate (~7.8s @ 44.1kHz)

# Pad audio to segment length
audio = pad(audio, SEGMENT)

# Demucs padding
le = ceil(samples / HOP)
pad_amount = HOP // 2 * 3  # 1536
audio_padded = reflect_pad(audio, (pad_amount, pad_amount + le * HOP - samples))

# STFT (params per the table above)
z = stft(audio_padded, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
         window=hann, normalized=True, center=True)

# Trim
z = z[..., :-1, :]      # Remove last freq bin: 2049 -> 2048
z = z[..., 2:2+le]      # Trim time: remove 2 frames each side

spec_real, spec_imag = z.real, z.imag
```

### 2. Run Inference

```python
out_real, out_imag, out_wave = session.run(
    ["out_spec_real", "out_spec_imag", "out_wave"],
    {"spec_real": spec_real, "spec_imag": spec_imag, "audio": audio}
)
```

### 3. Postprocessing (iSTFT + Combine)

```python
for each source:
    # Pad spectrogram back
    z = out_real[s] + 1j * out_imag[s]
    z = pad(z, freq=(0, 1), time=(2, 2))  # Reverse the trimming
    
    # iSTFT (same params as the forward STFT; see the normalization caveat above)
    target_len = HOP * ceil(samples / HOP) + 2 * pad_amount
    freq_audio = istft(z, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                       window=hann, normalized=True, center=True, length=target_len)
    
    # Trim Demucs padding
    freq_audio = freq_audio[..., pad_amount:pad_amount+samples]
    
    # Combine branches
    output[s] = freq_audio + out_wave[s]
```

## Embedded Metadata

The ONNX model includes metadata you can read at runtime:

```python
import onnx
import json

model = onnx.load("htdemucs.onnx")
metadata = {prop.key: prop.value for prop in model.metadata_props}

sources = json.loads(metadata["sources"])         # ["drums", "bass", "other", "vocals"]
sample_rate = int(metadata["sample_rate"])        # 44100
audio_channels = int(metadata["audio_channels"])  # 2
precision = metadata["precision"]                 # "fp32" or "fp16"
```

## RoFormer models

RoFormer models (`bs_roformer_sw`, `melband_roformer_kim`, …) export through the same command and the same STFT-outside-the-graph boundary, with these differences:

```bash
unblend export-onnx --model bs_roformer_sw --fp16
```

- **Exporter/opset:** exported with the dynamo exporter at opset ≥ 18 (the legacy exporter emits inconsistent shape metadata for the per-band mask heads, which onnxruntime rejects). `--opset` values below 18 are raised automatically.
- **Interface:** inputs `spec_real`/`spec_imag` `[B, C, F, T]` only — there is no `audio` input and no `out_wave` output (RoFormers are pure spectrogram maskers; skip the time-branch combine step entirely). Outputs `out_spec_real`/`out_spec_imag` are `[B, S, C, F, T]`.
- **Generic consumers should read the embedded STFT metadata** (`stft_n_fft`, `stft_hop_length`, `stft_win_length`, `stft_normalized` — typically 2048/512/2048/false). The current `unblend` npm runtime instead maintains and validates a hard-coded per-model registry; it does not consume ONNX metadata at runtime, so releases must keep that registry in lockstep with exported artifacts. `stft_normalized` is `false` for the shipped checkpoints, so the `sqrt(n_fft)` normalization caveat above does **not** apply: run a plain centered Hann STFT/iSTFT with no extra scaling and no Demucs-style pre-padding/trimming.
- **Feed exactly `segment_samples` per call** (from metadata; e.g. 588800 ≈ 13.35 s for `bs_roformer_sw`). Only the batch axis is dynamic.
- **Single-mask checkpoints** (metadata `output_complement: "true"`, e.g. `melband_roformer_kim`) emit one stem; compute the second client-side as `mixture - stem` after the iSTFT.
- **`--static-batch`** traces with a fixed batch=1 instead of a dynamic batch axis (metadata `batch_mode: "static"` vs `"dynamic"`). Use this for single-inference consumers; the default dynamic-batch export remains best for anyone batching multiple segments through the raw ONNX file.
- Extra metadata keys: `model_family` (`"roformer"`), `architecture` (`bs_roformer` / `mel_band_roformer`), `num_stems`, `output_complement`, `segment_samples`, `batch_mode`, the four `stft_*` keys, and `license` (`unlicensed` for BS-RoFormer-SW and `MIT` for Mel-Band RoFormer Kim).

> **Browser export details.** The RoFormer wrapper evaluates exact self-attention in bounded head groups and query chunks, and evaluates feed-forward expansion in bounded feature groups. This prevents full Q/K/V projections, 1.2–2.6 GB score matrices, and 200+ MB MLP activations from exhausting browser memory. It also emits per-band `Slice` operations and binary joins instead of wide `Split`/`Concat` dispatches, keeping WebGPU shader bindings within portable limits. RoFormer `--fp16` exports use mixed precision: projections, MLPs, masks, activations, and weights are fp16 while IO, normalization, rotary trig, and softmax remain fp32. Use `--static-batch` for browser artifacts; dynamic-batch exports remain available for native/server consumers.

## SCNet

SCNet exports through the same client-side-transform pattern as the RoFormers:
the graph covers everything between the STFT and iSTFT, and the caller supplies
the spectrogram via `compute_scnet_stft_for_export`.

Four SCNet-specific details matter:

- **The audio must be padded first.** SCNet's trunk takes a real FFT along the
  time axis, which needs an even frame count, so the model pads its input up to
  a hop boundary (and by one further hop if that lands on an odd count). The
  graph is traced at the *padded* frame count and will reject a spectrogram
  computed from unpadded audio. Use `unblend.scnet.stft_padding(samples,
  hop_length)` — the same function the model and exporter call — then trim that
  many samples off the end after the inverse transform. For `scnet_small` this
  is 485100 → 486400 samples, i.e. 476 frames rather than 474. The npm runtime
  keeps those lengths separate: it overlaps logical 485100-sample chunks,
  appends 1300 zeros to each model input, and trims the same tail after iSTFT.
- **The window depends on the variant.** Plain SCNet passes no `window` to
  `torch.stft`, so a Hann window would silently change the result; the masked
  variants use a periodic Hann. The export records which in
  `unblend.stft_window`: `hann` for `scnet_small` and `none` for
  `scnet_xl_wide_v5`. `compute_scnet_stft_for_export` takes a matching
  `window` argument — read the metadata rather than assuming either.
- **The masking head is inside the graph.** The masked variants add a frequency
  positional embedding before the trunk and multiply a predicted complex mask
  against the mixture afterwards. Both are traced, so the client contract is
  identical for either variant: STFT in, per-stem spectrograms out.
- **`irfft` cannot be exported directly.** SCNet's trunk applies an inverse
  real FFT in every other dual-path layer, and that lowers to an ONNX `DFT`
  node carrying both `inverse` and `onesided`, a combination the spec forbids
  and onnxruntime rejects at load. The exporter enables `onnx_safe` on each
  `FeatureConversion`, substituting an algebraically identical real-valued DFT
  expressed as two matmuls.

Known caveat: on production-size checkpoints the ONNX graph differs from the
PyTorch result by a fraction of a percent relative. Against one real 11-second
segment, the two registered browser fp16 artifacts measured 0.058%–0.512%
relative L2 error. The DFT substitution itself
accounts for under 1e-5; most of the accumulated difference comes from the
trunk's repeated bidirectional LSTM operations. A two-layer test model matches
to 6e-08.
