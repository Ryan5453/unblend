# ONNX Export

`unblend` includes the ability to export its models (HTDemucs and the RoFormer family) to the ONNX format for deployment in browsers, mobile, or other runtimes.
This is how the [un/blend web app](https://demucs.app) runs source separation in-browser. HTDemucs specifics are below; RoFormer differences are in [RoFormer models](#roformer-models) at the end.

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
- **STFT parameters come from the embedded metadata** (`stft_n_fft`, `stft_hop_length`, `stft_win_length`, `stft_normalized` — typically 2048/512/2048/false) rather than being fixed constants. `stft_normalized` is `false` for the shipped checkpoints, so the `sqrt(n_fft)` normalization caveat above does **not** apply: run a plain centered Hann STFT/iSTFT with no extra scaling and no Demucs-style pre-padding/trimming.
- **Feed exactly `segment_samples` per call** (from metadata; e.g. 588800 ≈ 13.35 s for `bs_roformer_sw`). Only the batch axis is dynamic.
- **Single-mask checkpoints** (metadata `output_complement: "true"`, e.g. `melband_roformer_kim`) emit one stem; compute the second client-side as `mixture - stem` after the iSTFT.
- Extra metadata keys: `model_family` (`"roformer"`), `architecture` (`bs_roformer` / `mel_band_roformer`), `num_stems`, `output_complement`, `segment_samples`, the four `stft_*` keys, and `license` (the shipped RoFormer checkpoints are CC-BY-NC-SA-4.0 — non-commercial).