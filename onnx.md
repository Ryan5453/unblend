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

`--precision` sets how weights are *stored* in the exported graph, not what it computes in. It is unrelated to `unblend separate --precision`, which casts the model and genuinely does compute in fp16; here narrowed weights are cast back to fp32 before every op, so only the download shrinks. The default `native` uses the dtype the checkpoint's Safetensors header declares: fp16 for HTDemucs, because upstream rounded those weights at release, and fp32 for RoFormer and SCNet. That halves the HTDemucs exports (169 MB to 85 MB) with bit-identical weights.

`fp32`, `bf16`, `fp16`, `fp8_e5m2`, and `fp8_e4m3` force the choice. `fp32` is the escape hatch for runtimes that mishandle narrow initializers; narrowing below what a checkpoint carries costs real quality (fp8 on HTDemucs loses ~0.7 dB SDR, mostly on drums). The mechanism is a low-precision initializer plus a `Cast` in front of every Conv, MatMul, Gemm, and recurrent op — browser runtimes produce audible noise when fp16 accumulates in those kernels.

The exception is RoFormer at `fp16`, which converts arithmetic too, through `onnxconverter-common`: its attention trunk tolerates fp16 where convolutional stacks do not, and `Clip`, `Cos`, `Reciprocal`, `ReduceMean`, `Sin`, `Softmax`, and `Sqrt` are held at fp32. Because both cases are labelled "fp16", read `weight_precision` and `compute_precision` from the metadata rather than `precision`, which reports storage only.

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
weights = metadata["weight_precision"]         # "fp16" or "fp32"
compute = metadata["compute_precision"]        # "fp16" only for mixed-precision RoFormer
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
