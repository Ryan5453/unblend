# ONNX Export

Unblend can export its models to ONNX for deployment in browsers, mobile apps, or other runtimes. This is how the [un/blend web app](https://demucs.app) runs separation in-browser.

The CLI can export any single-checkpoint model (ensembles are not currently supported) like this:

```bash
$ unblend export-onnx --model htdemucs
```

| Flag | Meaning |
|---|---|
| `-m/--model` | Model name (default `htdemucs`) |
| `-o/--output` | Output path (default `{model}_{precision}.onnx`, `_static` with `--static-batch`) |
| `--precision` | `native` (default), `fp16`, or `fp32` (see [Precision](#precision)) |
| `--opset` | Opset version (raised to 18 for RoFormer and SCNet) |
| `--static-batch` | Trace a fixed batch of 1 instead of a dynamic batch axis |

## Precision

Three things get called "precision" here, and they move independently:

- **Compute.** Every model in the registry was trained in fp32 and every architecture builds its modules in fp32. Nothing here is natively fp16.
- **Storage.** The HTDemucs checkpoints (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`) ship fp16 on disk; the RoFormer and SCNet checkpoints ship fp32.
- **Runtime.** The PyTorch path picks fp16 on MPS and on CUDA with tensor cores, fp32 on CPU. That choice never reaches the ONNX graph.

The fp16 HTDemucs checkpoints are upstream's decision, not ours. Demucs' `serialize_model` defaults to `half=True`, so Defossez rounded the weights to fp16 once at release to halve the download; loading widens them back to fp32 because the module is fp32. Unblend preserves the checkpoint exactly and does the same widening. The practical consequence is that HTDemucs weights are fp32 containers holding fp16-precision values, and exporting them as fp32 writes out 84 MB of mantissa bits that are all zero.

### What `--precision` does

`native`, the default, stores weights at whatever precision the checkpoint actually carries. It resolves per model rather than per family, by checking whether every weight survives a round trip through fp16:

| Model | Native | Default export |
|---|---|---|
| `htdemucs`, `htdemucs_6s` | fp16 | 85 MB (from 169 MB), 56 MB (from 110 MB) |
| RoFormer, SCNet | fp32 | unchanged |

For HTDemucs this is free. The rewrite reaches 250 of 533 initializers but 99.7% of the weight bytes, every converted tensor is bit-identical to the checkpoint, and end-to-end output moves by at most 1.7e-6 relative — float-association noise from the inserted `Cast` nodes, roughly 115 dB down, not lost precision. For RoFormer and SCNet, whose checkpoints are genuinely fp32, `native` changes nothing.

`fp16` forces the conversion, which does discard real precision on RoFormer and SCNet. `fp32` forces full width, which is the escape hatch for runtimes that mishandle fp16 initializers.

Forcing `fp16` means different things per family:

| Family | Weights | Compute | IO |
|---|---|---|---|
| HTDemucs, SCNet | fp16 | fp32 | fp32 |
| RoFormer | fp16 | fp16 | fp32 |

HTDemucs and SCNet get a weight-only rewrite: fp16 initializers with a `Cast` back to fp32 in front of every Conv, ConvTranspose, MatMul, Gemm, and recurrent op. Downloads halve, arithmetic is untouched. This matters for browser runtimes, where fp16 accumulation in Conv/MatMul kernels produces audible quantization noise; keeping compute in fp32 sidesteps that with no measurable quality loss.

RoFormer instead goes through `onnxconverter-common`'s float16 converter and computes in fp16, keeping only the graph inputs and outputs fp32. Its attention trunk tolerates that where the convolutional stacks do not, and the numerically sensitive ops (`Clip`, `Cos`, `Reciprocal`, `ReduceMean`, `Sin`, `Softmax`, `Sqrt`) are blocked from conversion and stay fp32. This needs the `onnx` extra installed.

Because those two cases both amount to "fp16", read `weight_precision` and `compute_precision` from the metadata rather than the older `precision` key, which reports storage only and cannot tell them apart.

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
