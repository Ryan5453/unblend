## Unreleased

### Added

- Native CUDA backend (`unblend.cuda`): fused FP16/BF16 kernels for the
  GroupNorm/GELU/GLU chains in HTDemucs and SCNet, a fused DConv envelope,
  channel-last variants for `channels_last` inputs, and a fused interleaved
  rotary-position kernel for BS-RoFormer / Mel-Band RoFormer.
  Activated automatically for eager FP16/BF16 inference on CUDA; models with
  nothing eligible skip extension compilation. Requires nvcc matching the
  installed torch wheel; without it, falls back to native PyTorch ops.
- Custom-kernel kill switch: `Separator(custom_kernels=False)` (or the CLI's
  `--native-ops`, or `UNBLEND_CUSTOM_KERNELS=0`) forces vanilla PyTorch ops on
  every device — fused Metal shaders off on MPS, CUDA kernels and the fused
  rotary path off on CUDA — as the baseline for A/B benchmarks. It also skips
  the one-time nvcc build entirely.
- The one-time CUDA kernel compilation now overlaps the checkpoint download:
  eligible loads warm the extension in a background thread while weights
  transfer, so a fresh environment pays roughly max(download, build) instead
  of the sum. A warning names the stall when a build blocks synchronously.

### Changed

- Single-checkpoint backends (RoFormer, SCNet) now download, cache-report,
  and remove through one generic path instead of a RoFormer-only special case.

### Performance

- MUSDB18-HQ paired A/B at SDR parity (≤0.013 dB): SCNet xl-wide 1.8×
  (A100) / 2.3× (H200); SCNet small 1.4×; HTDemucs/HTDemucs-ft up to 1.5×;
  BS-Roformer & Mel-Band 1.2×. Model-core speedups reach 4× on SCNet-xl.

## v1.0.0

This is the first release of Unblend. 
Please view the [Python API](https://github.com/Ryan5453/unblend/blob/main/api.md) docs, [npm package](https://github.com/Ryan5453/unblend/blob/main/web/demucs/README.md) docs, or the [ONNX export notes](https://github.com/Ryan5453/unblend/blob/main/onnx.md) for more information on how to embed Unblend in your application. 
