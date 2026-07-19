# Changelog

## 1.0.0

This is the first release of unblend (formerly `demucs-next`). The Python API is drastically different from the original Demucs API and will likely require significant refactoring for codebases that use the original API. Please read the [API documentation](https://github.com/Ryan5453/unblend/blob/main/api.md) for more information.

Highlights:

- Renamed from `demucs-next` to `unblend`: the distribution, module, and CLI are all `unblend` (`demucs` and `demucs-inference` remain as compatibility CLI aliases).
- New RoFormer backend: BS-RoFormer and Mel-Band RoFormer run through the same `Separator` API and chunked-inference engine as Demucs, with no new dependencies. Ships `bs_roformer_sw` (6-stem) and `melband_roformer_kim` (vocals).
- Apple-silicon RoFormer inference now uses measured MPS attention and fused RMSNorm paths (1.25–1.32× end-to-end over native PyTorch on an M2 Max), supports MPS complex-STFT reconstruction, and defaults to validated SDR-equal FP16.
- Model download and availability checks support RoFormer's single-checkpoint registry entries as well as Demucs layer lists.
- CUDA `torch.compile` targets the heavy neural-network core of every shipped architecture, including both RoFormers, while leaving DSP/reconstruction eager (V100 FP16: 1.50× SW / 1.34× Kim on a 76-second track). The CLI automatically enables it only when metadata-derived chunks × the runtime eager probe exceed a conservative architecture+dtype GPU-seconds threshold; explicit flags override.
- Progress callbacks work for single and list-input separation with aggregate totals and per-input chunk metadata.
- Model weights are license-labeled in `unblend models list` and the registry metadata: the Demucs weights are unlicensed (the code is MIT; the released weights carry no grant), and the shipped RoFormer checkpoints are CC-BY-NC-SA-4.0 (non-commercial).
- ONNX export (`unblend export-onnx`) supports the RoFormer models, and the `demucs-next` npm package runs them in-browser (WebGPU/WASM) alongside HTDemucs.
