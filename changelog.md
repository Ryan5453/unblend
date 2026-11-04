## v1.0.1

- Browser: on WebGPU without the `shader-f16` feature, the RoFormers' fp16 files used to load and then return garbage stems. They now reject at load with a `'session'`-stage `ModelLoadError`, except `melband_roformer_kim`, which falls back to WASM. A WebGPU error during a separation, such as running out of GPU memory, now fails the run instead of returning its outputs. The web app loads the RoFormers' fp32 files on such GPUs.
- Requires Python 3.11–3.14. Python 3.10 reached end of life in October 2026, and the patched filelock below needs 3.11.
- Requires filelock 4.2.1 or later, which fixes GHSA-69hj-f68j-2879 and GHSA-8mx5-7m5m-r77g.
- Cog image dependencies updated: certifi 2026.10.19, filelock 4.2.7, fsspec 2026.6.0 and setuptools 84.1.0.

## v1.0.0

The first release of Unblend, a music source separation inference library that began as a fork of [Demucs](https://github.com/facebookresearch/demucs).

- Four architectures behind one API: HTDemucs, BS-RoFormer, Mel-Band RoFormer and SCNet, with ten registered models including two cross-architecture ensembles. See the [readme](https://github.com/Ryan5453/unblend#models) for the list and each model's weight license.
- `unblend` CLI: `separate`, `tune`, `export-onnx`, `version`, and `models list / info / download / import / unregister / remove`.
- Python API ([api.md](https://github.com/Ryan5453/unblend/blob/main/api.md)), ONNX export ([onnx.md](https://github.com/Ryan5453/unblend/blob/main/onnx.md)), the `unblend` npm package for in-browser separation ([README](https://github.com/Ryan5453/unblend/blob/main/web/unblend/README.md)), and a Cog predictor.
- Runs on CPU, CUDA (FP16 with fused kernels; `torch.compile` for long jobs) and Apple Silicon (FP16, fused Metal kernels). Python 3.10–3.14.
- Custom and community checkpoints can be added through a models file, and `unblend models import` converts training checkpoints to Safetensors.

Known limitations: ensembles and multi-checkpoint models such as `htdemucs_ft` can't be exported to ONNX; HTDemucs ONNX export uses PyTorch's TorchScript exporter, which PyTorch has deprecated; Intel Macs aren't supported (no PyTorch wheels); Windows isn't tested in CI.
