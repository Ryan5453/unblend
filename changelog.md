## v1.0.0

The first release of Unblend, a music source separation inference library that began as a fork of [Demucs](https://github.com/facebookresearch/demucs).

- Four architectures behind one API: HTDemucs, BS-RoFormer, Mel-Band RoFormer and SCNet, with ten registered models including two cross-architecture ensembles. See the [readme](https://github.com/Ryan5453/unblend#models) for the list and each model's weight license.
- `unblend` CLI: `separate`, `tune`, `export-onnx`, `version`, and `models list / info / download / import / unregister / remove`.
- Python API ([api.md](https://github.com/Ryan5453/unblend/blob/main/api.md)), ONNX export ([onnx.md](https://github.com/Ryan5453/unblend/blob/main/onnx.md)), the `unblend` npm package for in-browser separation ([README](https://github.com/Ryan5453/unblend/blob/main/web/unblend/README.md)), and a Cog predictor.
- Runs on CPU, CUDA (FP16 with fused kernels; `torch.compile` for long jobs) and Apple Silicon (FP16, fused Metal kernels). Python 3.10–3.14.
- Custom and community checkpoints can be added through a models file, and `unblend models import` converts training checkpoints to Safetensors.

Known limitations: ensembles and multi-checkpoint models such as `htdemucs_ft` can't be exported to ONNX; HTDemucs ONNX export uses PyTorch's TorchScript exporter, which PyTorch has deprecated; Intel Macs aren't supported (no PyTorch wheels); Windows isn't tested in CI.
