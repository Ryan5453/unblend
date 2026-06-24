# Changelog

## 1.0.0 (2026-06-12)

This is the first release of demucs-next. The Python API is drastically different from the original API and will likely require significant refactoring for codebases that use the original API. Please read the [API documentation](api.md) for more information.

Highlights compared to upstream demucs:

- Modernized stack: Python 3.10–3.13, PyTorch 2.9+, TorchCodec for audio I/O (no more soundfile/torchaudio backends).
- Up to 5.2x faster separation via `torch.compile` + CUDAGraphs on CUDA, FP16/BF16 inference, custom Metal kernels on Apple Silicon, and batched chunk processing with auto-detected batch size.
- New `Separator` / `SeparatedSources` Python API with batched multi-file separation, progress callbacks, and single-stem loading (`only_load`) for `htdemucs_ft`.
- Rewritten Typer-based CLI (`demucs separate`, `demucs models`) with output templates and collision detection.
- ONNX export (`demucs export-onnx`) and a browser pipeline ([npm package](web/demucs/README.md)) running on ONNX Runtime Web / WebGPU.
- Only the hybrid transformer models (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`) are shipped; legacy wave-U-Net/MDX models were removed.

### CUDA performance

CUDA inference is 2-3x faster than it was earlier in the 1.0.0 pre-release cycle, at unchanged MUSDB18 SDR:

- `Separator` now defaults to `dtype="auto"`, which picks FP16 on CUDA GPUs with tensor cores (compute capability ≥ 7.0). FP16 measures SDR-identical to FP32 on MUSDB18 while running ~1.7x faster. Older CUDA GPUs and CPU stay at FP32; pass `dtype=None` or `torch.float32` to opt out.
- The separation pipeline is now GPU-resident for inputs that fit a conservative VRAM budget (the normal case for songs): the waveform moves to the GPU once, normalisation / chunk slicing / overlap-add accumulation / un-normalisation all run on-GPU, and the stems make a single GPU→CPU trip at the end. Longer inputs automatically fall back to the previous bounded CPU-accumulation path, so arbitrarily long audio still can't OOM the GPU.
- `cudnn.benchmark` autotuning is now only enabled for the `compile=True` path (where there is exactly one batch shape to tune). Eager inference uses cuDNN heuristics — measured within ~1% of autotuned speed — which removes a multi-second exhaustive search that re-ran for nearly every track (tail batches vary in shape per input).
- Eager (non-compiled) runs no longer zero-pad tail batches up to `chunk_batch_size`, and the auto-detected `chunk_batch_size` is sized for eager mode's real memory needs instead of reusing the CUDAGraphs reservation factor (roughly 2x larger batches on the same GPU).
- Plain 16-bit PCM WAV files decode through a direct header-parse + memcpy fast path (~2x faster than the FFmpeg demux pipeline, sample-exact); every other format still goes through torchcodec.
