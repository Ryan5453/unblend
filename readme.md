# unblend

`unblend` is a music source separation library with one `Separator` API over multiple model families: an optimized fork of [Demucs](https://github.com/adefossez/demucs), plus BS-RoFormer and Mel-Band RoFormer community checkpoints (`bs_roformer_sw`, `melband_roformer_kim`). The Demucs backend runs ~2.5× faster than upstream like-for-like (FP32), up to ~6× with FP16 + `torch.compile`; single-stem specialist loading reaches 19–23× by skipping unused models. Run `unblend models list` for stems and weight licenses (RoFormer weights are CC-BY-NC-SA-4.0, non-commercial; Demucs weights carry no license grant).


<details>
<summary>Benchmarks and SDR comparisons</summary>

All numbers: [MUSDB18-HQ](https://zenodo.org/records/3338373) test set (50 tracks), one V100-SXM2 32GB, `shifts=1`, `split_overlap=0.25` unless noted. Upstream is `adefossez/demucs@main` on torch 2.1.2 (the newest it supports), same GPU. Steady-state mean seconds per track, fresh process per config. Reproduce with `python benchmark.py --musdb-root /path/to/musdb18hq/test --include-upstream`.

### Versus the reference (FP32, like for like)

| Model | reference s/track | unblend s/track | Speedup | SDR (unblend / ref) |
|---|---:|---:|---:|---|
| htdemucs | 5.91 | 2.35 | 2.5x | 8.380 / 8.377 |
| htdemucs_ft | 22.96 | 8.62 | 2.7x | 8.535 / 8.535 |
| htdemucs_6s | 5.64 | 2.20 | 2.6x | 6.827 / 6.831 |

### Configurations (htdemucs)

| Backend | s/track | Mean SDR |
|---|---:|---:|
| CUDA FP16 + compile | 0.94 | 8.380 |
| CUDA FP16 (default) | 1.24 | 8.380 |
| CUDA FP32 + compile | 1.88 | 8.380 |
| CUDA FP32 | 2.18 | 8.380 |
| CUDA BF16 | 9.17 | 8.371 |
| reference, CUDA FP32 | 4.67 | 8.377 |

BF16 has no native support on V100 (Volta); on Ampere and newer it tracks FP16 speed. `--compile` is SDR-neutral (measured delta +0.000001 dB).

### Single-stem extraction (htdemucs_ft, vocals)

| Config | s/track | vs reference |
|---|---:|---:|
| reference FP32 (always runs all 4 sub-models) | 22.96 | 1x |
| unblend FP32 | 8.62 | 2.7x |
| + vocals specialist only (`--isolate-stem vocals`) | 2.37 | 9.7x |
| + FP16 + `--compile` | 1.20 | 19.2x |
| + batched library processing (`separate([...])`) | 0.97 | 23.6x |

Vocals SDR is unchanged throughout: 8.991 specialist-only vs 8.994 full ensemble.

### Quality/speed knobs (htdemucs FP32)

| shifts | overlap | Mean SDR | s/track |
|---:|---:|---:|---:|
| 1 | 0.1 | 8.343 | 2.15 |
| 1 | 0.25 (default) | 8.380 | 2.35 |
| 1 | 0.5 | 8.413 | 3.44 |
| 2 | 0.25 | 8.428 | 4.78 |
| 4 | 0.25 | 8.471 | 8.91 |

The defaults sit at the knee: everything above them buys hundredths of a dB at 1.5-3.8x the runtime.

### Memory

Peak process RSS, htdemucs_ft: 3.3 GB vs 4.6 GB for the reference. Compiled ensembles trade VRAM for speed — each sub-model holds a CUDAGraph capture pool for the process lifetime (~27.7 GB device-truth for compiled `htdemucs_ft`, which `torch.cuda.max_memory_allocated` cannot see) — so on ≤32 GB cards, batch a library *or* compile an ensemble, not both. See the [API docs](https://github.com/Ryan5453/unblend/blob/main/api.md) `compile` notes.

### Apple silicon and CPU (M2 Max; browser rows pending)

| Hardware | Backend | s/track | Mean SDR |
|---|---|---:|---:|
| M2 Max | MPS FP16 (default) | 5.10 | 8.380 |
| M2 Max | MPS BF16 | 6.50 | 8.373 |
| M2 Max | MPS FP32 | 13.52 | 8.380 |
| M2 Max | Browser ONNX FP16 (WebGPU) | X | X |
| M2 Max | Browser ONNX FP32 (WebGPU) | X | X |
| M2 Max | CPU FP32 | 86.50 | 8.380 |
| M2 Max | reference, MPS FP32 | 8.74 | 8.364 |
| M2 Max | reference, CPU FP32 | 101.20 | 8.389 |

</details>

## Installation

### Prerequisites

Before installing unblend, make sure your system has:

- FFmpeg v4+ available in your `PATH`
- [`uv`](https://docs.astral.sh/uv/#installation)
- Optionally, a working C/C++ compiler such as `g++` for CUDA compilation (the CLI enables it automatically only for workloads past the estimated break-even point; failures fall back to eager, and `--no-compile` disables it)

### Temporary Installation using UV

With UV, you can use the `uvx` command to run unblend without installing it permanently on your system. This sets up a temporary virtual environment for the duration of the command.

```bash
uvx unblend separate audio_file.mp3
```

The PyPI package and the CLI command are both named `unblend`, so no `--from` mapping is needed. (A `demucs` command alias is also installed for compatibility with the former name.)

`unblend` installs the `unblend` module and command, plus `demucs` and `demucs-inference` command aliases for compatibility with the former name. The module no longer clashes with the original `demucs` package, but the `demucs` command alias does — keep them in separate environments if you need both.

**Note**: unblend does not specify a specific PyTorch wheel. This means that GPUs will only work on Apple Silicon or PyTorch's default CUDA version (currently 12.8) on Linux when using uvx. unblend will fall back to CPU if one of the above conditions are not met. (This uvx default is independent of the Cog/Replicate build, which pins CUDA 12.4 in `cog.yaml`.)

### Install using UV

Create a virtual environment backed by a `uv`-managed Python:

```bash
uv python install 3.12
uv venv --managed-python --python 3.12
source .venv/bin/activate
```

Using a `uv`-managed Python is recommended because it will include the Python headers needed by PyTorch / Triton.

Then install unblend into that environment:

```bash
uv pip install unblend --torch-backend=auto
```

The `--torch-backend=auto` flag automatically detects your GPU and installs the appropriate version of PyTorch compatible with your system.

## CLI Usage

After installing unblend, you can use it like the following:

```bash
# View separation options
unblend separate --help

# Separate one audio file
unblend separate audio_file.mp3

# Separate multiple audio files
unblend separate audio_file_1.mp3 audio_file_2.mp3

# Separate every audio file in a directory tree (recurses into subdirectories;
# dotfiles and dot-directories are skipped).
unblend separate /path/to/music/folder
```

## Python API Usage

unblend provides a Python API for separating audio files. Please refer to the [API docs](https://github.com/Ryan5453/unblend/blob/main/api.md) for more information.

## ONNX & Browser Usage

unblend can also run in the browser via ONNX. See the [ONNX export notes](https://github.com/Ryan5453/unblend/blob/main/onnx.md) and the [`demucs-next` npm package docs](https://github.com/Ryan5453/unblend/blob/main/web/demucs/README.md) for details.

## Cog Usage

unblend provides a [Cog](https://github.com/replicate/cog), which allows you to easily deploy it as a REST API. You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

## Changelog

The [changelog](https://github.com/Ryan5453/unblend/blob/main/changelog.md) contains information about the changes between versions of unblend.
