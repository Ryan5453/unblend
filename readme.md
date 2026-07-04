# demucs-next

`demucs-next` is a fork of the [Demucs](https://github.com/adefossez/demucs) reference implementation, updated for modern Python, PyTorch, and TorchCodec. It runs up to 5.2x faster than upstream at equal quality (more when extracting a single stem from `htdemucs_ft`, which skips the other specialist sub-models), and is easier to install and run.


<details>
<summary>Benchmarks and SDR comparisons</summary>

50 tracks of [MUSDB18-HQ](https://zenodo.org/records/3338373), `htdemucs`, `shifts=1`, `split_overlap=0.25`. Steady-state mean seconds per track; each config measured in a fresh process. Reproduce with `python benchmark.py --musdb-root /path/to/musdb18hq/test --include-upstream`.

### `demucs-next`

| Hardware | Backend | s/track | Mean SDR |
|---|---|---:|---:|
| RTX A4000 | CUDA FP16 + compile | 1.33 | 8.358 |
| RTX A4000 | CUDA BF16 + compile | 1.34 | 8.355 |
| RTX A4000 | CUDA BF16 | 1.97 | 8.349 |
| RTX A4000 | CUDA FP16 (default) | 1.98 | 8.358 |
| RTX A4000 | CUDA FP32 + compile | 2.56 | 8.359 |
| RTX A4000 | CUDA FP32 | 3.49 | 8.359 |
| M2 Max | MPS FP16 (default) | 5.78 | 8.380 |
| M2 Max | MPS BF16 | 7.39 | 8.373 |
| M2 Max | MPS FP32 | 11.97 | 8.381 |
| M2 Max | Browser ONNX FP16 (WebGPU) | 18.64 | 8.399 |
| M2 Max | Browser ONNX FP32 (WebGPU) | 18.82 | 8.399 |
| Intel i9-10900X | CPU FP32 | 45.70 | 8.381 |
| M2 Max | CPU FP32 | 84.50 | 8.381 |

### [demucs reference](https://github.com/adefossez/demucs)

| Hardware | Backend | s/track | Mean SDR |
|---|---|---:|---:|
| RTX A4000 | CUDA FP32 | 6.93 | 8.357 |
| M2 Max | MPS FP32 | 8.77 | 8.387 |
| Intel i9-10900X | CPU FP32 | 48.51 | 8.350 |
| M2 Max | CPU FP32 | 102.38 | 8.294 |

</details>

## Installation

### Prerequisites

Before installing Demucs, make sure your system has:

- FFmpeg v4+ available in your `PATH`
- [`uv`](https://docs.astral.sh/uv/#installation)
- Optionally, a working C/C++ compiler such as `g++` if you plan to use `--compile`

### Temporary Installation using UV

With UV, you can use the `uvx` command to run Demucs without installing it permanently on your system. This sets up a temporary virtual environment for the duration of the command. 

```bash
uvx --from demucs-next demucs separate audio_file.mp3
```

The PyPI package is named `demucs-next`, but the installed CLI command is `demucs`, so `uvx` needs `--from demucs-next` to map the two.

**Note**: Demucs does not specify a specific PyTorch wheel. This means that GPUs will only work on Apple Silicon or PyTorch's default CUDA version (currently 12.8) on Linux when using uvx. Demucs will fall back to CPU if one of the above conditions are not met. (This uvx default is independent of the Cog/Replicate build, which pins CUDA 12.4 in `cog.yaml`.)

### Install using UV

Create a virtual environment backed by a `uv`-managed Python:

```bash
uv python install 3.12
uv venv --managed-python --python 3.12
source .venv/bin/activate
```

Using a `uv`-managed Python is recommended because it will include the Python headers needed by PyTorch / Triton.

Then install Demucs into that environment:

```bash
uv pip install demucs-next --torch-backend=auto
```

The `--torch-backend=auto` flag automatically detects your GPU and installs the appropriate version of PyTorch compatible with your system.

## CLI Usage

After installing Demucs, you can use it like the following:

```bash
# View separation options
demucs separate --help

# Separate one audio file
demucs separate audio_file.mp3

# Separate multiple audio files
demucs separate audio_file_1.mp3 audio_file_2.mp3

# Separate every audio file in a directory tree (recurses into subdirectories;
# dotfiles and dot-directories are skipped).
demucs separate /path/to/music/folder
```

## Python API Usage

Demucs provides a Python API for separating audio files. Please refer to the [API docs](api.md) for more information.

## ONNX & Browser Usage

Demucs can also run in the browser via ONNX. See the [ONNX export notes](onnx.md) and the [`demucs-next` npm package docs](web/demucs/README.md) for details.

## Cog Usage

Demucs provides a [Cog](https://github.com/replicate/cog), which allows you to easily deploy a Demucs model as a REST API. You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

## Changelog

The [changelog](changelog.md) contains information about the changes between versions of demucs-next.
