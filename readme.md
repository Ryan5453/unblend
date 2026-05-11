# demucs-next

> [!WARNING]
> `demucs-next` is still in alpha not recommended for production use.

Demucs is a SOTA music source separation model capable of separating drums, bass, and vocals from the rest of the accompaniment.
This is a fork of the [author's fork](https://github.com/adefossez/demucs) of the original Demucs repository.

`demucs-next` has been updated to use modern versions of Python, PyTorch, and TorchCodec. It is significantly faster and easier to use than upstream Demucs.

## Installation

### Prerequisites

Before installing Demucs, make sure your system has:

- FFmpeg v4+ available in your `PATH`
- [`uv`](https://docs.astral.sh/uv/#installation)
- Optionally, a working C/C++ compiler such as `g++` if you plan to use `--compile`

### Temporary Installation using UV

With UV, you can use the `uvx` command to run Demucs without installing it permanently on your system. This sets up a temporary virtual enviornment for the duration of the command. 

```bash
uvx demucs-next separate audio_file.mp3
```

**Note**: Demucs does not specify a specific PyTorch wheel. This means that GPUs will only work on Apple Silicon or PyTorch's default CUDA version (currently 12.8) on Linux when using uvx. Demucs will fall back to CPU if one of the above conditions are not met.

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

## Usage

After installing Demucs, you can use it like the following:

```bash
# View separation options
demucs separate --help

# Separate one audio file
demucs separate audio_file.mp3

# Separate multiple audio files
demucs separate audio_file_1.mp3 audio_file_2.mp3

# Separate all audio files in a directory
demucs separate /path/to/music/folder
```

## Performance

50 tracks of [MUSDB18-HQ](https://zenodo.org/records/3338373), `htdemucs`, `shifts=1`, `split_overlap=0.25`. Steady-state mean seconds per track. Reproduce with `python benchmark.py --include-upstream`.

### `demucs-next`

| Hardware | Backend | s/track | Mean SDR |
|---|---|---:|---:|
| RTX A4000 | CUDA FP16 | 2.85 | 8.297 |
| RTX A4000 | CUDA FP32 | 4.42 | 8.297 |
| M2 Max | MPS FP16 | 8.65 | 8.318 |
| M2 Max | MPS FP32 | 12.74 | 8.318 |
| M2 Max | Browser ONNX FP16* (WebGPU) | 18.64 | 8.399 |
| M2 Max | Browser ONNX FP32 (WebGPU) | 18.82 | 8.399 |
| Intel i9-10900X | CPU FP32 | 49.94 | 8.318 |
| M2 Max | CPU FP32 | 86.48 | 8.318 |

### Upstream demucs (`main`, `4.1.0a3`)

| Hardware | Backend | s/track | Mean SDR |
|---|---|---:|---:|
| RTX A4000 | CUDA FP32 | 7.63 | 8.246 |
| M2 Max | MPS FP32 | 10.05 | 8.288 |
| Intel i9-10900X | CPU FP32 | 54.26 | 8.299 |
| M2 Max | CPU FP32 | 110.99 | 8.306 |

`demucs-next` is faster than upstream on every config except M2 Max MPS FP32 (a torch 2.7 -> 2.8 `aten::copy_` regression we can't avoid; the recommended MPS FP16 path beats it). SDR equals or exceeds upstream everywhere.

\* Browser ONNX FP16 is weight-only FP16 — Conv/MatMul weights are stored as FP16 on disk (cutting the download from 161 MB to 87 MB), but a Cast node restores FP32 at session-create so compute runs in full FP32. ORT-WASM does not accumulate FP16 GEMMs in FP32 the way CUDA/MPS do, so a real FP16 graph produced audible quantization noise; this approach gets the size win without the precision cost. SDR is bit-equivalent to FP32.

## Cog Usage

Demucs provides a [Cog](https://github.com/replicate/cog), which allows you to easily deploy a Demucs model as a REST API. You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

## API Usage

Demucs provides a Python API for separating audio files. Please refer to the [API docs](docs/api.md) for more information.

## Changelog

The [changelog](docs/changelog.md) contains information about the changes between versions of demucs-next.
