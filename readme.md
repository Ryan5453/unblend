# <img src="https://raw.githubusercontent.com/Ryan5453/unblend/main/web/app/public/favicon.svg" width="30"> Unblend

Unblend is a music source separation inference library designed to be blazing fast and easy-to-use.
It implements one consistent API across four supported model architectures: HTDemucs, BS-RoFormer, Mel-Band RoFormer, and SCNet.

## Installation

### Prerequisites

- FFmpeg **v4–v8** available in your `PATH`. Decoding goes through `torchcodec`,
  which ships one loader per FFmpeg major and currently covers 4 through 8.
  FFmpeg 9 is **not** supported yet and fails at import, not at runtime — which
  matters on macOS, where `brew install ffmpeg` now installs 9. Use
  `brew install ffmpeg@7` there until `torchcodec` adds a 9 loader.
- [`uv`](https://docs.astral.sh/uv/#installation)
- Optional: C/C++ compiler such as GCC, Clang, or MSVC - enables torch.compile support
- Optional: NVCC (NVIDIA CUDA Compiler) - enables custom CUDA kernels

### Install using uv

Create a virtual environment backed by a uv-managed Python:

```bash
uv python install 3.12
uv venv --managed-python --python 3.12
source .venv/bin/activate
```

Then install Unblend into that environment:

```bash
uv pip install unblend --torch-backend=auto
```

### Temporary Installation

With uv, you can use the `uvx` command to run Unblend without installing it permanently on your system.

```bash
uvx unblend separate audio_file.mp3
```

Note: Unblend does not specify a specific PyTorch wheel. This means that GPUs will only work on Apple Silicon or PyTorch's default CUDA version on Linux when using uvx.


## CLI Usage

After installing Unblend, you can use it like the following:

```bash
$ unblend --help
$ unblend separate audio_file.mp3
$ unblend separate audio_file_1.mp3 audio_file_2.mp3
```

## API Usage

Unblend has various ways you can programatically use its API:
- [ONNX export](https://github.com/Ryan5453/unblend/blob/main/onnx.md) 
- [Python API](https://github.com/Ryan5453/unblend/blob/main/api.md)
- [Browser API](https://github.com/Ryan5453/unblend/blob/main/web/unblend/README.md) 
- [Cog](https://github.com/Ryan5453/unblend/blob/main/cog.yaml) (hosted on [Replicate](https://replicate.com/ryan5453/demucs))
