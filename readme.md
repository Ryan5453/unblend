# Unblend

Unblend is a music source separation inference library designed to be blazing fast and easy-to-use.
It implements one consistent API across the four supported model architectures: HTDemucs, BS-RoFormer, Mel-Band RoFormer, and SCNet.

## Installation

### Prerequisites

- FFmpeg v4+ available in your `PATH`
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

After installing unblend, you can use it like the following:

```bash
# View separation options
unblend separate --help

# Separate one audio file
unblend separate audio_file.mp3

# Separate multiple audio files
unblend separate audio_file_1.mp3 audio_file_2.mp3

# Separate every audio file in a directory tree
unblend separate /path/to/music/folder

# Ensemble two models, blended however you like
unblend separate --model roformer_vocals_ensemble --combine min_fft audio_file.mp3

# Bring in a checkpoint from elsewhere (verified, then registered)
unblend models import model.ckpt --config config.yaml --name my_model
```

## Unblend API

Unblend provides a [Python API](https://github.com/Ryan5453/unblend/blob/main/api.md) for integrating source separation into your own application. 
Additionally, there is a [Cog](https://github.com/replicate/cog) for HTDemucs which allows you to easily deploy it as a REST API. 
You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

Unblend can also run in the browser via ONNX.
See the [ONNX export notes](https://github.com/Ryan5453/unblend/blob/main/onnx.md) and the [npm package docs](https://github.com/Ryan5453/unblend/blob/main/web/demucs/README.md) for details.
