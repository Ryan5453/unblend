# <img src="https://raw.githubusercontent.com/Ryan5453/unblend/main/web/app/public/favicon.svg" width="30"> Unblend

Unblend is a music source separation inference library designed to be fast and easy to use. On CUDA it runs HTDemucs up to 10x faster than upstream Demucs at equal quality (measured on an H200 with `benchmark.py`, against upstream on the PyTorch 2.1 it supports).
It implements one consistent API across four supported model architectures: HTDemucs, BS-RoFormer, Mel-Band RoFormer, and SCNet.

## Installation

### Prerequisites

- FFmpeg v4–v8 (not v9 yet; see the macOS note below)
- [`uv`](https://docs.astral.sh/uv/#installation)
- Optional: C/C++ compiler such as GCC, Clang, or MSVC - enables torch.compile support
- Optional: NVCC (NVIDIA CUDA Compiler) and [Ninja](https://ninja-build.org/) - enables custom CUDA kernels

On macOS, Homebrew's `ffmpeg` formula is now FFmpeg 9, which PyTorch's audio library (torchcodec) can't load yet. Install FFmpeg 7 and point the loader at it:

```bash
brew install ffmpeg@7
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib"   # add to your shell profile
```

Unblend supports Python 3.10–3.14 and runs on CPU, NVIDIA GPUs (CUDA) and Apple Silicon (MPS). It is tested on Linux and macOS (Apple Silicon only: PyTorch no longer publishes Intel Mac wheels); Windows should work but isn't covered by CI.

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

Note: `uvx` can't pick a PyTorch build for your GPU, so it gets PyPI's default wheel: GPUs only work on Apple Silicon, or on Linux with PyTorch's default CUDA version.


## CLI Usage

After installing Unblend:

```bash
$ unblend --help
$ unblend separate audio_file.mp3
$ unblend separate audio_file_1.mp3 audio_file_2.mp3 music_folder/
```

Stems are written to `separated/{model}/{track}/{stem}.wav` by default, at the model's sample rate (44.1 kHz stereo for every registered model) whatever the input's. Change it with `-o`, using the variables `{model}`, `{track}`, `{parent}` (the name of the track's folder), `{stem}`, `{ext}`, `{date}`, `{time}` and `{timestamp}`, and pick the container with `-f` (e.g. `-f flac`). Two tracks with the same name in different folders need `{parent}` in the template, or they would write to the same place (the CLI refuses rather than overwrite). Other common options:

- `-m MODEL` picks a model (see `unblend models list`); the default `auto` picks an HTDemucs model.
- `--isolate-stem vocals` writes just `vocals` and `no_vocals`.
- `--seed 0`, or `--shifts 0`, makes output reproducible run to run.
- `unblend tune` measures the fastest batch size and compile setting for your GPU.

## Models

| Model | Architecture | Stems | Weights license |
| --- | --- | --- | --- |
| `htdemucs` (default) | HTDemucs | drums, bass, other, vocals | MIT |
| `htdemucs_ft` | HTDemucs (4 fine-tuned models) | drums, bass, other, vocals | MIT |
| `htdemucs_6s` | HTDemucs | drums, bass, other, vocals, guitar, piano | MIT |
| `bs_roformer_sw` | BS-RoFormer | bass, drums, other, vocals, guitar, piano | unlicensed |
| `bs_roformer_anvuew` | BS-RoFormer | vocals, other | GPL-3.0 |
| `melband_roformer_kim` | Mel-Band RoFormer | vocals, other | MIT |
| `scnet_small` | SCNet | drums, bass, other, vocals | unlicensed |
| `scnet_xl_wide_v5` | SCNet | drums, bass, other, vocals | unlicensed |
| `roformer_vocals_ensemble` | ensemble | vocals, other | mixed |
| `htdemucs_scnet_ensemble` | ensemble | drums, bass, other, vocals | mixed |

Unblend's MIT license covers its code, not the model weights. Each model is downloaded under its own terms, and some were trained on MUSDB18, whose license is limited to non-commercial research. `unblend models info NAME` shows each model's full license notes. Check them before using a model commercially.

## API Usage

Unblend has several programmatic interfaces:
- [ONNX export](https://github.com/Ryan5453/unblend/blob/main/onnx.md) 
- [Python API](https://github.com/Ryan5453/unblend/blob/main/api.md)
- [Browser API](https://github.com/Ryan5453/unblend/blob/main/web/unblend/README.md) 
- [Cog](https://github.com/Ryan5453/unblend/blob/main/cog.yaml) (hosted on [Replicate](https://replicate.com/ryan5453/demucs))

## Credits

Unblend began as a fork of [Demucs](https://github.com/facebookresearch/demucs) by Alexandre Défossez and Meta, and its RoFormer and SCNet code builds on [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer), [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) and [starrytong/SCNet](https://github.com/starrytong/SCNet). The architectures are described in:

- Rouard, Massa, Défossez. [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553). ICASSP 2023.
- Lu et al. [Music Source Separation with Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612). ICASSP 2024.
- Wang et al. [Mel-Band RoFormer for Music Source Separation](https://arxiv.org/abs/2310.01809). 2023.
- Tong et al. [SCNet: Sparse Compression Network for Music Source Separation](https://arxiv.org/abs/2401.13276). ICASSP 2024.

The people who trained each checkpoint are credited in its `unblend models info` provenance.
