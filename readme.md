# Unblend

Unblend is a music source separation library with one API across HTDemucs, BS-RoFormer,and Mel-Band RoFormer. 
Unblend's Demucs backend runs ~6× (19–23x for single-stem extraction) faster at equal quality. 

## Installation

### Prerequisites

- FFmpeg v4+ available in your `PATH`
- [`uv`](https://docs.astral.sh/uv/#installation)
- Optional: C/C++ compiler such as GCC, Clang, or MSVC (enables torch.compile support)

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
```

## Programmatic Use

Unblend provides a [Python API](https://github.com/Ryan5453/unblend/blob/main/api.md) for integrating source separation into your own application. 
Additionally, there is a [Cog](https://github.com/replicate/cog) for HTDemucs which allows you to easily deploy it as a REST API. 
You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

Unblend can also run in the browser via ONNX.
See the [ONNX export notes](https://github.com/Ryan5453/unblend/blob/main/onnx.md) and the [npm package docs](https://github.com/Ryan5453/unblend/blob/main/web/demucs/README.md) for details.

## Licensing

Unblend's own code is MIT licensed — see [LICENSE](https://github.com/Ryan5453/unblend/blob/main/LICENSE).

The model weights are **not** covered by that license. Unblend redistributes them, converted to safetensors, from its own Hugging Face mirror, and each carries its own terms:

| Model | Weights license |
| --- | --- |
| `bs_roformer_anvuew` | GPL-3.0 |
| `htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `melband_roformer_kim` | MIT |
| `bs_roformer_sw`, `scnet_small`, `scnet_xl_wide_v5` | No license grant |

The HTDemucs weights — including `htdemucs`, which Unblend selects by default — are MIT: Alexandre Defossez republished the exact official-signature checkpoints under that license on [Hugging Face](https://huggingface.co/adefossez/HTDemucs). Note that they were trained on MUSDB18-HQ plus 800 proprietary songs, so the license covers the weights, not the training data.
The `melband_roformer_kim` weights are MIT, declared by their author Kimberley Jensen on the [upstream model card](https://huggingface.co/KimberleyJSN/melbandroformer).
The `bs_roformer_anvuew` weights are GPL-3.0, declared by anvuew on the [upstream model card](https://huggingface.co/anvuew/BS-RoFormer). Whether copyleft attaches to model weights at all is legally untested; Unblend downloads them at runtime rather than bundling them.
The `bs_roformer_sw` weights carry no grant, and their origin is murky: the original trainer is unidentified, the account that published the checkpoint has since been deleted, and the surviving mirror declares its license as "unknown".
The two SCNet checkpoints carry no grant either. Both the SCNet architecture and ZFTurbo's Music-Source-Separation-Training, which published them, are MIT — but that covers their source code, not the checkpoint files, which are attached as release assets with no stated terms. Both were trained only on MUSDB18, whose own license restricts it to non-commercial research.

Run `unblend models list` to see the license for any model, or read [`unblend/metadata.json`](https://github.com/Ryan5453/unblend/blob/main/unblend/metadata.json) for the full per-model notes.
Confirming that your use of a given model complies with its terms is your responsibility; this summary is not legal advice.
