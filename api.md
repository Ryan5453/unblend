# <img src="https://raw.githubusercontent.com/Ryan5453/unblend/main/web/app/public/favicon.svg" width="30"> Python API

The Python API is built around two classes: `Separator`, which loads a model, and `SeparatedSources`, which holds its output.

```python
from unblend import Separator

separator = Separator("htdemucs")
sources = separator.separate("song.mp3")
for stem in sources.sources:
    sources.export_stem(stem, f"out/{stem}.wav")
```

## Separator

A `Separator` loads a model into memory, ready to separate audio.

```python
class Separator:
    def __init__(
        self,
        model: str | Model | ModelEnsemble = "htdemucs",
        device: str | None = None,
        only_load: str | None = None,
        dtype: torch.dtype | str | None = "auto",
        compile: bool = False,
        chunk_batch_size: int | None = None,
        custom_kernels: bool | None = None,
        combine: str | None = None,
        combine_params: dict | None = None,
    ) -> None:
```

A `Separator` takes the following parameters:

- `model` - The model to use for separation. While just passing in a string is the easiest, you can use `ModelRepository` to load models manually and then pass them in. A model you pass in is changed in place (moved to the device, cast to the dtype, with fused-kernel modules swapped in), so don't share one instance between Separators on different devices.
- `device` - The device/backend to use for loading and running the model. If left as `None` (the default), unblend auto-selects the best available backend at construction time. Pass `"cpu"`, `"cuda"`, `"cuda:N"` (a specific GPU), or `"mps"` (a `torch.device` works too) to force one.
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to ensembles like htdemucs_ft). This is a performance optimization (smaller download and memory footprint), it does not filter the output to one stem - use `SeparatedSources.isolate_stem` to actually isolate a stem. When the stem needs only one member, that member's outputs for stems the ensemble takes from elsewhere aren't real (it was trained for its own), so `isolate_stem` accepts only the stems listed in `reliable`, and their complement is the mix minus the stem.
- `dtype` - Inference *compute* precision. The default `"auto"` uses FP16 on CUDA GPUs with tensor cores (compute capability ≥ 7.0) and on MPS, and FP32 elsewhere. Pass `torch.float16`, `torch.bfloat16`, or `None`/`torch.float32` for FP32; FP16 or BF16 on CPU raises `ValidationError`. This is independent of the precision a checkpoint is stored at: weights saved at any float width load and run at whatever this is set to. BF16's short mantissa lands consistently about 19 dB further from FP32 than FP16 does (up to about 1e-3 RMS of error): roughly 25–58 dB SNR per stem on music, depending on the model and the stem, and a stem that is nearly silent in a passage can drop to single digits. Prefer FP16 unless you need BF16's range. In FP16, a chunk whose output overflows is redone through an FP32 copy of the model, and so, for HTDemucs, is a chunk that is mostly silence around a brief sound (HTDemucs scales it up until FP16 loses it). The copy is made once per call (per ensemble member), on the CPU if the device has no room for it. In music the HTDemucs case is typically a quiet intro or a fade-out, and costs little; redos are logged at INFO. FP16 HTDemucs otherwise lands about 50–60 dB from FP32 on most music, but a few loud passages come out nearer 20 dB; use FP32 for the most faithful output.
- `compile` - Optional, if `True`, compiles the model's neural-network core on CUDA — roughly 1.3–1.5× in FP16, at the cost of startup time and extra held VRAM, so it pays off on long jobs. CPU and MPS ignore it; on CUDA it raises `ValidationError` for SCNet, which has no compile target. A model compiled this way runs stock PyTorch ops inside the graph instead of the fused CUDA kernels; `enable_compile()` on an eager separator compiles through the fused kernels instead. The CLI decides per workload instead, with `--compile` / `--no-compile` to force either way.
- `chunk_batch_size` - Optional, how many segments to run per forward pass. The default (`None`) sizes it from free GPU memory on CUDA and backs off if that proves too large, and uses 1 on CPU and MPS; an explicit value (at most 1024) is used exactly as given, and OOM raises.
- `custom_kernels` - Optional, whether the fused CUDA/Metal kernels may be used. `None` (the default) defers to the `UNBLEND_CUSTOM_KERNELS` environment variable, which enables them unless set to `0`, `off`, `false` or `no`; `False` forces vanilla PyTorch ops on every device and skips the one-time CUDA extension build, which is the baseline for A/B comparisons. It raises `ValidationError` for a model instance that an earlier Separator already fitted with fused modules; load a fresh one.
- `combine` - Optional, overrides how an ensemble's member outputs are combined for this instance. See [Combining members](#combining-members) for the mode names. Raises `ValidationError` on a single-member model, or for a mode the ensemble's weights don't allow. It is checked before anything downloads, the same way with `only_load`: when the stem needs only one member, the override is validated but has nothing to do.
- `combine_params` - Optional, the STFT geometry the spectral combine modes use, as `{"n_fft": int, "hop_length": int}`. Keys you leave out keep the ensemble's own values (the registry default is `{"n_fft": 1024, "hop_length": 256}`); `hop_length` must be smaller than `n_fft` and divide it evenly. Like `combine`, raises `ValidationError` on a single-member model.

### Attributes

After construction, the following attributes are available on a `Separator` instance:

- `device` - The device type being used for processing (`"cpu"`, `"cuda"` or `"mps"`; the GPU index from `"cuda:N"` isn't included).
- `dtype` - The dtype being used for inference (`torch.dtype | None`).
- `model` - The loaded model instance (`Model | ModelEnsemble`).
- `audio_channels` - Number of audio channels the model expects (`int`).
- `sample_rate` - Sample rate the model operates at (`int`).
- `chunk_batch_size` - Number of segments processed per forward call (`int`). Sized from free memory at construction on CUDA, 1 on CPU and MPS, unless you passed an explicit value.

With `compile=True`, warmup happens at the end of `__init__`; `separator.warmup()` re-primes it later. `warmup()` is CUDA-only: it raises `ValidationError` on CPU/MPS or models outside the HTDemucs/RoFormer compile targets. Workload-aware callers can instead construct eagerly, inspect their job, and call `separator.enable_compile()` to compile/capture the existing CUDA model in place without reloading weights; repeated calls are no-ops.

Once you have a `Separator` instance, you can use the `separate` method to separate one audio input — or a list of inputs — into its constituent stems.

```python
def separate(
    self,
    audio: tuple[Tensor, int] | Path | str | bytes | list[...],
    shifts: int = 1,
    split_overlap: float = 0.25,
    seed: int | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    use_only_stem: str | None = None,
    chunk_batch_size: int | None = None,
) -> SeparatedSources | list[SeparatedSources]:
```

`separate` takes:

- `audio` - The audio to separate: a `(Tensor, sample_rate)` tuple, a file path, a URL FFmpeg can open, or raw bytes. A single input returns one `SeparatedSources`; a `list` returns `list[SeparatedSources]` and is the efficient way to serve many short clips at once. Tensors must be `[channels, samples]` (or 1-D mono) floating-point audio in the usual amplitude range; integer, boolean, complex and time-major tensors are rejected.
- `shifts` - How many randomly time-shifted passes to average, which stabilizes the output at the cost of proportionally more compute. An integer in `[0, 20]`; `0` disables shifting and is deterministic.
- `split_overlap` - The overlap between consecutive segments. Must be in the range `[0.0, 1.0)`. Higher values smooth segment boundaries at the cost of more compute per track.
- `seed` - Optional seed for the shift offsets, which are random whenever `shifts` is at least 1. Output is reproducible across runs at the same seed. Every input in a list call gets the same offsets, so a seeded input separates identically alone or in a list. The global `random`/`torch` RNGs are not touched.
- `progress_callback` - A callback function receiving aggregate and per-input progress for both single and list input. List-input events remain one monotonic global stream while identifying the input advanced by each completed chunk. View the [Progress Callbacks](#progress-callbacks) section for more information.
- `use_only_stem` - Run only the specialist member for this stem, in an ensemble like `htdemucs_ft`. Like `only_load` this is a **performance optimization**, not a filter — every source is still returned; when the result's `reliable` lists stems, only those are real (when it is `None`, all are). Prefer `only_load` when you know the stem before constructing the `Separator`.
- `chunk_batch_size` - Override the auto-detected `chunk_batch_size` for this call without persisting it. Pass `None` (default) to use `self.chunk_batch_size`. A compiled separator captures one fixed batch shape, so per-call overrides are rejected there — pass it to `Separator(...)` instead.

Audio is processed in chunks of the model's training segment length, which isn't configurable. On CUDA, output for a track accumulates on the GPU while it fits in a share (about 30%) of free VRAM; longer inputs accumulate on the CPU instead, so GPU memory use stops growing with audio length. On MPS the whole track's output is kept on the GPU.

Example:

```python
# Single input
sources = separator.separate(
    "mixture.wav",
    shifts=4,
    split_overlap=0.25,
    seed=1234,
)

# Batched list input — pools tail chunks across inputs and supports progress
results = separator.separate(
    ["a.wav", "b.wav", "c.wav"],
    progress_callback=progress_callback,
)
for sources in results:
    ...
```

## SeparatedSources

`Separator.separate` returns a `SeparatedSources` holding the stems, their sample rate, and the input.

### Attributes

- `sources` - Dictionary mapping stem names (e.g. `"vocals"`, `"drums"`) to their audio tensors (`dict[str, Tensor]`). You can iterate the keys to get available stem names.
- `sample_rate` - Sample rate of the separated audio (`int`), inherited from the model.
- `original` - The input as the model saw it (`Tensor`): resampled to `sample_rate` and converted to the model's channel count, not the raw file.
- `reliable` - When only one ensemble member ran (`only_load` or `use_only_stem`), the stems the ensemble takes from that member alone (`tuple[str, ...] | None`). The other entries in `sources` are then that member's untrained outputs, not real stems, and `isolate_stem` accepts only these. `None` when every output is a real stem (including when that member is the source of every stem).

`export_stem` encodes a stem to a file or to bytes:

```python
def export_stem(
    self,
    stem_name: str,
    path: Path | str | None = None,
    format: str = "wav",
    clip: str | None = "rescale",
) -> Path | bytes:
```

It takes:

- `stem_name` - The name of the stem to export.
- `path` - The path to save the stem to. If not provided, the encoded file is returned as `bytes`.
- `format` - The container to encode, anything FFmpeg supports. Used when returning bytes, or appended to a `path` that doesn't already end in an audio extension (so `"out/Song ft. Artist"` becomes `"out/Song ft. Artist.wav"`). An unencodable format raises `ValidationError`. Opus (`opus`, `webm`) can't encode 44.1 kHz, so stems at a rate Opus doesn't support are resampled to 48 kHz first.
- `clip` - The clipping mode to use to prevent audio distortion. One of `"rescale"` (default — divide by `max(1, 1.01 * max(|x|))`, so it only acts on peaks above about 0.99; it scales each stem on its own, so a rescaled stem no longer sums with the others to the mix), `"clamp"` (hard clip to `±0.99`), `"tanh"` (soft clip; it bends every sample slightly, not just the peaks), or `None` (no clipping).

`isolate_stem` narrows a `SeparatedSources` to one stem. This returns a new `SeparatedSources` instance with the chosen stem and an accompanying complement stem (no_{STEM}) that is the sum of all other stems — or, when the result's `reliable` is set (one ensemble member ran, and the ensemble takes other stems from elsewhere), the mix minus the stem. Isolating a stem not listed in the result's `reliable` raises `ValidationError`.

```python
def isolate_stem(self, name: str) -> "SeparatedSources":
```

## Auto Model Selection

`select_model` picks an HTDemucs-family model for a stem. This is what the CLI's `--model auto` uses.

```python
from unblend import select_model

def select_model(
    isolate_stem: str | None = None,
) -> tuple[str, str | None]:
```

If you are attempting to isolate a single stem, pass in the name of the stem to the `isolate_stem` parameter.

This will return a tuple of the model name and the stem to exclusively load from the model. When creating a `Separator` instance, you pass these in as the `model` and `only_load` parameters respectively.

The routing is:

| `isolate_stem` | model | `only_load` |
|---|---|---|
| `vocals`, `bass`, `other` | `htdemucs_ft` | the requested stem |
| `drums`, `guitar`, `piano` | `htdemucs_6s` | `None` |
| anything else / `None` | `htdemucs` | `None` |

## ModelRepository

unblend provides a `ModelRepository` class to more deeply control the model loading process. This is used internally by the `Separator` class but can be used directly to load models manually to then pass to Separator itself.

`ModelRepository` is initialized with no required parameters. (i.e. `repo = ModelRepository()`) Pass `extra_models=` to overlay specific [models files](#custom-models) instead of the default ones (`extra_models=[]` means the shipped registry only), and `metadata_path=` to replace the shipped `metadata.yaml` itself.

### get_cache_info

```python
def get_cache_info(self) -> dict[str, dict]:
```

This will return a dictionary of information about the cached models. Models with at least one cached file are included, so a partially-downloaded model shows up with `"complete": False`.

```python
{
    "model_name": {
        "files": {       # Cached checkpoint files, keyed by checksum prefix
            "checksum": {
                "path": str,       # Path to the cached file
                "size_bytes": int, # Size of the file in bytes
                "complete": bool,  # Its size matches the registry (False: truncated)
            }
        },
        "size_bytes": int,  # Total size of the cached files in bytes
        "total_files": int, # Number of remote checkpoint files (local paths aren't cached)
        "complete": bool,    # True when every file is cached and complete
    },
    ...
}
```

### get_model

```python
def get_model(self, name: str, only_load: str | None = None, progress_callback: Callable[[str, dict[str, Any]], None] | None = None) -> Model | ModelEnsemble:
```

When using the `get_model` method, the following parameters are available:

- `name` - The name of the model to load.
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to multi-member entries like htdemucs_ft).
- `progress_callback` - Optional, a callback function to receive progress updates. View the [Progress Callbacks](#progress-callbacks) section for more information.

This will return either a `Model` or `ModelEnsemble` instance corresponding to the given model name.

### list_models

```python
def list_models(self) -> dict[str, dict]:
```

This returns a deep copy of every registry entry, keyed by name, with a derived `backend` added: the family every member belongs to (`"demucs"`, `"roformer"` or `"scnet"`), or `"ensemble"` when members come from different families. Entries have the [fields a models file uses](#entry-fields), so the shape depends on the entry:

```python
{
    "htdemucs": {
        "backend": "demucs",
        "architecture": "htdemucs",
        "sources": ["drums", "bass", "other", "vocals"],
        "checkpoint": {"format": "safetensors", "url": str, "sha256": str, "size_bytes": int},
        "config": dict,
        "license": "unlicensed",
        "license_note": str,
        "provenance": str,
    },
    "htdemucs_ft": {        # several checkpoints of one architecture
        "backend": "demucs",
        "members": [{"format": "safetensors", "url": str, ...}, ...],
        "weights": [[1.0, 0.0, 0.0, 0.0], ...],
        ...
    },
    "htdemucs_scnet_ensemble": {   # members from different families
        "backend": "ensemble",
        "members": [{"model": "htdemucs"}, {"model": "scnet_xl_wide_v5"}],
        "combine": "weighted_mean",
        ...
    },
}
```

`get_cache_info()` keys each model's `files` by the first 16 characters of the artifact's SHA-256.

### remove_model

```python
def remove_model(self, name: str, include_shared: bool = False, also_removing: Iterable[str] = ()) -> bool:
```

Removes a model's downloaded weights from the cache. Files that another fully downloaded model also uses (an ensemble's members, say) are kept unless `include_shared=True` or every model using them is listed in `also_removing`; `shared_artifacts(name)` reports them. Returns `True` if anything was removed, `False` for an unknown model or an empty cache; raises `ModelLoadingError` if a cached file can't be removed (e.g. permissions).

### get_cache_dir

A module-level function (not a `ModelRepository` method), imported directly:

```python
from unblend.repo import get_cache_dir

def get_cache_dir() -> Path:
```

This returns the model cache directory (created on first download).
`UNBLEND_CACHE_DIR` relocates it; the default is `~/.unblend/models`. Values are
tilde-expanded and resolved.

## Progress Callbacks

unblend provides a callback-based system for monitoring progress during long-running operations like model downloads and audio processing. This system is designed to be UI-agnostic, allowing you to implement a progress display into your own CLI or other application.

All unblend progress callbacks are designed to use the same API. You should implement a method that matches the following signature:

```python
def progress_callback(event: str, data: dict[str, Any]) -> Any:
    pass
```

### Model Downloading

`ModelRepository.get_model` sends these events. `Separator` doesn't take a download callback, so to show download progress, call `get_model` yourself and pass the model to `Separator`:

- `download_start`: Fired when the download process begins.
  - `model_name`: Name of the model being downloaded.
  - `total_files`: Number of checkpoint files the model loads, cached or local; not all may need downloading.
- `file_start`: Fired when a checkpoint file starts downloading.
  - `model_name`: Name of the model.
  - `file_index`: Index of the current file (1-based).
  - `total_files`: Total number of files.
  - `file_size_bytes`: Size of the file in bytes.
- `file_progress`: Fired periodically while a file downloads.
  - `model_name`: Name of the model.
  - `file_index`: Index of the current file.
  - `total_files`: Total number of files.
  - `progress_percent`: Percentage complete (0-100).
  - `downloaded_bytes`: Bytes downloaded so far.
  - `total_bytes`: Total bytes to download.
- `file_complete`: Fired when a file is downloaded and verified, or found in the cache.
  - `model_name`: Name of the model.
  - `file_index`: Index of the current file.
  - `total_files`: Total number of files.
  - `cached`: Optional. True if the file was already cached.
- `download_complete`: Fired when every file is available.
  - `model_name`: Name of the model.
  - `total_files`: Total number of files.

### Audio Separation

When using `Separator.separate`, the callback receives the following events:

- `processing_start`: Fired before processing segments.
  - `total_chunks`: Total number of segments across every input, shift, and ensemble member.
  - `total_inputs`: Number of input waveforms.
  - `input_total_chunks`: Per-input total segment counts, in input order.
- `chunk_complete`: Fired after each routed segment is processed.
  - `completed_chunks`: Aggregate segments completed so far.
  - `total_chunks`: Aggregate segment total.
  - `input_index`: Zero-based index of the input advanced by this event.
  - `input_completed_chunks`: Segments completed for that input.
  - `input_total_chunks`: Segment total for that input.
- `processing_complete`: Fired after all segments are processed, with the same aggregate/per-input totals as `processing_start`.

## Version

`unblend.__version__` (or `unblend.get_version()`) is the installed version string, e.g. `"1.0.0"`.

## Other Exports

Lower-level symbols. To run inference without `Separator` (no normalization, resampling or batching), load a model with `ModelRepository().get_model(name)` and call `unblend.apply.apply_model(model, mix, shifts=0)` on a `[batch, channels, samples]` tensor at the model's `samplerate`; HTDemucs expects the track normalized as described in onnx.md.

### Models

- `Model` — base `nn.Module` returned by `ModelRepository.get_model` for a single-checkpoint model (e.g. plain `htdemucs`). Has `.sources` (stem names), `.samplerate`, `.audio_channels`.
- `ModelEnsemble` — `nn.Module` returned for multi-member entries like `htdemucs_ft`. Holds `.models` (an `nn.ModuleList` of `Model`), `.weights` (per-source mixing rows), and the shared `.sources` / `.samplerate` / `.audio_channels`.

### Device

```python
from unblend import default_device

def default_device() -> str:
```

Returns `"cuda"`, `"mps"`, or `"cpu"`, whichever is available — the same selection `Separator(device=None)` uses.

```python
from unblend import default_dtype

def default_dtype(device: str) -> torch.dtype | None:
```

Returns the inference dtype `dtype="auto"` picks for a device (`torch.float16` on MPS and CUDA with tensor cores; `None`, meaning FP32, on CPU and older CUDA GPUs). Raises `ValidationError` for other device strings, or for `"cuda"` without CUDA available.

### Exceptions

Errors unblend raises for bad input, models or audio derive from `UnblendError` (bugs and OS errors such as a full disk surface as ordinary Python exceptions):

- `UnblendError` — base class for everything raised by `unblend`.
- `ValidationError` — invalid argument (bad device, bad dtype, unknown stem, out-of-range parameter).
- `ModelLoadingError` — model not found, metadata malformed, sha256 mismatch, download failure.
- `LoadAudioError` — input audio could not be decoded.

```python
from unblend import UnblendError, ValidationError, ModelLoadingError, LoadAudioError
```

## Custom models

Unblend ships a fixed registry, but you can add your own models without
modifying the package or hosting weights anywhere.

Put entries in `~/.unblend/models.yaml`, or in other files listed in
`UNBLEND_EXTRA_MODELS` (`os.pathsep`-separated). A default `ModelRepository()`
(and so `Separator` and the CLI) loads both; passing `extra_models=` loads exactly
the files you give instead. Entries are **added** to the shipped registry; a file
that reuses a built-in name is rejected rather than shadowing it. A broken
`~/.unblend/models.yaml` is skipped with a warning (unless a file you listed uses
one of its models, which then reports the fault), while a broken file you listed
explicitly is an error. `unblend models unregister NAME` removes an entry.

```yaml
version: 1
models:
  my_scnet:
    architecture: scnet_masked
    license: see upstream model card
    sources: [drums, bass, other, vocals]
    samplerate: 44100
    segment_samples: 485100
    config:                       # the upstream config's `model:` section, verbatim
      dims: [4, 32, 64, 128]
      nfft: 4096
      hop_size: 1024
    checkpoint:
      format: safetensors
      path: ~/models/my_scnet.safetensors
```

Models files are YAML, which is what the ecosystem's configs are written in —
so an upstream `model:` section is a paste, not a translation, and an entry can
carry comments. A file named `.json` is read as JSON, so anything already
written that way keeps working. The shipped registry is
[`unblend/metadata.yaml`](https://github.com/Ryan5453/unblend/blob/main/unblend/metadata.yaml).

```bash
unblend models list          # your model appears, marked "Local"
unblend separate --model my_scnet track.wav
```

### Entry fields

| Field | Meaning |
| --- | --- |
| `architecture` | Which implementation builds the model. One of `htdemucs`, `bs_roformer`, `mel_band_roformer`, `scnet`, `scnet_masked`. |
| `sources` | Output stem names, in the order the model emits them. |
| `samplerate` | Sample rate the weights operate at. |
| `segment_samples` | Training chunk length, in samples. |
| `config` | Constructor kwargs for the architecture — the upstream config file's `model:` section, verbatim. |
| `checkpoint` | Where the weights are (see below). Exactly one set of weights. |
| `members` | Two or more members instead of a `checkpoint` (see below) — an ensemble, or a bag of same-architecture checkpoints. |
| `license` | Free-form label, shown by `models list` and `list_models`. Unblend does not interpret it. |
| `license_note`, `provenance` | Optional free-form text, shown by `models info`. |
| `weights` | For an ensemble: one row per member, one column per source. Defaults to all ones. |
| `combine`, `combine_params` | For an ensemble: how member outputs are combined (see below). |
| `segment` | Optional, in seconds: shortens (never enlarges) the configured training segment. |

An entry names its weights exactly once: `checkpoint` for one set, `members`
for two or more. There is no `backend` field — the loader family is derived
from `architecture`, and declaring it is an error rather than a second source
of truth that can disagree. Every architecture belongs to exactly one family:
`htdemucs` to `demucs`, the two RoFormers to `roformer`, the two SCNets to
`scnet`. `list_models` reports the derived value back to you; drop it before
copying an entry into a models file.

### Where the weights come from

Every backend describes its weights the same way, and either source works for
any architecture:

```yaml
format: safetensors
path: ~/models/my_model.safetensors
```

```yaml
format: safetensors
url: https://huggingface.co/me/my-model/resolve/main/model.safetensors
sha256: 3f786850e387550fdab836ed7e6dc881de23001b00000000000000000000beef
size_bytes: 219000000
```

- A **local `path`** is read where it lies. A relative path is relative to the models file's directory (the real file's, if the models file is a symlink). Nothing is copied into the cache, so
  `models remove` will never delete it and `models list` reports it as `Local`.
  `sha256` and `size_bytes` are optional there, and verified when present.
- An **https `url`** is downloaded once into the model cache
  (`UNBLEND_CACHE_DIR`, default `~/.unblend/models`) and served from there on
  every later run. A download has to be verifiable, so `sha256` and
  `size_bytes` are required — the file is checked before it is promoted into the
  cache, and again on each load.
- A multi-checkpoint entry lists one such artifact per member under `members`,
  and may mix local and remote ones. A Demucs entry's `config` must declare the
  same `sources` as the entry.

### Ensembles

An ensemble entry lists `members` instead of a `checkpoint`. A member inherits
anything it does not state from the entry, so a bag of same-architecture
checkpoints stays terse while a mixed one spells each member out:

```yaml
version: 1
models:
  my_bag:
    architecture: scnet          # inherited by both members below
    sources: [drums, bass, other, vocals]
    samplerate: 44100
    segment_samples: 485100
    config:
      dims: [4, 64, 128, 256]
      nfft: 4096
      hop_size: 1024
    combine: avg_wave
    members:
      - checkpoint: { format: safetensors, path: ~/models/a.safetensors }
      - checkpoint: { format: safetensors, path: ~/models/b.safetensors }
```

A member may also give the artifact fields directly (`- { format: safetensors,
url: ..., sha256: ..., size_bytes: ... }`) instead of under `checkpoint:`, as the
shipped `htdemucs_ft` does.

A member can also just name another registered single-checkpoint model (not
an ensemble), which is how you ensemble models that already ship without
restating their config:

```yaml
version: 1
models:
  my_vocals:
    sources: [vocals, other]
    combine: min_fft
    members:
      - model: melband_roformer_kim
      - model: bs_roformer_anvuew
```

Members must agree on stems (same names, same order), sample rate and channel
count. They need *not* agree on normalization: HTDemucs wants track-level
normalized audio and the other architectures want it raw, so an ensemble mixing
them takes raw audio and normalizes around the members that need it — each sees
exactly what it would see running alone, and members are combined in the input's
own scale.

Members that share a checkpoint with another registered model share its cache
file, so an ensemble of registered models downloads nothing new, and
`models remove` on it keeps those shared files unless you pass
`--include-shared`.

Unblend ships two, so the modes below are usable without writing any config:
`roformer_vocals_ensemble` (Mel-Band + BS-RoFormer, two stems) and
`htdemucs_scnet_ensemble` (HTDemucs + SCNet xl-wide, four stems). Each costs
both members' inference time.

### Combining members

`combine` names how member outputs are reduced. The names are the ecosystem's
(ZFTurbo's Music-Source-Separation-Training, audio-separator, UVR), so a recipe
written against those transfers verbatim.

| Mode | What it does |
| --- | --- |
| `weighted_mean` (default), `avg_wave` | Per-source weighted average of the waveforms. The two names are the same mode. |
| `median_wave` | Element-wise median of the waveforms (with two members, the same as their average). |
| `min_wave`, `max_wave` | The sample with the smallest/largest absolute value, sign kept. |
| `avg_fft` | Weighted average of the complex spectrograms. |
| `median_fft` | Per bin, the member ranked in the middle by magnitude (with two members, the average). |
| `min_fft`, `max_fft` | Per bin, the whole complex value from the member with the smallest/largest magnitude. |
| `uvr_min_spec`, `uvr_max_spec` | UVR's names for `min_fft`/`max_fft` — the same operation. |

`combine_params` sets the STFT geometry for the spectral modes; it defaults to
`{"n_fft": 1024, "hop_length": 256}`, and `n_fft` must be a whole multiple of
`hop_length` and larger than it.

Three things worth knowing:

- **The selection modes need a 0/1 mask.** `median_*`, `min_*` and `max_*` pick
  among members rather than blending them, so a real-valued weight has nowhere
  to apply. Upstream tools silently ignore weights there; Unblend rejects them,
  so a recipe never quietly does something other than what it says. A zero still
  means "this member does not contribute to this stem".
- **Only `weighted_mean` streams.** It is linear, so it folds into a running
  accumulator and holds two tensors at a time. The others need every member's
  finished output side by side; the spectral ones transform in blocks so peak
  memory tracks the block rather than the track.
- **The default is also the quality pick.** ZFTurbo's own testing found the plain
  weighted average was always better or equal in SDR; the other modes are for
  taste and interop (`min_fft` is the conservative one — it keeps only what the
  members agree on). Unblend has not measured them.

Anything registered can be overridden per run without touching metadata:

```bash
unblend separate --model roformer_vocals_ensemble --combine min_fft track.wav
```

```python
Separator(model="roformer_vocals_ensemble", combine="min_fft")
```

### Importing a checkpoint from elsewhere

Unblend does not define a weight layout of its own: module and parameter names
in every architecture are pinned to their reference implementations (lucidrains
/ ZFTurbo for the RoFormers, the official SCNet, Meta's Demucs), so community
checkpoints trained against those load verbatim, `strict=True`. What Unblend
insists on is the *container*: Safetensors, so loading is pickle-free.

`unblend models import` does the repackaging:

```bash
unblend models import model.ckpt --config config.yaml --name my_model \
    --license "see upstream model card"
```

```
✓ Loaded as mel_band_roformer and strict-loaded 684 tensors
✓ Wrote /home/me/.unblend/imported/my_model.safetensors (870.6 MB)
✓ Registered my_model in /home/me/.unblend/models.yaml

Try it: unblend separate --model my_model track.wav
```

What it does:

- **Reads the tensors out of whatever container they arrived in** — a bare
  state dict, or a training framework's checkpoint with the weights under
  `state_dict` and a `model.` / `module.` prefix on every key. It loads with
  `torch.load(weights_only=True)`, so a checkpoint that needs real unpickling
  is refused rather than executed. That includes the official Demucs `.th`
  files, which pickle their model class; Unblend already ships those weights.
- **Translates the config.** ZFTurbo's Music-Source-Separation-Training layout
  is understood directly: `model:` is the constructor config (keys the model
  doesn't take, such as `flash_attn`, are dropped and listed), `audio.chunk_size`
  and `audio.sample_rate` are the geometry (without a `chunk_size`, a
  `training.segment` in seconds is converted at the sample rate, including one
  given with `--samplerate`), `training.instruments` are the stems. A
  `training.target_instrument` marks a single-head model, whose second stem is
  synthesised as `mixture - prediction`. Anything the config does not
  say can be given with `--architecture`, `--stem`, `--samplerate` and
  `--segment-samples`.
- **Infers the architecture and proves it.** Parameter names identify the
  family, and the masked SCNet is identifiable from weights plain SCNet has no
  slot for. BS- and Mel-Band RoFormer share their names, so the config decides —
  and failing that both are tried. Whichever candidate is chosen, the model is
  *built and strict-loaded* before anything is written: a mistranslated config
  or a mislabelled architecture fails here, naming what did not fit, instead of
  producing an entry that breaks at separation time. `--architecture` is
  verified the same way, not trusted.
- **Writes a self-describing artifact.** The fields that build the model
  (architecture, config, stems, sample rate, segment) go into the Safetensors
  header, which the entry's recorded sha256 covers — so an embedded config is
  verified along with the weights, which a models file beside it is not. A
  hand-written entry for such a file with a local `path` needs only its
  `sources`, `format: safetensors` and the path; the rest is read from the
  header (a few KB, whatever the file's size). License terms aren't in the
  header, so state `license` in the entry if `models info` should show them. A
  `url` entry is checked before anything is downloaded, so it has to state
  `architecture` and `config` itself (and, except for HTDemucs, `samplerate`
  and `segment_samples`): paste them from the entry `--print` gives.
- **Validates it before registering it**: the updated models file is checked
  by the registry itself before it replaces the old one (kept as `.bak`). The
  name is checked before anything is written, so a clash never overwrites an
  existing model's weights. `--print` emits the entry instead of writing it,
  and `--models-file` chooses where it lands (default
  `~/.unblend/models.yaml`, which is always loaded). Only architectures
  Unblend implements can load at all. MDX-Net and VR-arch weights (much of
  UVR's model list, including everything shipped as `.onnx`) are different
  architectures, not different packaging — the import says so rather than
  failing obscurely.

### Notes
- Weights load strictly, so a config that disagrees with the checkpoint fails
  loudly rather than degrading silently.
- Everything is validated when the repository is constructed — a malformed
  entry fails before any download starts, not part-way through one.
- For a single-head model given two `sources`, the second is synthesised as
  `mixture - prediction`. Order matters: an instrumental model must declare
  `["other", "vocals"]`, and getting it backwards produces silently wrong
  output rather than an error.
- `--isolate-stem` on an ensemble runs (and downloads) one member when only one
  contributes to that stem, whatever the combine mode: every mode reduces to
  the identity over a single member. `htdemucs_ft`'s one-hot matrix is what
  makes single-stem extraction there cost one member instead of four. When
  several members contribute, all of them run.
