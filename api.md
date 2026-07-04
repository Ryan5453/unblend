# Demucs API

The Demucs Python API is primarily comprised of two classes: `Separator` and `SeparatedSources`.

## Separator

The `Separator` class is a high level representation of a Demucs audio source separation model. When you want to separate an audio file into its constituent stems, you will first need to create an instance of the `Separator` class which will load the model into memory for use.

```python
separator = Separator(
    model: str | Model | ModelEnsemble = "htdemucs",
    device: str | None = None,
    only_load: str | None = None,
    dtype: torch.dtype | str | None = "auto",
    compile: bool = False,
)
```

A `Separator` takes the following parameters:

- `model` - The model to use for separation. While just passing in a string is the easiest, you can use `ModelRepository` to load models manually and then pass them in.
- `device` - The device/backend to use for loading and running the model. If left as `None` (the default), Demucs auto-selects the best available backend at construction time (cuda > mps > cpu). Pass `"cpu"`, `"cuda"`, or `"mps"` to force one.
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to bag-of-models like htdemucs_ft). This is a **performance optimization** (smaller download and memory footprint) — it does **not** filter the output to one stem; the result still contains all of the model's sources, with only the named stem at full quality. Use `SeparatedSources.isolate_stem` to actually isolate a stem.
- `dtype` - Inference precision. The default `"auto"` picks the fastest dtype that keeps SDR at FP32 level: FP16 on CUDA GPUs with tensor cores (compute capability ≥ 7.0) and on MPS; FP32 on CPU and older CUDA GPUs. FP16 and BF16 both measure SDR-equal to FP32 on MUSDB18 (within 0.01 dB) at ~1.7× the speed — FP16 tracks FP32 slightly closer, which is why auto prefers it. Pass `torch.float16` or `torch.bfloat16` explicitly to force reduced precision (CUDA/MPS only; CPU is rejected), or `None` / `torch.float32` to force FP32. On MPS, FP16 uses custom Metal kernels in `demucs.metal`; BF16 works but, as of PyTorch 2.11, is ~22 % slower than FP16 (its native ops aren't well-optimized on MPS yet) — use it when you want BF16's FP32 exponent range.
- `compile` - Optional, if `True`, applies `torch.compile` (CUDAGraphs / Inductor) to the HTDemucs neural network core. Significantly improves steady-state throughput on CUDA at the cost of a heavy first-call compile (~7–55 s depending on dtype). Silently ignored on MPS and CPU (`torch.compile` on MPS hits dynamo recompile limits on the Metal kernels' varying channel counts). With a `ModelEnsemble` (e.g. `htdemucs_ft`), each sub-model captures its own CUDAGraph and stays GPU-resident across requests — the classic per-sub-model device-restore is skipped so the capture isn't invalidated. Plan ~`num_sub_models × per-model VRAM` accordingly.

### Attributes

After construction, the following attributes are available on a `Separator` instance:

- `device` - The device being used for processing (`str`).
- `dtype` - The dtype being used for inference (`torch.dtype | None`).
- `model` - The loaded model instance (`Model | ModelEnsemble`).
- `audio_channels` - Number of audio channels the model expects (`int`).
- `sample_rate` - Sample rate the model operates at (`int`).
- `chunk_batch_size` - Number of segments processed per forward call. On CUDA this is auto-detected at init time from a single eager forward measurement + `mem_get_info`; with `compile=True` the estimate is additionally capture-verified, halving on CUDA OOM (max 4 attempts), while `compile=False` trusts the estimate directly. Read this attribute after construction to see the value picked for your GPU. On MPS it defaults to 2; on CPU it defaults to 1.

If you enable `compile=True`, warmup happens automatically at the end of `__init__` (via a zero-tensor pass through `Separator.separate`, so the CUDAGraph captured is the same one real requests reuse). You can call `separator.warmup()` again later to re-prime if needed; the method takes no arguments because tail-padding inside `apply_model` guarantees a single batch shape per session.

```python
separator.warmup()  # no args — there's exactly one batch shape after tail-padding
```

`warmup()` is CUDA-only: it raises `ValidationError` on CPU/MPS, and on non-HTDemucs models.

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

When separating audio, you have the ability to specify the following parameters:

- `audio` - The audio to separate. **Polymorphic input**: a single `(Tensor, sample_rate)` tuple / file path / raw bytes returns a single `SeparatedSources`. Passing a `list` of those returns `list[SeparatedSources]` and pools tail chunks across inputs (so every forward pass runs at full `chunk_batch_size`, no wasted slots). Useful when serving many short clips concurrently — see `apply_model_multi` in `demucs/apply.py`.
- `shifts` - The number of random shifts for equivariant stabilization. In simple terms, this is a technique to make the model more robust to small changes in the audio, such as small shifts in time or pitch. More shifts mean generally higher quality separation but also longer processing time. Must be an integer in `[1, 20]`.
- `split_overlap` - The overlap between consecutive segments. Must be in the range `[0.0, 1.0)`. Higher values smooth segment boundaries at the cost of more compute per track.
- `seed` - Optional random seed for reproducible shift-based inference. With list input, per-input shift offsets advance from this seed in sequence — outputs are reproducible across runs at the same seed, but a list call with `seed=N` does NOT produce bit-identical outputs to N separate single-file calls with `seed=N`. Setting this also reseeds the process-global `random` and `torch` RNGs as a side effect, affecting other code in the host process.
- `progress_callback` - A callback function to receive progress updates. Only supported on single-input calls; passing it alongside a `list` input raises `ValidationError`. View the [Progress Callbacks](#progress-callbacks) section for more information.
- `use_only_stem` - If specified, perform the separation using only the specialized model for this stem (a `ModelEnsemble` of fine-tuned specialists like `htdemucs_ft`). Like `only_load`, this is a **performance optimization** and does **not** filter the output to one stem — the result still contains all of the model's sources, with only the named stem at full quality. Use `SeparatedSources.isolate_stem` to actually isolate a stem. In most cases you should use `only_load` when creating the `Separator` instance instead of this.
- `chunk_batch_size` - Override the auto-detected `chunk_batch_size` for this call without persisting it. Pass `None` (default) to use `self.chunk_batch_size`.

The model's training segment length (`max_allowed_segment * samplerate`, e.g. 7.8s for HTDemucs) is used internally for every chunk; it isn't a knob because there's no useful range — shorter chunks get padded back up to that length before inference (so they're strictly slower without quality benefit) and longer chunks would extrapolate the cross-transformer's positional embeddings past their training range (degrading quality).

**Bounded GPU memory.** On CUDA, the input waveform and output accumulators stay GPU-resident when they fit a conservative fraction of the *currently free* VRAM (~30 % after a 2 GiB reserve) — the normal case for songs, where the whole separation then runs on-GPU with a single GPU→CPU transfer at the end. Inputs too long for that budget (think hours of audio) automatically fall back to CPU accumulation with per-batch GPU→CPU transfers, which bounds VRAM usage by `model + cudagraph_pool + active_batch` regardless of audio length — a 10-hour file uses the same VRAM as a 6-minute one, just with the old per-batch transfer cost. On MPS (unified memory) the accumulator always stays on-device.

**WAV fast path.** Plain 16-bit PCM WAV inputs (file path or bytes) are decoded with a direct header parse + `int16`→`float32` conversion, roughly 2x faster than and sample-exact with the torchcodec/FFmpeg path. Every other format and codec — and any malformed WAV — transparently falls back to torchcodec, so this only affects decode speed, never output.

Example:

```python
# Single input
sources = separator.separate(
    "mixture.wav",
    shifts=4,
    split_overlap=0.25,
    seed=1234,
)

# Batched list input — pools tail chunks across inputs
results = separator.separate(["a.wav", "b.wav", "c.wav"])
for sources in results:
    ...
```

## SeparatedSources

After running `Separator.separate`, you will be returned a `SeparatedSources` instance. This instance contains the separated audio sources, the sample rate of the audio, and the original audio.

### Attributes

- `sources` - Dictionary mapping stem names (e.g. `"vocals"`, `"drums"`) to their audio tensors (`dict[str, Tensor]`). You can iterate the keys to get available stem names.
- `sample_rate` - Sample rate of the separated audio (`int`), inherited from the model.
- `original` - The original unseparated audio tensor (`Tensor`).

If you're happy with the pure audio stems, you have the ability to export them to an audio container (rather than the Tensors that are stored in the `SeparatedSources` instance).

```python
def export_stem(
    self,
    stem_name: str,
    path: Path | str | None = None,
    format: str = "wav",
    clip: str | None = "rescale",
) -> Path | bytes:
```

When exporting a stem, you have the ability to specify the following parameters:

- `stem_name` - The name of the stem to export.
- `path` - The path to save the stem to. If not provided, the stem will be returned as raw audio bytes.
- `format` - The format to export the stem to. Anything supported by FFmpeg. Only used when returning bytes or when `path` has no extension; a `path` with an extension determines the container itself.
- `clip` - The clipping mode to use to prevent audio distortion. One of `"rescale"` (default — divide by `1.01 * max(|x|)` when above unity), `"clamp"` (hard clip to `±0.99`), `"tanh"` (soft clip), or `None` (no clipping).

However, Demucs provides an option to be able to isolate a single stem from the `SeparatedSources` instance. This returns a new `SeparatedSources` instance with the chosen stem and an accompanying complement stem (no_{STEM}) that is the sum of all other stems.

```python
def isolate_stem(self, name: str) -> "SeparatedSources":
```

## Auto Model Selection

As Demucs provides many models to perform audio source separation, it is often difficult to know which model to use for a given task. Demucs provides a function to attempt to select the best model for a given task.

```python
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
| `guitar`, `piano` | `htdemucs_6s` | `None` |
| `drums` | `htdemucs` | `None` |
| anything else / `None` | `htdemucs` | `None` |

## ModelRepository

Demucs provides a `ModelRepository` class to more deeply control the model loading process. This is used internally by the `Separator` class but can be used directly to load models manually to then pass to Separator itself.

`ModelRepository` is initialized with no parameters. (i.e. `repo = ModelRepository()`)

### get_cache_info

```python
def get_cache_info(self) -> dict[str, dict]:
```

This will return a dictionary of information about the cached models.

```python
{
    "model_name": {
        "layers": {       # A dictionary mapping layer checksums to their cache information
            "checksum": {
                "path": str,       # Path to the cached layer file
                "size_bytes": int, # Size of the layer in bytes
            }
        },
        "size_bytes": int, # Total size of the model in bytes
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
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to bag-of-models like htdemucs_ft).
- `progress_callback` - Optional, a callback function to receive progress updates. View the [Progress Callbacks](#progress-callbacks) section for more information.

This will return either a `Model` or `ModelEnsemble` instance corresponding to the given model name.

### list_models

```python
def list_models(self) -> dict[str, dict]:
```

This will return a dictionary of all available models.

```python
{
    "model_name": {
        "sources": list, # Stem names, in output order
        "models": list,  # Layer entries, each {"checksum": str, "remote": str, "sha256": str}
        "weights": list, # Bag-of-models only: per-source mixing weights
    }
}
```

### remove_model

```python
def remove_model(self, name: str) -> bool:
```

Pass in the name of the model you would like to remove and it will remove the weights from the filesystem.

### get_cache_dir

A module-level function (not a `ModelRepository` method), imported directly:

```python
from demucs.repo import get_cache_dir

def get_cache_dir() -> Path:
```

This will return the directory where the models are cached. This path is fully resolved.

## Progress Callbacks

Demucs provides a callback-based system for monitoring progress during long-running operations like model downloads and audio processing. This system is designed to be UI-agnostic, allowing you to implement a progress display into your own CLI or other application.

All Demucs progress callbacks are designed to use the same API. You should implement a method that matches the following signature:

```python
def progress_callback(event: str, data: dict[str, Any]) -> Any:
    pass
```

### Model Downloading

When using `ModelRepository.get_model` (or creating a `Separator` which calls it internally), the callback receives the following events:

- `download_start`: Fired when the download process begins.
  - `model_name`: Name of the model being downloaded.
  - `total_layers`: Total number of layers to download.
- `layer_start`: Fired when a specific layer starts downloading.
  - `model_name`: Name of the model.
  - `layer_index`: Index of the current layer (1-based).
  - `total_layers`: Total number of layers.
  - `layer_size_bytes`: Size of the layer in bytes.
- `layer_progress`: Fired periodically during download and loading.
  - `model_name`: Name of the model.
  - `layer_index`: Index of the current layer.
  - `total_layers`: Total number of layers.
  - `progress_percent`: Percentage complete (0-100).
  - `downloaded_bytes`: Bytes downloaded so far.
  - `total_bytes`: Total bytes to download.
  - `phase`: Optional. Set to "verifying" during checksum verification.
- `layer_complete`: Fired when a layer is successfully loaded and cached.
  - `model_name`: Name of the model.
  - `layer_index`: Index of the current layer.
  - `total_layers`: Total number of layers.
  - `cached`: Optional. True if the layer was found in cache.
- `download_complete`: Fired when all layers are downloaded and loaded.
  - `model_name`: Name of the model.
  - `total_layers`: Total number of layers.

### Audio Separation

When using `Separator.separate`, the callback receives the following events:

- `processing_start`: Fired before processing segments.
  - `total_chunks`: Total number of segments to process.
- `chunk_complete`: Fired after each segment is processed.
  - `completed_chunks`: Number of segments completed so far.
  - `total_chunks`: Total number of segments.
- `processing_complete`: Fired after all segments are processed.
  - `total_chunks`: Total number of segments.

## Version

You can get the version of the `demucs` package you have installed:

```python
def get_version() -> str:
```

Returns the version string (e.g. `"1.0.0"`).

## Other Exports

`demucs` re-exports a handful of lower-level symbols from `demucs/__init__.py` for callers who want to drive inference below the `Separator` layer. Most users should stick with `Separator`; these are intentionally minimally documented.

### Models

- `Model` — base `nn.Module` returned by `ModelRepository.get_model` for a single-layer model (e.g. plain `htdemucs`). Has `.sources` (stem names), `.samplerate`, `.audio_channels`.
- `ModelEnsemble` — `nn.Module` returned for bag-of-models like `htdemucs_ft`. Holds `.models` (list[Model]), `.weights` (per-source mixing rows), and the shared `.sources` / `.samplerate` / `.audio_channels`.

### Lower-level apply

```python
from demucs import apply_model, apply_model_multi
```

```python
def apply_model(
    model: Model | ModelEnsemble,
    mix: Tensor | TensorChunk,
    device: str | torch.device | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    use_only_stem: str | None = None,
    chunk_batch_size: int = 1,
) -> Tensor:
```

```python
def apply_model_multi(
    model: Model | ModelEnsemble,
    mixes: list[Tensor | TensorChunk],
    device: str | torch.device | None = None,
    shifts: int = 0,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    use_only_stem: str | None = None,
    chunk_batch_size: int = 1,
) -> list[Tensor]:
```

`apply_model_multi` is the batched variant that pools tail chunks across inputs so every forward pass runs at full `chunk_batch_size`. `apply_model` is a thin single-input wrapper around it. Both expect raw `[channels, samples]` or `[batch, channels, samples]` tensors (already normalized — `Separator` handles normalization internally).

### Device

```python
from demucs import default_device

def default_device() -> str:
```

Returns `"cuda"`, `"mps"`, or `"cpu"`, whichever is available — the same selection `Separator(device=None)` uses.

### Exceptions

All raised exceptions derive from `DemucsError`:

- `DemucsError` — base class for everything raised by `demucs`.
- `ValidationError` — invalid argument (bad device, bad dtype, unknown stem, out-of-range parameter).
- `ModelLoadingError` — model not found, metadata malformed, sha256 mismatch, download failure.
- `LoadAudioError` — input audio could not be decoded.

```python
from demucs import DemucsError, ValidationError, ModelLoadingError, LoadAudioError
```
