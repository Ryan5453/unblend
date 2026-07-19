# unblend API

The unblend Python API is primarily comprised of two classes: `Separator` and `SeparatedSources`.

## Separator

The `Separator` class is a high level representation of an audio source separation model. When you want to separate an audio file into its constituent stems, you will first need to create an instance of the `Separator` class which will load the model into memory for use.

```python
separator = Separator(
    model: str | Model | ModelEnsemble = "htdemucs",
    device: str | None = None,
    only_load: str | None = None,
    dtype: torch.dtype | str | None = "auto",
    compile: bool = False,
    chunk_batch_size: int | None = None,
)
```

A `Separator` takes the following parameters:

- `model` - The model to use for separation. While just passing in a string is the easiest, you can use `ModelRepository` to load models manually and then pass them in.
- `device` - The device/backend to use for loading and running the model. If left as `None` (the default), unblend auto-selects the best available backend at construction time (cuda > mps > cpu). Pass `"cpu"`, `"cuda"`, or `"mps"` to force one.
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to bag-of-models like htdemucs_ft). This is a **performance optimization** (smaller download and memory footprint) — it does **not** filter the output to one stem; the result still contains all of the model's sources, with only the named stem at full quality. Use `SeparatedSources.isolate_stem` to actually isolate a stem.
- `dtype` - Inference precision. The default `"auto"` uses FP16 on CUDA GPUs with tensor cores (compute capability ≥ 7.0) and on MPS; CPU and older CUDA GPUs use FP32. HTDemucs and RoFormer FP16 both measure SDR-equal to FP32. RoFormer gains are 2.2–2.4× on a V100 and 1.06–1.07× on an M2 Max after the MPS attention/RMSNorm optimizations (10 full MUSDB18-HQ tracks). Pass `torch.float16` or `torch.bfloat16` explicitly to force reduced precision (CUDA/MPS only; CPU is rejected), or `None` / `torch.float32` to force FP32. On MPS, custom Metal kernels accelerate normalization; BF16 works but measured ~27% slower than FP16 for HTDemucs.
- `compile` - Optional, if `True`, applies `torch.compile` (Inductor/CUDAGraphs) to the architecture's heavy neural-network core on CUDA: `forward_core` for HTDemucs and the axial transformer trunk for BS-/Mel-Band RoFormer. STFT/iSTFT and reconstruction remain eager. On a V100 FP16 compile measured 1.50× for SW and 1.34× for Kim on a 76-second track. Compilation adds initialization latency and a persistent CUDAGraph private memory pool; ensemble members each capture their own pool, which can be much larger than `torch.cuda.max_memory_allocated` reports. Auto-sized runs halve/recapture on OOM. MPS and CPU remain eager. The Python API is explicit (`compile=False`); the CLI defaults to cache-free workload-aware auto mode, with `--compile` / `--no-compile` overrides.
- `chunk_batch_size` - Optional explicit chunks-per-forward batch size, bypassing auto-detection. Auto (`None`, the default) sizes from measured memory and degrades instead of dying: eager runs halve and retry the failed batch (sticky down to 1), while compiled runs tear down and recapture at half (bounded at 4 recaptures per request). An explicit value is respected exactly—no halving, OOM raises—and under compile fixes the captured shape, so per-call overrides are rejected.

### Attributes

After construction, the following attributes are available on a `Separator` instance:

- `device` - The device being used for processing (`str`).
- `dtype` - The dtype being used for inference (`torch.dtype | None`).
- `model` - The loaded model instance (`Model | ModelEnsemble`).
- `audio_channels` - Number of audio channels the model expects (`int`).
- `sample_rate` - Sample rate the model operates at (`int`).
- `chunk_batch_size` - Number of segments processed per forward call. On CUDA this is auto-detected from a warmed batch-1 eager memory/timing probe plus `mem_get_info`; compile additionally capture-verifies and halves on OOM. The CLI reuses the timing for its auto-compile decision. On MPS it is sized from the unified-memory budget (8 / 4 / 2 as `torch.mps.recommended_max_memory()` crosses 20 / 10 GB); CPU defaults to 1.

If you enable `compile=True`, warmup happens automatically at the end of `__init__` (via a zero-tensor pass through `Separator.separate`, so the CUDAGraph captured is the same one real requests reuse). You can call `separator.warmup()` again later to re-prime if needed; the method takes no arguments because tail-padding inside `apply_model` guarantees a single batch shape per session.

```python
separator.warmup()  # no args — there's exactly one batch shape after tail-padding
```

`warmup()` is CUDA-only: it raises `ValidationError` on CPU/MPS or models outside the HTDemucs/RoFormer compile targets. Workload-aware callers can instead construct eagerly, inspect their job, and call `separator.enable_compile()` to compile/capture the existing CUDA model in place without reloading weights; repeated calls are no-ops.

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

- `audio` - The audio to separate. **Polymorphic input**: a single `(Tensor, sample_rate)` tuple / file path / raw bytes returns a single `SeparatedSources`. Passing a `list` of those returns `list[SeparatedSources]` and pools tail chunks across inputs (so every forward pass runs at full `chunk_batch_size`, no wasted slots). Useful when serving many short clips concurrently — see `apply_model_multi` in `unblend/apply.py`.
- `shifts` - The number of random shifts for equivariant stabilization. In simple terms, this is a technique to make the model more robust to small changes in the audio, such as small shifts in time or pitch. More shifts mean generally higher quality separation but also longer processing time. Must be an integer in `[1, 20]`.
- `split_overlap` - The overlap between consecutive segments. Must be in the range `[0.0, 1.0)`. Higher values smooth segment boundaries at the cost of more compute per track.
- `seed` - Optional random seed for reproducible shift-based inference. With list input, per-input shift offsets advance from this seed in sequence — outputs are reproducible across runs at the same seed, but a list call with `seed=N` does NOT produce bit-identical outputs to N separate single-file calls with `seed=N`. Setting this also reseeds the process-global `random` and `torch` RNGs as a side effect, affecting other code in the host process.
- `progress_callback` - A callback function receiving aggregate and per-input progress for both single and list input. List-input events remain one monotonic global stream while identifying the input advanced by each completed chunk. View the [Progress Callbacks](#progress-callbacks) section for more information.
- `use_only_stem` - If specified, perform the separation using only the specialized model for this stem (a `ModelEnsemble` of fine-tuned specialists like `htdemucs_ft`). Like `only_load`, this is a **performance optimization** and does **not** filter the output to one stem — the result still contains all of the model's sources, with only the named stem at full quality. Use `SeparatedSources.isolate_stem` to actually isolate a stem. In most cases you should use `only_load` when creating the `Separator` instance instead of this.
- `chunk_batch_size` - Override the auto-detected `chunk_batch_size` for this call without persisting it. Pass `None` (default) to use `self.chunk_batch_size`.

The model's training segment length (`max_allowed_segment * samplerate`, e.g. 7.8s for HTDemucs) is used internally for every chunk; it isn't a knob because there's no useful range — shorter chunks get padded back up to that length before inference (so they're strictly slower without quality benefit) and longer chunks would extrapolate the cross-transformer's positional embeddings past their training range (degrading quality).

**Bounded GPU memory.** On CUDA, the input waveform and output accumulators stay GPU-resident when they fit a conservative fraction of the *currently free* VRAM (~30 % after a reserve of at least 2 GiB, grown to cover the measured per-batch forward working set) — the normal case for songs, where the whole separation then runs on-GPU with a single GPU→CPU transfer at the end. Inputs too long for that budget (think hours of audio) automatically fall back to CPU accumulation with per-batch GPU→CPU transfers, which bounds VRAM usage by `model + cudagraph_pool + active_batch` regardless of audio length — a 10-hour file uses the same VRAM as a 6-minute one, just with the old per-batch transfer cost. On MPS (unified memory) the accumulator always stays on-device.

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

# Batched list input — pools tail chunks across inputs and supports progress
results = separator.separate(
    ["a.wav", "b.wav", "c.wav"],
    progress_callback=progress_callback,
)
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

However, unblend provides an option to be able to isolate a single stem from the `SeparatedSources` instance. This returns a new `SeparatedSources` instance with the chosen stem and an accompanying complement stem (no_{STEM}) that is the sum of all other stems.

```python
def isolate_stem(self, name: str) -> "SeparatedSources":
```

## Auto Model Selection

As unblend provides many models to perform audio source separation, it is often difficult to know which model to use for a given task. unblend provides a function to attempt to select the best model for a given task.

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

unblend provides a `ModelRepository` class to more deeply control the model loading process. This is used internally by the `Separator` class but can be used directly to load models manually to then pass to Separator itself.

`ModelRepository` is initialized with no required parameters. (i.e. `repo = ModelRepository()`)

### get_cache_info

```python
def get_cache_info(self) -> dict[str, dict]:
```

This will return a dictionary of information about the cached models. Models with at least one cached layer are included, so a partially-downloaded model shows up with `"complete": False`.

```python
{
    "model_name": {
        "layers": {       # A dictionary mapping cached layer checksums to their cache information
            "checksum": {
                "path": str,       # Path to the cached layer file
                "size_bytes": int, # Size of the layer in bytes
            }
        },
        "size_bytes": int,  # Total size of the cached layers in bytes
        "total_layers": int, # Number of layers the model has in metadata
        "complete": bool,    # True when every layer is cached
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

Pass in the name of the model you would like to remove and it will remove the weights from the filesystem. Returns `True` if anything was removed, `False` for an unknown model or an empty cache; raises `ModelLoadingError` if a cached layer can't be removed (e.g. permissions).

### get_cache_dir

A module-level function (not a `ModelRepository` method), imported directly:

```python
from unblend.repo import get_cache_dir

def get_cache_dir() -> Path:
```

This will return the directory where the models are cached (created on first download). Set the `UNBLEND_CACHE_DIR` environment variable to relocate it (default: `~/.unblend/models`); the value is tilde-expanded and resolved.

## Progress Callbacks

unblend provides a callback-based system for monitoring progress during long-running operations like model downloads and audio processing. This system is designed to be UI-agnostic, allowing you to implement a progress display into your own CLI or other application.

All unblend progress callbacks are designed to use the same API. You should implement a method that matches the following signature:

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

You can get the version of the `unblend` package you have installed:

```python
def get_version() -> str:
```

Returns the version string (e.g. `"1.0.0"`).

## Other Exports

`unblend` re-exports a handful of lower-level symbols from `unblend/__init__.py` for callers who want to drive inference below the `Separator` layer. Most users should stick with `Separator`; these are intentionally minimally documented.

### Models

- `Model` — base `nn.Module` returned by `ModelRepository.get_model` for a single-layer model (e.g. plain `htdemucs`). Has `.sources` (stem names), `.samplerate`, `.audio_channels`.
- `ModelEnsemble` — `nn.Module` returned for bag-of-models like `htdemucs_ft`. Holds `.models` (list[Model]), `.weights` (per-source mixing rows), and the shared `.sources` / `.samplerate` / `.audio_channels`.

### Lower-level apply

```python
from unblend import apply_model, apply_model_multi
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

All raised exceptions derive from `UnblendError`:

- `UnblendError` — base class for everything raised by `unblend`. Also importable as `DemucsError` (a backward-compatible alias).
- `ValidationError` — invalid argument (bad device, bad dtype, unknown stem, out-of-range parameter).
- `ModelLoadingError` — model not found, metadata malformed, sha256 mismatch, download failure.
- `LoadAudioError` — input audio could not be decoded.

```python
from unblend import UnblendError, ValidationError, ModelLoadingError, LoadAudioError
```
