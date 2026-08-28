# Unblend API

The Python API is primarily comprised of two classes: `Separator` and `SeparatedSources`.

## Separator

The `Separator` class is a high level representation of an audio source separation model. When you want to separate an audio file into its constituent stems, you will first need to create an instance of the `Separator` class which will load the model into memory for use.

```python
from unblend import Separator 

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
- `device` - The device/backend to use for loading and running the model. If left as `None` (the default), unblend auto-selects the best available backend at construction time. Pass `"cpu"`, `"cuda"`, or `"mps"` to force one.
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to ensembles like htdemucs_ft). This is a performance optimization (smaller download and memory footprint), it does not filter the output to one stem - use `SeparatedSources.isolate_stem` to actually isolate a stem.
- `dtype` - Inference precision. The default `"auto"` uses FP16 on CUDA GPUs with tensor cores (compute capability ≥ 7.0) and on MPS; CPU and older CUDA GPUs use FP32.
- `compile` - Optional, if `True`, compiles the model's neural-network core on CUDA — roughly 1.3–1.5× in FP16, at the cost of startup time and extra held VRAM, so it pays off on long jobs. CPU and MPS ignore it. The CLI decides per workload instead, with `--compile` / `--no-compile` to force either way.
- `chunk_batch_size` - Optional, how many segments to run per forward pass. The default (`None`) sizes it from available memory and backs off if that proves too large; an explicit value is used exactly as given, and OOM raises.

### Attributes

After construction, the following attributes are available on a `Separator` instance:

- `device` - The device being used for processing (`str`).
- `dtype` - The dtype being used for inference (`torch.dtype | None`).
- `model` - The loaded model instance (`Model | ModelEnsemble`).
- `audio_channels` - Number of audio channels the model expects (`int`).
- `sample_rate` - Sample rate the model operates at (`int`).
- `chunk_batch_size` - Number of segments processed per forward call (`int`). Measured per device at construction unless you passed an explicit value.

If you enable `compile=True`, warmup happens automatically at the end of `__init__`. Call `separator.warmup()` to re-prime later if needed; it takes no arguments because tail-padding leaves exactly one batch shape per session.

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

- `audio` - The audio to separate. A single `(Tensor, sample_rate)` tuple, file path, or raw bytes returns one `SeparatedSources`; a `list` of those returns `list[SeparatedSources]` and is the efficient way to serve many short clips at once. Tuple tensors must be floating-point audio already at the usual amplitude range — integer PCM, boolean, and complex tensors are rejected rather than silently reinterpreted.
- `shifts` - How many randomly time-shifted passes to average, which stabilizes the output at the cost of proportionally more compute. Must be an integer in `[1, 20]`.
- `split_overlap` - The overlap between consecutive segments. Must be in the range `[0.0, 1.0)`. Higher values smooth segment boundaries at the cost of more compute per track.
- `seed` - Optional random seed for reproducible shift-based inference. With list input, per-input shift offsets advance from this seed in sequence — outputs are reproducible across runs at the same seed, but a list call with `seed=N` does NOT produce bit-identical outputs to N separate single-file calls with `seed=N`. Setting this also reseeds the process-global `random` and `torch` RNGs as a side effect, affecting other code in the host process.
- `progress_callback` - A callback function receiving aggregate and per-input progress for both single and list input. List-input events remain one monotonic global stream while identifying the input advanced by each completed chunk. View the [Progress Callbacks](#progress-callbacks) section for more information.
- `use_only_stem` - Run only the specialist member for this stem, in an ensemble like `htdemucs_ft`. Like `only_load` this is a **performance optimization**, not a filter — every source is still returned, with only the named one at full quality. Prefer `only_load` when you know the stem before constructing the `Separator`.
- `chunk_batch_size` - Override the auto-detected `chunk_batch_size` for this call without persisting it. Pass `None` (default) to use `self.chunk_batch_size`. A compiled separator captures one fixed batch shape, so per-call overrides are rejected there — pass it to `Separator(...)` instead.

The model's training segment length (`max_allowed_segment * samplerate`, e.g. 7.8s for HTDemucs) is used internally for every chunk; it isn't a knob because there's no useful range — shorter chunks get padded back up to that length before inference (so they're strictly slower without quality benefit) and longer chunks would extrapolate the cross-transformer's positional embeddings past their training range (degrading quality).

**Bounded GPU memory.** VRAM use does not grow with audio length. Songs keep their accumulators on the GPU and pay one transfer at the end; inputs too long for the free-VRAM budget fall back to CPU accumulation with per-batch transfers, so a 10-hour file needs the same VRAM as a 6-minute one and simply runs a little slower. On MPS the accumulator always stays on-device.

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
- `only_load` - Optional, if specified, load only the specialized model for this stem (only applicable to multi-member entries like htdemucs_ft).
- `progress_callback` - Optional, a callback function to receive progress updates. View the [Progress Callbacks](#progress-callbacks) section for more information.

This will return either a `Model` or `ModelEnsemble` instance corresponding to the given model name.

### list_models

```python
def list_models(self) -> dict[str, dict]:
```

This returns deep copies of the registered metadata as a tagged union. Inspect
`backend` before reading backend-specific fields:

```python
{
    "htdemucs": {
        "backend": "demucs",
        "architecture": "htdemucs",
        "sources": list,  # Stem names, in output order
        "checkpoint": {"url": str, "sha256": str, "size_bytes": int},
        "weights": list | None,  # Mixing weights, for a multi-member entry
        "config": dict,
    },
    "bs_roformer_sw": {
        "backend": "roformer",
        "architecture": "bs_roformer" | "mel_band_roformer",
        "sources": list,
        "checkpoint": {"url": str, "sha256": str, "size_bytes": int},
        "config": dict,
        "samplerate": int,
        "segment_samples": int,
    },
}
```

`get_cache_info()` uses the same common cache shape for both families. Demucs
`layers` keys are registered checksum prefixes; a RoFormer has one layer keyed
by the first 16 characters of its checkpoint SHA-256.

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
- `ModelEnsemble` — `nn.Module` returned for multi-member entries like `htdemucs_ft`. Holds `.models` (list[Model]), `.weights` (per-source mixing rows), and the shared `.sources` / `.samplerate` / `.audio_channels`.

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

Point `UNBLEND_EXTRA_MODELS` at a models file (or pass `extra_models=` to
`ModelRepository`). Its entries are **added** to the shipped registry — a file
that reuses a built-in name is rejected rather than shadowing it, so dropping
one in cannot silently swap the weights behind `htdemucs`.

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
export UNBLEND_EXTRA_MODELS=~/my-models.json
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
| `license` | Free-form label, passed through to `models list` and `list_models` untouched. Unblend does not interpret it. |
| `weights` | For an ensemble: one row per member, one column per source. Defaults to all ones. |
| `combine`, `combine_params` | For an ensemble: how member outputs are combined (see below). |
| `segment` | Optional: shortens (never enlarges) the configured training segment. |

An entry names its weights exactly once: `checkpoint` for one set, `members`
for two or more. There is no `backend` field — the loader family is derived
from `architecture`, and declaring it is an error rather than a second source
of truth that can disagree. Every architecture belongs to exactly one family:
`htdemucs` to `demucs`, the two RoFormers to `roformer`, the two SCNets to
`scnet`. `list_models` reports the derived value back to you.

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

- A **local `path`** is read where it lies. Nothing is copied into the cache, so
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

A member can also just name another registered model, which is how you ensemble
models that already ship without restating their config:

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
file, so an ensemble of registered models downloads nothing new (and
`models remove` on it removes those shared files).

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
| `weighted_mean` (default), `avg_wave` | Per-source weighted average of the waveforms. |
| `median_wave` | Element-wise median of the waveforms. |
| `min_wave`, `max_wave` | The sample with the smallest/largest absolute value, sign kept. |
| `avg_fft` | Weighted average of the complex spectrograms. |
| `median_fft` | Per bin, the member ranked in the middle by magnitude. |
| `min_fft`, `max_fft` | Per bin, the whole complex value from the member with the smallest/largest magnitude. |
| `uvr_min_spec`, `uvr_max_spec` | UVR's names for `min_fft`/`max_fft` — the same operation. |

`combine_params` sets the STFT geometry for the spectral modes; it defaults to
`{"n_fft": 1024, "hop_length": 256}`, and `n_fft` must be a whole multiple of
`hop_length`.

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
- **`avg_wave` is the quality default.** ZFTurbo's own testing found the plain
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
✓ Loaded as mel_band_roformer and strict-loaded 1219 tensors
✓ Wrote ~/.unblend/imported/my_model.safetensors (869.4 MB)
✓ Registered my_model in ~/.unblend/models.json

Try it: unblend separate --model my_model track.wav
```

What it does:

- **Reads the tensors out of whatever container they arrived in** — a bare
  state dict, or a training framework's checkpoint with the weights under
  `state_dict` and a `model.` / `module.` prefix on every key. It loads with
  `torch.load(weights_only=True)`, so a checkpoint that needs real unpickling
  is refused rather than executed.
- **Translates the config.** ZFTurbo's Music-Source-Separation-Training layout
  is understood directly: `model:` is the constructor config, `audio.chunk_size`
  and `audio.sample_rate` are the geometry, `training.instruments` are the
  stems. A `training.target_instrument` marks a single-head model, whose second
  stem is synthesised as `mixture - prediction`. Anything the config does not
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
- **Writes a self-describing artifact.** The registry fields go into the
  Safetensors header, which the file's own sha256 covers — so an embedded
  config is verified along with the weights, which a models file beside it is
  not. A hand-written entry for such a file needs only its `sources` and where
  it is; the rest is read from the header (a few KB, whatever the file's size).
- **Registers and re-loads it**, so the entry is known to survive the
  registry's validation. `--print` emits the entry instead of writing it, and
  `--models-file` chooses where it lands (default: `$UNBLEND_EXTRA_MODELS`, else
  `~/.unblend/models.yaml`).

Only architectures Unblend implements can load at all. MDX-Net and VR-arch
weights (much of UVR's model list, including everything shipped as `.onnx`) are
different architectures, not different packaging — the import says so rather
than failing obscurely.

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
