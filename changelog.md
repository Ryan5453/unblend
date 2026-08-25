## Unreleased

### Added

- Native CUDA backend (`unblend.cuda`): fused FP16/BF16 kernels for the
  GroupNorm/GELU/GLU chains in HTDemucs and SCNet, a fused DConv envelope,
  channel-last variants for `channels_last` inputs, and a fused interleaved
  rotary-position kernel for BS-RoFormer / Mel-Band RoFormer.
  Activated automatically for eager FP16/BF16 inference on CUDA; models with
  nothing eligible skip extension compilation. Requires nvcc matching the
  installed torch wheel; without it, falls back to native PyTorch ops.
- Custom-kernel kill switch: `Separator(custom_kernels=False)` (or the CLI's
  `--native-ops`, or `UNBLEND_CUSTOM_KERNELS=0`) forces vanilla PyTorch ops on
  every device — fused Metal shaders off on MPS, CUDA kernels and the fused
  rotary path off on CUDA — as the baseline for A/B benchmarks. It also skips
  the one-time nvcc build entirely.
- The one-time CUDA kernel compilation now overlaps the checkpoint download:
  eligible loads warm the extension in a background thread while weights
  transfer, so a fresh environment pays roughly max(download, build) instead
  of the sum. A warning names the stall when a build blocks synchronously.
- Custom models can use any architecture Unblend implements, HTDemucs
  included, and take their weights from either a local file or an https URL.
  Local files are read where they lie; URLs are downloaded once into the model
  cache and reused, so a model hosted on Hugging Face costs one fetch rather
  than one per run. See "Custom models" in the [Python API](api.md) docs.
- Registry entries may name only their `architecture`; the backend that builds
  it is derived. `backend` remains accepted.
- Ensembles are backend-neutral: any entry can declare `members`, each with its
  own architecture, config and artifact, or naming another registered model to
  reuse it. The Demucs bags' `models` spelling still works and is unchanged.
  Members sharing a checkpoint share its cache file, so an ensemble built from
  registered models downloads nothing new.
- Ensemble combining modes (`combine`): `weighted_mean` (default, also spelled
  `avg_wave`), `median_wave`, `min_wave`, `max_wave`, `avg_fft`, `median_fft`,
  `min_fft`, `max_fft`, plus `uvr_min_spec`/`uvr_max_spec` as UVR's names for
  the last two. Names follow ZFTurbo's Music-Source-Separation-Training,
  audio-separator and UVR so existing recipes transfer. The selection modes
  require a 0/1 participation mask rather than silently ignoring weights, and
  `combine_params` sets the spectral modes' STFT geometry.
- `roformer_vocals_ensemble`: a registered ensemble of `melband_roformer_kim`
  and `bs_roformer_anvuew`, so the combine modes are usable without writing a
  metadata entry. Costs both members' inference time; licensing is the union of
  the two, as its `license_note` records.
- `Separator(combine=..., combine_params=...)` and `unblend separate --combine`
  override an ensemble's mode for one run.
- Ensemble members no longer have to share a normalization contract. HTDemucs
  wants track-level normalized audio and the other architectures want it raw;
  a mixed ensemble now takes raw audio and normalizes around the members that
  need it, so each sees what it would see running alone and members are
  combined in the input's own scale. Ensembles whose members already agree are
  unchanged, normalizing once in `Separator` as before.
- `htdemucs_scnet_ensemble`: HTDemucs plus SCNet xl-wide over the same four
  stems, which the normalization work above makes possible.
- `unblend models import`: repackage a checkpoint from elsewhere as
  Safetensors and register it. Reads the tensors out of a training framework's
  container (stripping a `model.`/`module.` prefix) with
  `torch.load(weights_only=True)`, translates a Music-Source-Separation-Training
  config, infers the architecture from the parameter names, and **builds the
  model and strict-loads it before writing anything** — so a mistranslated
  config fails at import with the mismatch named, not at separation time.
- Model registries are YAML. The shipped registry is now
  `unblend/metadata.yaml`, and `UNBLEND_EXTRA_MODELS` files, imported entries
  and the configs `models import` translates are all read as YAML — the format
  the ecosystem already writes configs in, and one that allows comments and
  readable multi-line licence notes. Files named `.json` are still read as
  JSON, so existing models files keep working. PyYAML is now a dependency.
- Imported artifacts describe themselves: architecture, stems, geometry and
  config are written into the Safetensors header, which the file's sha256
  covers. A registry entry for such a file needs only `sources` and the path.

### Changed

- Single-checkpoint backends (RoFormer, SCNet) now download, cache-report,
  and remove through one generic path instead of a RoFormer-only special case.
- Every backend resolves, verifies, caches, and removes its weights through
  one shared artifact path, replacing the parallel Demucs and single-checkpoint
  implementations. All registry entries are now fully validated when the
  repository is constructed rather than part-way through a download.
- `unblend separate --model` and `unblend tune --model` accept any registered
  model name, including models added through `UNBLEND_EXTRA_MODELS`. The
  choices were a hand-written enum that had also drifted from the shipped
  registry (it omitted `bs_roformer_anvuew` and both SCNet models).
  `--isolate-stem` likewise accepts any stem the selected model emits.
- `license` in a registry entry is documented as a free-form pass-through
  label; nothing validates or interprets it.
- Registry artifacts no longer carry a `checksum` field. It was always the
  first 16 characters of the entry's own `sha256`, which is what names the
  cache file, so it is now derived — cached files keep their names and nothing
  re-downloads.
- The single-stem shortcut now keys off which members contribute to the stem
  rather than requiring a weight of exactly 1.0, so it applies to every combine
  mode (each reduces to the identity over one member) and to ensembles whose
  specialist rows are not normalized.

### Performance

- MUSDB18-HQ paired A/B at SDR parity (≤0.013 dB): SCNet xl-wide 1.8×
  (A100) / 2.3× (H200); SCNet small 1.4×; HTDemucs/HTDemucs-ft up to 1.5×;
  BS-Roformer & Mel-Band 1.2×. Model-core speedups reach 4× on SCNet-xl.

## v1.0.0

This is the first release of Unblend. 
Please view the [Python API](https://github.com/Ryan5453/unblend/blob/main/api.md) docs, [npm package](https://github.com/Ryan5453/unblend/blob/main/web/demucs/README.md) docs, or the [ONNX export notes](https://github.com/Ryan5453/unblend/blob/main/onnx.md) for more information on how to embed Unblend in your application. 
