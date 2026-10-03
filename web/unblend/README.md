# <img src="https://raw.githubusercontent.com/Ryan5453/unblend/main/web/app/public/favicon.svg" width="30"> unblend (browser)

Run Unblend's models in the browser with onnxruntime-web. It is much slower than the [Python package](https://github.com/Ryan5453/unblend), but needs no server.

## Install

```bash
npm install unblend
```

`onnxruntime-web` is a regular dependency, so there's nothing else to install and no `<script>` tag. Workers are referenced via `new Worker(new URL('./workers/*.js', import.meta.url))`, so use a bundler that understands that pattern.

**Vite:** add `unblend` to `optimizeDeps.exclude` or esbuild mangles the worker URLs:

```ts
export default defineConfig({
  optimizeDeps: { exclude: ['unblend'] },
});
```

To keep ORT's `.wasm` out of your bundle (e.g. Cloudflare Pages' 25 MB cap), serve it from a CDN via the `wasmPaths` option. The CDN path must serve the exact `onnxruntime-web` version this package depends on (currently `1.26.0`), since ORT's JS glue and `.wasm` are versioned together and a mismatch fails at load time. With jsDelivr that is `https://cdn.jsdelivr.net/npm/onnxruntime-web@<version>/dist/`, e.g. `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/`.

TypeScript consumers need TypeScript 5.7 or newer (unless `skipLibCheck` is on): the bundled declarations use `Uint8Array<ArrayBuffer>` so model bytes can be transferred to the worker without casts.

WASM multi-threading requires a cross-origin isolated page:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Usage

```ts
import { Separator } from 'unblend';

const separator = await Separator.load('htdemucs', {
  backend: 'webgpu',   // default; falls back to 'wasm' when the model fits
  precision: 'fp32',   // 'fp16' = half the download; see options.precision
});

const audioContext = new AudioContext({ sampleRate: 44100 });
const audioBuffer = await audioContext.decodeAudioData(await file.arrayBuffer());
await audioContext.close(); // the decoded AudioBuffer stays usable

const result = await separator.separate(audioBuffer, {
  onProgress: (p) => console.log(p.fraction),
});

console.log(result.stems); // stem name → interleaved L/R Float32Array

await separator.unload();
```

Decoding is your job — `AudioContext.decodeAudioData` covers MP3/AAC/FLAC/WAV/Ogg. Output stems are interleaved `[L0, R0, L1, R1, ...]`; encode WAV yourself.

### `Separator.load(model, options?)`

- `model`: an id from the table below.
- `options.backend`: `'webgpu'` (default) | `'wasm'`. Falls back to WASM automatically unless the model requires WebGPU (`bs_roformer_sw`, `scnet_xl_wide_v5`) — those reject incompatible browsers up front.
- `options.precision`: `'fp32'` (default) | `'fp16'`. The fp16 files are about half the size. For HTDemucs they are bit-identical to fp32 (the checkpoints are fp16 already). The others stay close but not identical, and a little further on WebGPU than on the CPU. SCNet's fp16 files store fp16 weights but compute in fp32, and land about 53–74 dB SNR from fp32 per stem. The RoFormers' fp16 files also compute in fp16: `melband_roformer_kim` lands about 45–65 dB from fp32 and `bs_roformer_sw` about 15–50 dB. A nearly silent stem can score much lower, since there is little signal for the error to be measured against. All of these are well below what separation itself gets wrong: on MUSDB18-HQ the fp16 and fp32 files score the same SDR.
- `options.wasmPaths`: URL prefix for ORT `.wasm` assets if not bundling them. It must serve the same `onnxruntime-web` version as the installed dependency (`1.26.0`), e.g. `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/`.
- `options.numThreads`: WASM thread count (default 4).
- `options.signal`: `AbortSignal` for load cancellation.
- `options.onProgress`: `(phase: 'download' | 'compile', loaded, total, source?) => void` — download bytes, then one `'compile'` call while ORT builds the session. During `'download'`, `source` is `'cache'` when the weights are read from Cache Storage and `'network'` otherwise.
- `options.cache`: keep the weights in Cache Storage so later loads skip the download (default `true`). Entries are keyed by the immutable artifact URL; only complete downloads are stored, and loading continues uncached when Cache Storage is unavailable or full. Clear them with `clearModelCache()`.
- `options.modelUrl`: fetch the `.onnx` from this URL instead of the registered artifact, e.g. to test a local export. Never cached, and its size is not checked. The pipeline does not adapt to the file: it always uses the built-in STFT, window and chunk geometry of `model`, so the URL must point to an export of that same model. The file's embedded metadata (`stft_n_fft`, `stft_hop_length`, `stft_win_length`, `stft_window`, `stft_normalized`, `segment_samples`, `sources`, `num_stems`, ...) is checked at load time: a missing key or a contradiction rejects with a `'fetch'`-stage `ModelLoadError`, so the file must be an export from the current `unblend export-onnx`. Export RoFormer models with `--static-batch`, as the published files are: the metadata check doesn't catch a dynamic-batch RoFormer export, and WebGPU mis-plans its memory.
- `options.graphOptimizationLevel`: `'disabled'` | `'basic'` | `'extended'` | `'all'` (default), for diagnosing ORT optimizer bugs.

A registered artifact whose download length differs from its published size is rejected rather than handed to ORT. Download failures (network, HTTP status, wrong size, metadata that contradicts the model) reject with a `ModelLoadError` whose `stage` is `'fetch'`; failures creating the ORT session have `stage: 'session'`.

Each instance owns its workers and rejects concurrent `separate()` calls. Aborting a separation, unloading, or any failure once a separation has started (a worker error, an unexpected model output shape, and so on) terminates the workers and invalidates the instance: later `separate()` calls reject, so load a fresh one to retry. Input validation errors thrown before the run starts (wrong sample rate, empty buffer, bad `shifts`/`seed`) leave the instance usable.

### `separator.separate(audioBuffer, options?)`

- `options.onProgress`: `(p: SeparationProgress) => void` — `{ stage, segIdx, totalSegs, fraction }`, fired as each segment starts and completes.
- `options.signal`: aborts destructively (invalidates the instance).
- `options.shifts`: random sub-second shifts to average, 0–20 (default 1). `0` runs one unshifted pass, deterministic without a seed. Runtime scales linearly with the number of passes.
- `options.seed`: integer seed making shifts (and outputs) deterministic. JS-only parity, independent of Python's RNG.

Returns `{ stems, wallMs, inferenceMs, numSegments }`.

Instance properties: `.model`, `.sources`, `.license`, `.backend`, `.precision`.

### `clearModelCache()`

```ts
import { clearModelCache } from 'unblend';

await clearModelCache(); // true if a cache existed and was deleted
```

Deletes every cached model (the `MODEL_CACHE_NAME` Cache Storage bucket, `'unblend-models'`). Already-loaded separators keep working; the next `Separator.load` downloads again. Resolves `false` when there was nothing to delete or Cache Storage is unavailable.

## Input requirements

- **Sample rate:** exactly 44.1 kHz — STFT parameters are baked into the graphs. Resample with `OfflineAudioContext` first.
- **Channels:** mono is duplicated; beyond two channels only the first two are used. Output is always 2 channels per stem.

## Memory

Full-track buffers dominate page memory on long inputs. Every stereo float32 copy of the track costs about 21 MB per minute of audio (44,100 × 2 × 4 bytes per second), and a separation holds one per output stem plus a normalized copy of the input, on top of your decoded `AudioBuffer`. Plan on roughly `(stems + 2) × 21 MB` per minute: about 127 MB/min for 4-stem models and 170 MB/min for `htdemucs_6s`/`bs_roformer_sw`, so a 10-minute 6-stem run peaks near 1.7 GB of main-thread memory. This does not count the model and its activations, which live in the ONNX worker (or on the GPU) and are sized by the model, not the track length.

Mobile browsers may kill a tab well before that. To keep the peak down, release your `AudioBuffer` once `separate()` resolves, and encode each stem then drop your reference to its `Float32Array` as you go rather than holding every stem alongside its encoded copy.

## Models

| Model | Stems | Family | Weights license |
|---|---|---|---|
| `htdemucs` | drums, bass, other, vocals | HTDemucs | MIT |
| `htdemucs_6s` | + guitar, piano | HTDemucs | MIT |
| `bs_roformer_sw` | bass, drums, other, vocals, guitar, piano | BS-RoFormer | unlicensed |
| `melband_roformer_kim` | vocals, other¹ | Mel-Band RoFormer | MIT |
| `scnet_small` | drums, bass, other, vocals | SCNet Masked Small | unlicensed |
| `scnet_xl_wide_v5` | drums, bass, other, vocals | SCNet XL IHF | unlicensed |

¹ Computed client-side as `mixture - vocals`.

The RoFormer models are markedly higher quality but larger (~350–480 MB fp16) and slower. The package's MIT license covers its code, not the weights. Surface `separator.license` in your app, and see `unblend models info NAME` in the Python CLI for the full terms.

Artifacts are hosted on Hugging Face under immutable revisions; sizes and SHA-256 digests live in [`src/model-artifacts.ts`](https://github.com/Ryan5453/unblend/blob/main/web/unblend/src/model-artifacts.ts), exported as `MODEL_ARTIFACTS` (`MODEL_ARTIFACTS[model][precision].sizeBytes` gives the download size) (verified against Hugging Face at release time, not in the browser).

## Constants

`SAMPLE_RATE`, `SEGMENT_OVERLAP`, `MODEL_CONFIGS` (per-model DSP geometry), `specDims(config)` — use `MODEL_CONFIGS[model]` for a model's geometry. The config objects are frozen.
