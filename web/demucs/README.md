# Unblend Browser API

Unblend implements a browser API for easy access of the models. However, it is significantly slower than the native implementation.

## Install

```bash
npm install unblend
```

`onnxruntime-web` is bundled — no peer install, no `<script>` tag. Workers are referenced via `new Worker(new URL('./workers/*.js', import.meta.url))`, so use a bundler that understands that pattern.

**Vite:** add `unblend` to `optimizeDeps.exclude` or esbuild mangles the worker URLs:

```ts
export default defineConfig({
  optimizeDeps: { exclude: ['unblend'] },
});
```

To keep ORT's `.wasm` out of your bundle (e.g. Cloudflare Pages' 25 MB cap), serve it from a CDN via the `wasmPaths` option.

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
  precision: 'fp32',   // 'fp16' = half the download, near-identical output
});

const audioBuffer = await new AudioContext({ sampleRate: 44100 })
  .decodeAudioData(await file.arrayBuffer());

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
- `options.precision`: `'fp32'` (default) | `'fp16'`.
- `options.wasmPaths`: URL prefix for ORT `.wasm` assets if not bundling them.
- `options.numThreads`: WASM thread count (default 4).
- `options.signal`: `AbortSignal` for load cancellation.

Each instance owns its workers and rejects concurrent `separate()` calls. Aborting a separation or unloading invalidates the instance — load a fresh one to retry.

### `separator.separate(audioBuffer, options?)`

- `options.onProgress`: `(p: SeparationProgress) => void` — `{ segIdx, totalSegs, fraction }`.
- `options.signal`: aborts destructively (invalidates the instance).
- `options.shifts`: random sub-second shifts to average, 1–20 (default 1). Runtime scales linearly.
- `options.seed`: integer seed making shifts (and outputs) deterministic. JS-only parity, independent of Python's RNG.

Returns `{ stems, wallMs, inferenceMs, numSegments }`.

Instance properties: `.model`, `.sources`, `.license`, `.backend`, `.precision`.

## Input requirements

- **Sample rate:** exactly 44.1 kHz — STFT parameters are baked into the graphs. Resample with `OfflineAudioContext` first.
- **Channels:** 1 or 2. Mono is duplicated internally; output is always 2 channels per stem.

## Models

| Model | Stems | Family |
|---|---|---|
| `htdemucs` | drums, bass, other, vocals | HTDemucs |
| `htdemucs_6s` | + guitar, piano | HTDemucs |
| `bs_roformer_sw` | bass, drums, other, vocals, guitar, piano | BS-RoFormer |
| `melband_roformer_kim` | vocals, other¹ | Mel-Band RoFormer |
| `scnet_small` | drums, bass, other, vocals | SCNet Masked Small |
| `scnet_xl_wide_v5` | drums, bass, other, vocals | SCNet XL IHF |

¹ Computed client-side as `mixture - vocals`.

The RoFormer models are markedly higher quality but larger (~350–480 MB fp16) and slower. Weight licenses vary — surface `separator.license` in your app where appropriate.

Artifacts are hosted on Hugging Face under immutable revisions; sizes and SHA-256 digests live in `src/model-artifacts.ts` and can be verified with `npm run verify:model-artifacts`.

## Constants

`SAMPLE_RATE`, `SEGMENT_OVERLAP`, `MODEL_CONFIGS` (per-model DSP geometry), `specDims(config)` — plus HTDemucs-specific `NFFT`, `HOP_LENGTH`, `SEGMENT_SAMPLES`, `SEGMENT_SECONDS`. Prefer `MODEL_CONFIGS[model]`.
