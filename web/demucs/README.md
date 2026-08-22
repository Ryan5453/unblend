# unblend

Browser-side audio source separation using ONNX models — HTDemucs (from [Demucs](https://github.com/adefossez/demucs)) and the RoFormer family (BS-RoFormer / Mel-Band RoFormer community checkpoints). Runs entirely in the browser, spreading the STFT, ONNX inference, and iSTFT across three Web Workers. WebGPU is preferred; HTDemucs and Mel-Band can fall back to WASM, while the six-stem BS-RoFormer requires WebGPU.

For backend/server-side workflows, use the `unblend` Python package — it is significantly faster than the in-browser ONNX path.

## Install

```bash
npm install unblend
```

`onnxruntime-web` is a regular dependency and is bundled for you; there is no separate peer install and no `<script>` tag. The package ships compiled ES modules plus type declarations from `./dist`. The three workers are referenced via `new Worker(new URL('./workers/*.js', import.meta.url))`, so you need a bundler that understands that pattern (Vite, Webpack 5). ORT's `.wasm` assets are emitted into your bundle by default; pass `wasmPaths` to load them from a URL at runtime instead.

### Vite consumers

Add `unblend` to `optimizeDeps.exclude` so Vite processes the workers (and resolves ORT) instead of pre-bundling them with esbuild, which mangles the worker URLs:

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ['unblend'] },
});
```

If you target a host with a per-file size cap (e.g. Cloudflare Pages' 25MB limit), keep ORT's `.wasm` out of your bundle and serve it from a CDN via `wasmPaths`; a small `generateBundle` plugin can strip any emitted `ort-*.wasm`/`ort-*.mjs`.

### Cross-Origin Isolation

WASM multi-threading and `SharedArrayBuffer` require the page to be cross-origin isolated. Set these response headers on every request your app serves:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Input Requirements

- **Sample rate:** exactly 44.1 kHz. The STFT parameters and segment length are baked into the ONNX graph and cannot be changed. Resample with `OfflineAudioContext` before calling.
- **Channels:** 1 or 2. Mono is duplicated to fake-stereo internally. 3+ channel input is silently truncated to the first two channels (mirroring the Python `convert_audio_channels` contract); downmix yourself if you need a different stereo image.
- `separate` takes a Web Audio `AudioBuffer`; channel interleaving is handled for you.

Output is always 2 channels per stem regardless of input channel count.

## Models

| Model | Stems | Family | Weights license |
|---|---|---|---|
| `htdemucs` | drums, bass, other, vocals | HTDemucs | unlicensed¹ |
| `htdemucs_6s` | + guitar, piano | HTDemucs | unlicensed¹ |
| `bs_roformer_sw` | bass, drums, other, vocals, guitar, piano | BS-RoFormer | unlicensed² |
| `melband_roformer_kim` | vocals, other³ | Mel-Band RoFormer | MIT |

¹ The HTDemucs *code* is MIT (Meta), but the released weights carry no license grant.
² The surviving BS-RoFormer-SW checkpoint has no license grant. Surface `separator.license` in your app where appropriate.
³ `other` is computed client-side as `mixture - vocals` (the checkpoint has a single vocal mask head).

The RoFormer models are markedly higher quality (SDR ~11-14 dB vs ~8-9 dB for HTDemucs on vocals) but larger (~350-480 MB fp16) and slower per segment. Their browser artifacts use exact query-chunked, head-grouped attention, feature-grouped feed-forward layers, narrow per-band slices, and binary joins. This preserves the checkpoint math while avoiding multi-gigabyte attention/QKV allocations, 200+ MB MLP activations, and WebGPU's per-stage storage-buffer binding limit.

> All eight FP32/FP16 ONNX artifacts are hosted publicly under immutable Hugging Face revisions. HTDemucs fp16 artifacts use half-precision weight storage with fp32 compute; RoFormer fp16 artifacts use the browser-oriented mixed-precision layout described below. Exact byte sizes and SHA-256 digests are checked into `model-artifacts.ts`; maintainers can stream-verify every remote artifact with `npm run verify:model-artifacts` from this package directory.

## Constants

- `SAMPLE_RATE` — `44100`. The only valid input sample rate.
- `MODEL_CONFIGS` — per-model DSP geometry and metadata (`nfft`, `hopLength`, `segmentSamples`, `sources`, `license`, …). Use `specDims(config)` for a model's spectrogram dims.
- `SEGMENT_SAMPLES` / `SEGMENT_SECONDS` / `NFFT` / `HOP_LENGTH` — the HTDemucs values (`343980` / ~7.8s / `4096` / `1024`), kept for back-compat; prefer `MODEL_CONFIGS[model]`.
- `SEGMENT_OVERLAP` — `0.25`. Overlap fraction between consecutive segments.

## Usage

```ts
import { Separator } from 'unblend';

const controller = new AbortController();
const separator = await Separator.load('htdemucs', {
  backend: 'webgpu',   // falls back when the selected model fits WASM
  precision: 'fp32',   // 'fp16' = smaller download, near-identical output (not bit-exact)
  signal: controller.signal,
});

// audioBuffer: a 44.1kHz Web Audio AudioBuffer (1 or 2 channels)
const result = await separator.separate(audioBuffer, {
  signal: controller.signal,
  onProgress: (p) => console.log(p),
});

console.log(result.stems); // stem name → interleaved L/R Float32Array

await separator.unload();
```

### `Separator.load(model, options)`

Loads a model and returns a ready-to-use `Separator`. Model URLs are resolved from the package's registry when loaded.

- `model`: `'htdemucs'` | `'htdemucs_6s'` | `'bs_roformer_sw'` | `'melband_roformer_kim'` (see the Models table)
- `options.backend`: `'webgpu'` (default) | `'wasm'`. HTDemucs and Mel-Band fall back to WASM automatically if WebGPU is unavailable or session creation fails. BS-RoFormer requires WebGPU: its six-stem CPU working set exceeds ONNX Runtime Web's fixed WASM heap, so the library rejects that combination before risking an allocation crash or Safari tab refresh.
- `options.precision`: `'fp32'` (default) | `'fp16'`. HTDemucs fp16 stores rounded weights in half precision and computes in fp32. RoFormer fp16 uses mixed precision for weights and its largest activations, retaining fp32 for model IO, normalization, rotary trig, and softmax. Both are roughly half the download and near-identical, but not bit-exact.
- `options.wasmPaths`: override the ORT `.wasm` asset URL prefix
- `options.numThreads`: WASM thread count (default 4)
- `options.signal`: optional `AbortSignal`. A pre-aborted signal creates no workers; abort during loading terminates every worker already created and rejects with the signal reason. An aborted WebGPU load never falls through into a WASM retry.

Each `Separator` instance owns its own three workers (STFT, ONNX, iSTFT). Multiple instances can run concurrently — call `load()` more than once to run different models in parallel. A single instance rejects concurrent `separate()` calls.

Aborting an active separation, unloading during it, or encountering a worker/pipeline failure hard-terminates all three workers and permanently invalidates that `Separator`; load a fresh instance before retrying. This is deliberate because an in-flight ORT `session.run()` cannot be safely cancelled and then reused. Idle `unload()` first attempts a bounded graceful ONNX release and then terminates all workers.

### Instance members

- `separator.model` — the loaded `ModelType`.
- `separator.sources` — stem names produced by the model.
- `separator.license` — license of the model weights (`'unlicensed'` for HTDemucs and BS-RoFormer-SW; `'MIT'` for Mel-Band RoFormer Kim).
- `separator.backend` — `'webgpu'` | `'wasm'` actually in use after fallback.
- `separator.precision` — `'fp32'` | `'fp16'`.
- `separator.separate(audioBuffer, options?)` — separates one `AudioBuffer`; successful calls may be repeated sequentially. `options.signal` cancels destructively as described above.
- `separator.unload()` — releases model resources and tears down all three workers. The instance cannot be used afterward.

### `separate` options and result

```ts
interface SeparationOptions {
    onProgress?: (p: SeparationProgress) => void;
    signal?: AbortSignal; // abort invalidates this Separator; load a new one
    shifts?: number;    // random sub-second shifts to average, 1-20 (default 1);
                        // each extra shift reruns the separation, so runtime scales linearly
    seed?: number;      // optional integer seed for the shift-offset PRNG. With a fixed
                        // seed the offsets — and outputs — are deterministic. Defaults to
                        // non-deterministic (Math.random()). Reduced mod 2^32; independent
                        // of Python's RNG so same-seed parity is within JS only.
}

interface SeparationProgress {
    segIdx: number;     // 1-based index of the segment that just finished (cumulative across shifts)
    totalSegs: number;
    fraction: number;   // segIdx / totalSegs ∈ (0, 1]
}

interface SeparationResult {
    stems: Record<string, Float32Array>;  // stem name → interleaved L/R samples
    wallMs: number;       // total wall time including STFT/iSTFT
    inferenceMs: number;  // sum of ONNX inference time across segments
    numSegments: number;  // summed across shift rounds
}
```

Each stem `Float32Array` has length `numSamples * 2` and is interleaved: `[L0, R0, L1, R1, ...]`. To produce a WAV blob for download or playback, encode it yourself — see `web/app/src/utils/wav-utils.ts` in the demo app.

The pipeline processes the audio in overlapping segments (~7.8s for HTDemucs; the RoFormer models use their own traced chunk lengths, e.g. ~13.4s for `bs_roformer_sw`) with crossfaded boundaries, pipelined across the STFT, ONNX, and iSTFT workers so STFT for segment N+1 runs while segment N is in inference.

## Decoding Audio

`unblend` does not handle audio decoding — you bring the `AudioBuffer`. In the browser, the easiest route is `AudioContext.decodeAudioData`, which handles MP3, AAC, FLAC, WAV, and Ogg using the browser's built-in decoders:

```ts
const ctx = new AudioContext({ sampleRate: 44100 });
const arrayBuffer = await file.arrayBuffer();
const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
const result = await separator.separate(audioBuffer);
```

For broader format support (ALAC, WMA, exotic containers), use `mediabunny` or `ffmpeg.wasm`. See `web/app/src/utils/audio-decoder.ts` in the demo app for a two-tier fallback example.

## Limitations

- **Browser only.** WebGPU, Web Workers, and `onnxruntime-web` together are not portable to Node or Deno without significant adaptation.
- **Speed.** ONNX in the browser is ~3× slower than the Python package on equivalent hardware. A 4-minute song takes 30–90 seconds depending on backend and device.
- **Memory.** Model weights and inference workspaces are large, and returned stems require one full-track Float32 buffer per source. The overlap-add stage uses segment-sized circular buffers rather than a second full-track copy, but long tracks and six-stem models can still require substantial memory. Unload instances you no longer need.

## RoFormer browser memory

RoFormer attention remains all-to-all, but the ONNX exporter evaluates bounded query slices within independent head groups, immediately projects each group back to the model dimension, and sums those partial projections. Each query still attends to the complete key/value sequence, so this is not windowed or approximate attention. Feed-forward expansion is likewise evaluated in feature groups and summed after the second projection. Band splitting, mask-head selection, and GLU export as explicit slices with binary joins instead of wide multi-output `Split` or `Concat` dispatches. Browser deployments should use the shipped static-batch artifacts; dynamic-batch exports are intended for native/server runtimes.

The RoFormer fp16 artifacts additionally keep their largest projections, MLP activations, masks, and trained weights in half precision. Numerically sensitive RMS reductions, rotary trigonometry, softmax, and the public model IO remain fp32. This materially reduces Safari/WebGPU peak memory as well as download size without changing the model architecture.

Mel-Band RoFormer also completes through the WASM fallback (substantially more slowly than WebGPU). BS-RoFormer's six full spectrogram outputs push native CPU inference to roughly a 4.25 GB peak even after chunking, beyond the fixed ORT-WASM heap, so it deliberately requires WebGPU rather than attempting a known `std::bad_alloc` path.
