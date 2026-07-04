# demucs-next

Browser-side audio source separation using ONNX [Demucs](https://github.com/adefossez/demucs) models. Runs HTDemucs entirely in the browser (WebGPU when available, WASM otherwise), spreading the STFT, ONNX inference, and iSTFT across three Web Workers.

For backend/server-side workflows, use the Python `demucs` package — it is significantly faster than the in-browser ONNX path.

## Install

```bash
npm install demucs-next
```

`onnxruntime-web` is a regular dependency and is bundled for you; there is no separate peer install and no `<script>` tag. The package ships compiled ES modules plus type declarations from `./dist`. The three workers are referenced via `new Worker(new URL('./workers/*.js', import.meta.url))`, so you need a bundler that understands that pattern (Vite, Webpack 5). ORT's `.wasm` assets are emitted into your bundle by default; pass `wasmPaths` to load them from a URL at runtime instead.

### Vite consumers

Add `demucs-next` to `optimizeDeps.exclude` so Vite processes the workers (and resolves ORT) instead of pre-bundling them with esbuild, which mangles the worker URLs:

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ['demucs-next'] },
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

## Constants

- `SAMPLE_RATE` — `44100`. The only valid input sample rate.
- `SEGMENT_SAMPLES` — `343980`. ~7.8s at 44.1 kHz; the training segment length the ONNX graph is traced at.
- `SEGMENT_SECONDS` — `SEGMENT_SAMPLES / SAMPLE_RATE` (~7.8s).
- `SEGMENT_OVERLAP` — `0.25`. Overlap fraction between consecutive segments.
- `NFFT` — `4096`. STFT FFT size.
- `HOP_LENGTH` — `1024`. STFT hop length.

## Usage

```ts
import { Separator } from 'demucs-next';

const separator = await Separator.load('htdemucs', {
  backend: 'webgpu',   // falls back to 'wasm' automatically
  precision: 'fp32',   // 'fp16' = smaller download, near-identical output (not bit-exact)
});

// audioBuffer: a 44.1kHz Web Audio AudioBuffer (1 or 2 channels)
const result = await separator.separate(audioBuffer, {
  onProgress: (p) => console.log(p),
});

console.log(result.stems); // stem name → interleaved L/R Float32Array

await separator.unload();
```

### `Separator.load(model, options)`

Loads a model and returns a ready-to-use `Separator`. The model is fetched from HuggingFace (`Ryan5453/demucs-onnx`) on first load and cached by the browser.

- `model`: `'htdemucs'` (4 stems: drums, bass, other, vocals) | `'htdemucs_6s'` (6 stems: drums, bass, guitar, piano, other, vocals)
- `options.backend`: `'webgpu'` (default) | `'wasm'`. WebGPU falls back to WASM automatically if unavailable or if session creation fails.
- `options.precision`: `'fp32'` (default) | `'fp16'`. `'fp16'` is a weight-only-fp16 variant — roughly half the download with near-identical (not bit-exact) output: the weights are rounded to fp16, but compute still runs in fp32, so the difference is well below the audible floor.
- `options.wasmPaths`: override the ORT `.wasm` asset URL prefix
- `options.numThreads`: WASM thread count (default 4)

Each `Separator` instance owns its own three workers (STFT, ONNX, iSTFT). Multiple instances can run concurrently — call `load()` more than once to run different models in parallel. (A single instance is not safe to call `separate` on concurrently; calls on the same instance should be sequential.)

### Instance members

- `separator.model` — the loaded `ModelType`.
- `separator.sources` — stem names produced by the model.
- `separator.backend` — `'webgpu'` | `'wasm'` actually in use after fallback.
- `separator.precision` — `'fp32'` | `'fp16'`.
- `separator.separate(audioBuffer, options?)` — separates one `AudioBuffer`; safe to call repeatedly.
- `separator.unload()` — releases model resources and tears down all three workers. The instance cannot be used afterward.

### `separate` options and result

```ts
interface SeparationOptions {
    onProgress?: (p: SeparationProgress) => void;
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

The pipeline processes the audio in overlapping ~7.8s segments with crossfaded boundaries, pipelined across the STFT, ONNX, and iSTFT workers so STFT for segment N+1 runs while segment N is in inference.

## Decoding Audio

`demucs-next` does not handle audio decoding — you bring the `AudioBuffer`. In the browser, the easiest route is `AudioContext.decodeAudioData`, which handles MP3, AAC, FLAC, WAV, and Ogg using the browser's built-in decoders:

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
- **Memory.** Each loaded model occupies ~300 MB of GPU/heap memory. Unload instances you no longer need.
