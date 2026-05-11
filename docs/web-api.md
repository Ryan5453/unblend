# Demucs Web API

`demucs-web` is the browser-side counterpart to the Python `demucs` package. It runs HTDemucs ONNX models entirely in the browser using `onnxruntime-web` (WebGPU when available, WASM otherwise) and is designed to be embedded in a web app.

For backend/server-side workflows, you should use the Python package as it is significantly faster than the in-browser ONNX path.

## Install

```bash
npm install demucs-web onnxruntime-web
```

`onnxruntime-web` is a peer dependency. The current implementation reads `ort` from `window.ort`, so you must also load it via a `<script>` tag in your HTML before the bundle runs:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/ort.all.min.js"></script>
```

The package ships TypeScript sources directly (no prebuilt `dist`); your bundler compiles them. Worker files are loaded via `new URL('./workers/...', import.meta.url)`, which Vite, webpack 5, and Rollup all resolve automatically.

### Cross-Origin Isolation

WASM multi-threading and `SharedArrayBuffer` require the page to be cross-origin isolated. Set these response headers on every request your app serves:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Input Requirements

- **Sample rate:** exactly 44.1 kHz. HTDemucs is trained at this rate; the STFT parameters and segment length are baked into the ONNX graph and cannot be changed. Resample with `OfflineAudioContext` before calling.
- **Channels:** 1 or 2. Mono is duplicated to fake-stereo internally. 3+ channel input is not supported — downmix to stereo first.
- **Format:** `Float32Array` interleaved L/R samples, length `numSamples * 2`. The current `separateAudioBuffer` entry point accepts an `AudioBuffer` and does the interleaving for you.

Output is always 2 channels per stem regardless of input channel count.

## Constants

- `SAMPLE_RATE` — `44100`. The only valid input sample rate.
- `SEGMENT_SAMPLES` — `343980`. ~7.8s at 44.1 kHz; the training segment length the ONNX graph is traced at.
- `SEGMENT_OVERLAP` — `0.25`. Overlap fraction between consecutive segments.
- `NFFT` — `4096`. STFT FFT size.
- `HOP_LENGTH` — `1024`. STFT hop length.

## Loading a Model

```ts
import { loadModel, unloadModel, type ModelType } from 'demucs-web';

const result = await loadModel(
    model: ModelType,
    addLog: (message: string, level: 'info' | 'success' | 'error') => void,
    preferredBackend?: 'webgpu' | 'wasm',
);
```

Parameters:

- `model` — `'htdemucs'` (4 stems: drums, bass, other, vocals) or `'htdemucs_6s'` (6 stems: drums, bass, guitar, piano, other, vocals).
- `addLog` — callback for human-readable status messages. Useful for surfacing load progress in your UI.
- `preferredBackend` — defaults to `'webgpu'`. Falls back to `'wasm'` automatically if WebGPU is unavailable or fails to initialize.

Returns a `ModelLoadResult`:

```ts
interface ModelLoadResult {
    success: boolean;
    sources: string[];   // e.g. ['drums', 'bass', 'other', 'vocals']
    backend?: 'webgpu' | 'wasm';
}
```

The model is fetched from HuggingFace (`Ryan5453/demucs-onnx`) on first load and cached by the browser. A typical model is ~80 MB.

To free GPU memory and tear down the worker:

```ts
await unloadModel();
```

### Backend Selection

- **WebGPU** runs on the main thread and uses GPU acceleration. Fastest path; available in Chrome, Edge, and recent Safari.
- **WASM** runs in a Web Worker (so it doesn't block the UI) using SIMD + multi-threading. Available everywhere modern browsers run, but ~2–3× slower than WebGPU.

`isWebGPUAvailable()` and `getBackend()` let you query availability and current state.

## Separating Audio

```ts
import { separateAudioBuffer } from 'demucs-web';

const result = await separateAudioBuffer(
    audioBuffer: AudioBuffer,
    options?: SeparationOptions,
);
```

Parameters:

- `audioBuffer` — a Web Audio `AudioBuffer` at 44.1 kHz with 1 or 2 channels.
- `options.onProgress` — optional callback fired after each segment completes:
  ```ts
  (progress: { segIdx: number; totalSegs: number; fraction: number }) => void
  ```

Returns a `SeparationResult`:

```ts
interface SeparationResult {
    stems: Record<string, Float32Array>;  // stem name → interleaved L/R samples
    wallMs: number;                       // total wall time including STFT/iSTFT
    inferenceMs: number;                  // sum of ONNX inference time across segments
    numSegments: number;                  // number of segments processed
}
```

Each stem `Float32Array` has length `numSamples * 2` and is interleaved: `[L0, R0, L1, R1, ...]`. To convert to a WAV blob for download or playback, encode it yourself or use the same approach as the demo app (`web/app/src/utils/wav-utils.ts`).

The pipeline processes the audio in overlapping ~7.8s segments with crossfaded boundaries. STFT, ONNX inference, and iSTFT run pipelined across two Web Workers and (when on WebGPU) the main thread, so STFT for segment N+1 happens while segment N is in inference.

### Cancelling

There is no `AbortSignal` in the current API. To stop a separation in progress, call `terminateSTFTWorker()` and `terminateISTFTWorker()` — both will cause any in-flight `separateAudioBuffer` call to reject. The next call automatically re-initializes them.

## Helpers

- `isWebGPUAvailable(): Promise<boolean>` — checks `navigator.gpu` and that an adapter can be obtained.
- `getBackend(): 'webgpu' | 'wasm' | null` — returns the backend currently in use, or `null` if no model is loaded.
- `isUsingWorker(): boolean` — `true` when inference runs in a Web Worker (always true for WASM, false for WebGPU).
- `getSources(): string[]` — the stem names produced by the currently loaded model.

## Decoding Audio

`demucs-web` does not handle audio decoding — you bring the `Float32Array`. In the browser, the easiest route is `AudioContext.decodeAudioData`, which handles MP3, AAC, FLAC, WAV, and Ogg using the browser's built-in decoders:

```ts
const ctx = new AudioContext({ sampleRate: 44100 });
const arrayBuffer = await file.arrayBuffer();
const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
const stems = await separateAudioBuffer(audioBuffer);
```

For broader format support (ALAC, WMA, exotic containers), use `mediabunny` or `ffmpeg.wasm` and pass the resulting samples in. See `web/app/src/utils/audio-decoder.ts` in the demo app for a two-tier fallback example.

## Limitations

- **Browser only.** WebGPU, Web Workers, and `onnxruntime-web` together are not portable to Node or Deno without significant adaptation.
- **Speed.** ONNX in the browser is ~3× slower than the Python package on equivalent hardware. A 4-minute song takes 30–90 seconds depending on backend and device.
- **Memory.** Each loaded model occupies ~300 MB of GPU/heap memory. Unload before loading a different model.
- **Single instance.** The current API uses module-level globals for the session and workers; you cannot run two separations concurrently in the same tab.
