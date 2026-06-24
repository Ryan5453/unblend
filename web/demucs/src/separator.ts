import { SAMPLE_RATE, type ModelType } from './constants';
import { OnnxClient } from './onnx-client';
import { STFTClient } from './stft-client';
import { ISTFTClient } from './istft-client';
import {
    runPipeline,
    type SeparationOptions,
    type SeparationResult,
} from './pipeline';

export type ModelPrecision = 'fp32' | 'fp16';

const MODEL_URLS: Record<ModelType, Record<ModelPrecision, string>> = {
    'htdemucs': {
        fp32: 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs_fp32.onnx',
        fp16: 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs_fp16.onnx',
    },
    'htdemucs_6s': {
        fp32: 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs_6s_fp32.onnx',
        fp16: 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs_6s_fp16.onnx',
    },
};

// Must match the source order baked into the trained model.
const MODEL_SOURCES: Record<ModelType, string[]> = {
    'htdemucs': ['drums', 'bass', 'other', 'vocals'],
    'htdemucs_6s': ['drums', 'bass', 'guitar', 'piano', 'other', 'vocals'],
};

export interface LoadModelOptions {
    /** Defaults to 'webgpu', falling back to 'wasm' if unavailable. */
    backend?: 'webgpu' | 'wasm';
    /**
     * Model weight precision. ``'fp16'`` loads a weight-only-fp16 variant of
     * the model: every Conv/MatMul/Gemm/ConvTranspose weight is stored as
     * fp16 on disk, with a Cast(fp16->fp32) node inserted right after each
     * weight so compute still runs in full fp32 (ORT folds the constant Cast
     * at session-create, no runtime cost). The download is roughly half the
     * size of the fp32 model. Output is near-identical to fp32 (the weights
     * themselves are rounded to fp16, so it is not bit-exact, but compute still
     * runs in fp32 so the difference is well below the audible floor). Defaults
     * to ``'fp32'``.
     *
     * Note: this is NOT real fp16 compute. ORT-WASM doesn't accumulate fp16
     * GEMMs in fp32 the way CUDA/MPS do, so a true fp16 graph produced
     * audible 8-bit-style quantization noise. Weight-only fp16 sidesteps
     * that entirely. The label is kept as ``'fp16'`` because that's what
     * users will recognize from CUDA/MPS — it just means "smaller model" in
     * the browser context.
     */
    precision?: ModelPrecision;
    /** Override ORT's .wasm asset URL prefix; defaults to bundler-resolved. */
    wasmPaths?: string;
    /** WASM thread count; defaults to 4. */
    numThreads?: number;
}

async function isWebGPUAvailable(): Promise<boolean> {
    if (!navigator.gpu) {
        return false;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
    } catch {
        return false;
    }
}

export class Separator {
    readonly model: ModelType;
    readonly sources: string[];
    readonly backend: 'webgpu' | 'wasm';
    readonly precision: ModelPrecision;

    private readonly onnx: OnnxClient;
    private readonly stft: STFTClient;
    private readonly istft: ISTFTClient;
    private disposed = false;

    private constructor(
        model: ModelType,
        sources: string[],
        backend: 'webgpu' | 'wasm',
        precision: ModelPrecision,
        onnx: OnnxClient,
        stft: STFTClient,
        istft: ISTFTClient,
    ) {
        this.model = model;
        this.sources = sources;
        this.backend = backend;
        this.precision = precision;
        this.onnx = onnx;
        this.stft = stft;
        this.istft = istft;
    }

    /**
     * Load a model and return a ready-to-use Separator. Each instance owns
     * its own three workers (STFT, iSTFT, ONNX); call load() more than once
     * to run multiple models in parallel.
     */
    static async load(
        model: ModelType,
        options: LoadModelOptions = {}
    ): Promise<Separator> {
        const preferredBackend = options.backend ?? 'webgpu';
        // Only probe WebGPU when it could actually be used; skip the adapter
        // request entirely if the caller explicitly asked for 'wasm'.
        let backend: 'webgpu' | 'wasm' =
            preferredBackend === 'webgpu' && (await isWebGPUAvailable())
                ? 'webgpu'
                : 'wasm';
        const precision: ModelPrecision = options.precision ?? 'fp32';
        const modelUrls = MODEL_URLS[model];
        if (!modelUrls) {
            throw new Error(
                `Unknown model '${model}'. Valid models: ` +
                `${Object.keys(MODEL_URLS).join(', ')}.`
            );
        }
        const modelUrl = modelUrls[precision];
        if (!modelUrl) {
            throw new Error(
                `Unknown precision '${precision}'. Valid precisions: ` +
                `${Object.keys(modelUrls).join(', ')}.`
            );
        }

        let onnx = new OnnxClient();
        const workerOptions = {
            wasmPaths: options.wasmPaths,
            numThreads: options.numThreads,
        };

        try {
            await onnx.load(modelUrl, backend, workerOptions);
        } catch (err) {
            // WebGPU can be available but fail at session create (driver bugs,
            // unsupported ops); transparently fall back to WASM. The WebGPU
            // worker may be left in an indeterminate state, so tear it down and
            // retry on a fresh worker rather than reusing it.
            if (backend === 'webgpu') {
                onnx.terminate();
                backend = 'wasm';
                onnx = new OnnxClient();
                try {
                    await onnx.load(modelUrl, backend, workerOptions);
                } catch (wasmErr) {
                    onnx.terminate();
                    throw wasmErr;
                }
            } else {
                onnx.terminate();
                throw err;
            }
        }

        // The STFT/iSTFT clients spin up their own workers in their
        // constructors, and ``new Worker(...)`` can throw synchronously (e.g.
        // CSP blocks worker creation). If that happens, the onnx worker is
        // already loaded — tear it down (and any stft client we did manage to
        // create) before rethrowing so we don't leak a live worker.
        let stft: STFTClient | null = null;
        let istft: ISTFTClient | null = null;
        try {
            stft = new STFTClient();
            istft = new ISTFTClient();
        } catch (err) {
            onnx.terminate();
            stft?.terminate();
            throw err;
        }
        // Both are non-null here: the try block either assigns both or rethrows.
        return new Separator(
            model, MODEL_SOURCES[model], backend, precision, onnx, stft!, istft!
        );
    }

    /**
     * Separate ``audioBuffer`` (44.1kHz, 1 or 2 channels) into stems. Safe to
     * call repeatedly; not safe to call concurrently on the same instance.
     */
    async separate(
        audioBuffer: AudioBuffer,
        options: SeparationOptions = {}
    ): Promise<SeparationResult> {
        if (this.disposed) {
            throw new Error('Separator has been unloaded');
        }
        // The ONNX graph is traced at 44.1kHz; feeding another rate would
        // silently produce pitch/tempo-wrong stems rather than erroring.
        if (audioBuffer.sampleRate !== SAMPLE_RATE) {
            throw new Error(
                `Separator expects ${SAMPLE_RATE} Hz audio, got ` +
                `${audioBuffer.sampleRate} Hz. Resample the AudioBuffer to ` +
                `${SAMPLE_RATE} Hz before calling separate().`
            );
        }
        if (audioBuffer.length === 0) {
            throw new Error('audioBuffer is empty (0 samples).');
        }
        return runPipeline(
            { onnx: this.onnx, stft: this.stft, istft: this.istft },
            audioBuffer,
            this.sources,
            options
        );
    }

    /**
     * Release model resources and tear down all three workers. The instance
     * cannot be used after this returns.
     */
    async unload(): Promise<void> {
        if (this.disposed) return;
        this.disposed = true;

        try {
            // Bound the graceful release so a wedged worker can't hang unload()
            // forever; terminate() below frees it either way.
            await Promise.race([
                this.onnx.unload(),
                new Promise<void>((resolve) => setTimeout(resolve, 2000)),
            ]);
        } catch {
            // Worker may already be in a bad state; terminate regardless.
        }
        this.onnx.terminate();
        this.stft.terminate();
        this.istft.terminate();
    }
}
