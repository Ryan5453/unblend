import type { ModelType } from './constants';
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
     * size of the fp32 model with bit-equivalent output. Defaults to
     * ``'fp32'``.
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
        const hasWebGPU = await isWebGPUAvailable();
        let backend: 'webgpu' | 'wasm' =
            preferredBackend === 'webgpu' && hasWebGPU ? 'webgpu' : 'wasm';
        const precision: ModelPrecision = options.precision ?? 'fp32';
        const modelUrl = MODEL_URLS[model][precision];

        const onnx = new OnnxClient();
        const workerOptions = {
            wasmPaths: options.wasmPaths,
            numThreads: options.numThreads,
        };

        try {
            await onnx.load(modelUrl, backend, workerOptions);
        } catch (err) {
            // WebGPU can be available but fail at session create (driver bugs,
            // unsupported ops); transparently fall back to WASM.
            if (backend === 'webgpu') {
                backend = 'wasm';
                await onnx.load(modelUrl, backend, workerOptions);
            } else {
                onnx.terminate();
                throw err;
            }
        }

        const stft = new STFTClient();
        const istft = new ISTFTClient();
        return new Separator(
            model, MODEL_SOURCES[model], backend, precision, onnx, stft, istft
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
            await this.onnx.unload();
        } catch {
            // Worker may already be in a bad state; terminate regardless.
        }
        this.onnx.terminate();
        this.stft.terminate();
        this.istft.terminate();
    }
}
