import type { ModelType } from './constants';
import { OnnxClient } from './onnx-client';
import { STFTClient } from './stft-client';
import { ISTFTClient } from './istft-client';
import {
    runPipeline,
    type SeparationOptions,
    type SeparationResult,
} from './pipeline';

const MODEL_URLS: Record<ModelType, string> = {
    'htdemucs': 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs.onnx',
    'htdemucs_6s': 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs_6s.onnx',
};

// Must match the source order baked into the trained model.
const MODEL_SOURCES: Record<ModelType, string[]> = {
    'htdemucs': ['drums', 'bass', 'other', 'vocals'],
    'htdemucs_6s': ['drums', 'bass', 'guitar', 'piano', 'other', 'vocals'],
};

export interface LoadModelOptions {
    /** Defaults to 'webgpu', falling back to 'wasm' if unavailable. */
    backend?: 'webgpu' | 'wasm';
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

    private readonly onnx: OnnxClient;
    private readonly stft: STFTClient;
    private readonly istft: ISTFTClient;
    private disposed = false;

    private constructor(
        model: ModelType,
        sources: string[],
        backend: 'webgpu' | 'wasm',
        onnx: OnnxClient,
        stft: STFTClient,
        istft: ISTFTClient,
    ) {
        this.model = model;
        this.sources = sources;
        this.backend = backend;
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

        const onnx = new OnnxClient();
        const workerOptions = {
            wasmPaths: options.wasmPaths,
            numThreads: options.numThreads,
        };

        try {
            await onnx.load(MODEL_URLS[model], backend, workerOptions);
        } catch (err) {
            // WebGPU can be available but fail at session create (driver bugs,
            // unsupported ops); transparently fall back to WASM.
            if (backend === 'webgpu') {
                backend = 'wasm';
                await onnx.load(MODEL_URLS[model], backend, workerOptions);
            } else {
                onnx.terminate();
                throw err;
            }
        }

        const stft = new STFTClient();
        const istft = new ISTFTClient();
        return new Separator(model, MODEL_SOURCES[model], backend, onnx, stft, istft);
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
