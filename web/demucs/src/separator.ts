import {
    MODEL_CONFIGS,
    SAMPLE_RATE,
    dspConfig,
    type ModelConfig,
    type ModelType,
} from './constants.js';
import { MODEL_ARTIFACTS } from './model-artifacts.js';
import { OnnxClient, type LoadProgressCallback } from './onnx-client.js';
import { STFTClient } from './stft-client.js';
import { ISTFTClient } from './istft-client.js';
import {
    runPipeline,
    validateSeparationOptions,
    type SeparationOptions,
    type SeparationResult,
} from './pipeline.js';

export type ModelPrecision = 'fp32' | 'fp16';

export interface LoadModelOptions {
    /** Defaults to 'webgpu', falling back to 'wasm' if unavailable. */
    backend?: 'webgpu' | 'wasm';
    /**
     * Model weight precision. ``'fp16'`` is weight-only storage: selected
     * initializers are stored in fp16 and cast to fp32 before their consumers,
     * while compute, activations, and IO remain fp32.
     */
    precision?: ModelPrecision;
    /** Override ORT's .wasm asset URL prefix; defaults to bundler-resolved. */
    wasmPaths?: string;
    /** WASM thread count; defaults to 4. */
    numThreads?: number;
    /**
     * Fetch the ONNX weights from this URL instead of the registered
     * Hugging Face artifact. The model's config/precision still come from
     * ``model``; only the byte source changes. Intended for testing a locally
     * exported model before it is published.
     */
    modelUrl?: string;
    /**
     * Override ORT's graph optimization level; defaults to 'all'. Exposed for
     * diagnosing EP-specific optimizer bugs (a lower level skips fusion/
     * constant-folding passes that may behave differently per execution
     * provider).
     */
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
    /** Abort model probing/loading and terminate every worker already created. */
    signal?: AbortSignal;
    /** Reports model download bytes, then a single 'compile' call while ORT
     *  initializes the session (no progress signal exists for that step). */
    onProgress?: LoadProgressCallback;
}

function abortReason(signal: AbortSignal): unknown {
    return signal.reason ?? new DOMException('The operation was aborted', 'AbortError');
}

function throwIfAborted(signal?: AbortSignal): void {
    if (signal?.aborted) throw abortReason(signal);
}

function awaitWithSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
    if (!signal) return promise;
    if (signal.aborted) return Promise.reject(abortReason(signal));
    return new Promise<T>((resolve, reject) => {
        let settled = false;
        const finish = (callback: () => void) => {
            if (settled) return;
            settled = true;
            signal.removeEventListener('abort', onAbort);
            callback();
        };
        const onAbort = () => finish(() => reject(abortReason(signal)));
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(
            value => finish(() => resolve(value)),
            error => finish(() => reject(error)),
        );
    });
}

async function isWebGPUAvailable(): Promise<boolean> {
    if (!navigator.gpu) return false;
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
    /** License of the model weights. */
    readonly license: string;

    private readonly config: ModelConfig;
    private readonly onnx: OnnxClient;
    private readonly stft: STFTClient;
    private readonly istft: ISTFTClient;
    private disposed = false;
    private active = false;

    private constructor(
        model: ModelType,
        config: ModelConfig,
        backend: 'webgpu' | 'wasm',
        precision: ModelPrecision,
        onnx: OnnxClient,
        stft: STFTClient,
        istft: ISTFTClient,
    ) {
        this.model = model;
        this.config = config;
        this.sources = config.sources;
        this.license = config.license;
        this.backend = backend;
        this.precision = precision;
        this.onnx = onnx;
        this.stft = stft;
        this.istft = istft;
    }

    /** Load a model and return a ready-to-use Separator. */
    static async load(
        model: ModelType,
        options: LoadModelOptions = {}
    ): Promise<Separator> {
        // This check intentionally precedes model validation and worker creation.
        throwIfAborted(options.signal);

        const preferredBackend = options.backend ?? 'webgpu';
        if (preferredBackend !== 'webgpu' && preferredBackend !== 'wasm') {
            throw new Error(
                `Unknown backend '${preferredBackend}'. Valid backends: webgpu, wasm.`
            );
        }
        const precision: ModelPrecision = options.precision ?? 'fp32';
        const modelArtifacts = MODEL_ARTIFACTS[model];
        if (!modelArtifacts) {
            throw new Error(
                `Unknown model '${model}'. Valid models: ${Object.keys(MODEL_ARTIFACTS).join(', ')}.`
            );
        }
        const artifact = modelArtifacts[precision];
        if (!artifact) {
            throw new Error(
                `Unknown precision '${precision}'. Valid precisions: ${Object.keys(modelArtifacts).join(', ')}.`
            );
        }
        const modelUrl = options.modelUrl ?? artifact.url;

        let onnx: OnnxClient | null = null;
        let stft: STFTClient | null = null;
        let istft: ISTFTClient | null = null;
        const cleanup = (reason?: unknown) => {
            onnx?.terminate(reason);
            stft?.terminate(reason);
            istft?.terminate(reason);
        };
        const onAbort = () => cleanup(abortReason(options.signal!));
        options.signal?.addEventListener('abort', onAbort, { once: true });

        try {
            let backend: 'webgpu' | 'wasm' = preferredBackend === 'webgpu'
                && await awaitWithSignal(isWebGPUAvailable(), options.signal)
                ? 'webgpu'
                : 'wasm';
            throwIfAborted(options.signal);

            onnx = new OnnxClient();
            const workerOptions = {
                wasmPaths: options.wasmPaths,
                numThreads: options.numThreads,
                graphOptimizationLevel: options.graphOptimizationLevel,
            };
            try {
                await awaitWithSignal(
                    onnx.load(modelUrl, backend, workerOptions, options.onProgress),
                    options.signal
                );
            } catch (error) {
                // Never reinterpret an abort as a WebGPU failure/fallback.
                throwIfAborted(options.signal);
                if (backend !== 'webgpu') throw error;
                onnx.terminate(error);
                backend = 'wasm';
                onnx = new OnnxClient();
                await awaitWithSignal(
                    onnx.load(modelUrl, backend, workerOptions, options.onProgress),
                    options.signal
                );
            }
            throwIfAborted(options.signal);

            const config = MODEL_CONFIGS[model];
            stft = new STFTClient();
            istft = new ISTFTClient();
            await awaitWithSignal(Promise.all([
                stft.configure(dspConfig(config)),
                istft.configure(dspConfig(config)),
            ]), options.signal);
            throwIfAborted(options.signal);

            return new Separator(model, config, backend, precision, onnx, stft, istft);
        } catch (error) {
            cleanup(error);
            throw error;
        } finally {
            options.signal?.removeEventListener('abort', onAbort);
        }
    }

    /**
     * Separate one 44.1kHz AudioBuffer. Calls on one instance are sequential.
     * Abort or any worker-backed failure invalidates this instance permanently.
     */
    async separate(
        audioBuffer: AudioBuffer,
        options: SeparationOptions = {}
    ): Promise<SeparationResult> {
        if (this.disposed) throw new Error('Separator has been unloaded');
        if (this.active) throw new Error('Separation already in progress');
        if (audioBuffer.sampleRate !== SAMPLE_RATE) {
            throw new Error(
                `Separator expects ${SAMPLE_RATE} Hz audio, got ${audioBuffer.sampleRate} Hz. `
                + `Resample the AudioBuffer to ${SAMPLE_RATE} Hz before calling separate().`
            );
        }
        if (audioBuffer.length === 0) throw new Error('audioBuffer is empty (0 samples).');
        validateSeparationOptions(options);
        throwIfAborted(options.signal);

        this.active = true;
        const onAbort = () => this.hardInvalidate(abortReason(options.signal!));
        options.signal?.addEventListener('abort', onAbort, { once: true });
        try {
            const result = await runPipeline(
                { onnx: this.onnx, stft: this.stft, istft: this.istft },
                audioBuffer,
                this.config,
                options,
            );
            throwIfAborted(options.signal);
            return result;
        } catch (error) {
            this.hardInvalidate(options.signal?.aborted ? abortReason(options.signal) : error);
            throw options.signal?.aborted ? abortReason(options.signal) : error;
        } finally {
            options.signal?.removeEventListener('abort', onAbort);
            this.active = false;
        }
    }

    /** Release resources. Active work is cancelled destructively. */
    async unload(): Promise<void> {
        if (this.disposed) return;
        if (this.active) {
            this.hardInvalidate(new Error('Separator unloaded during active separation'));
            return;
        }

        this.disposed = true;
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
            await Promise.race([
                this.onnx.unload(),
                new Promise<void>(resolve => {
                    timer = setTimeout(resolve, 2000);
                }),
            ]);
        } catch {
            // The worker may already be unhealthy; termination below is final.
        } finally {
            if (timer !== undefined) clearTimeout(timer);
            this.terminateWorkers();
        }
    }

    private hardInvalidate(reason: unknown): void {
        if (this.disposed) return;
        this.disposed = true;
        this.terminateWorkers(reason);
    }

    private terminateWorkers(reason?: unknown): void {
        this.onnx.terminate(reason);
        this.stft.terminate(reason);
        this.istft.terminate(reason);
    }
}
