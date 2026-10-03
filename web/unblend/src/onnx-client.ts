import type { ModelType } from './constants.js';
import type { ModelByteSource, ModelCacheEntry } from './model-fetch.js';

interface LoadModelMessage {
    type: 'load';
    requestId: number;
    modelUrl: string;
    model?: ModelType;
    cache?: ModelCacheEntry;
    expectedBytes?: number;
    modelBytes?: Uint8Array<ArrayBuffer>;
    returnBytesOnFailure?: boolean;
    backend: 'webgpu' | 'wasm';
    wasmPaths?: string;
    numThreads?: number;
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
}

interface RunInferenceMessage {
    type: 'run';
    requestId: number;
    specReal: Float32Array;
    specImag: Float32Array;
    /** Absent for models without an audio input (RoFormer and SCNet). */
    audio?: Float32Array;
    specShape: number[];
    audioShape?: number[];
}

interface UnloadMessage {
    type: 'unload';
    requestId: number;
}

interface LoadResponse {
    type: 'load';
    requestId: number;
    success: boolean;
    backend?: 'webgpu' | 'wasm';
    error?: string;
    stage?: 'fetch' | 'session';
    modelBytes?: Uint8Array<ArrayBuffer>;
}

interface RunResponse {
    type: 'run';
    requestId: number;
    success: boolean;
    outSpecReal?: Float32Array;
    outSpecImag?: Float32Array;
    outWave?: Float32Array;
    outSpecShape?: number[];
    outWaveShape?: number[];
    error?: string;
}

interface UnloadResponse {
    type: 'unload';
    requestId: number;
    success: boolean;
}

interface ProgressMessage {
    type: 'progress';
    requestId: number;
    phase: 'download' | 'compile';
    loaded: number;
    total: number;
    source?: ModelByteSource;
}

type WorkerResponse = LoadResponse | RunResponse | UnloadResponse;
type IncomingMessage = WorkerResponse | ProgressMessage;
type OutgoingMessage =
    | Omit<LoadModelMessage, 'requestId'>
    | Omit<RunInferenceMessage, 'requestId'>
    | Omit<UnloadMessage, 'requestId'>;

/** Reported during `load()`: real byte progress while downloading, then a
 *  single 'compile' call once ORT starts parsing/initializing the session
 *  (which has no progress signal of its own). During 'download', ``source``
 *  is ``'cache'`` when the bytes are read from Cache Storage instead of the
 *  network; it is undefined for 'compile'. */
export type LoadProgressCallback = (
    phase: 'download' | 'compile',
    loaded: number,
    total: number,
    source?: ModelByteSource,
) => void;

export interface InferenceResult {
    outSpecReal: Float32Array;
    outSpecImag: Float32Array;
    /** Present only for models with a time-domain branch (HTDemucs). */
    outWave?: Float32Array;
    outSpecShape: number[];
    outWaveShape?: number[];
}

/**
 * A failed ``OnnxClient.load``. ``stage`` separates download failures
 * (network, HTTP status, truncation), which another backend cannot fix, from
 * session-creation failures, which carry the fetched bytes for a retry.
 */
export class ModelLoadError extends Error {
    readonly stage: 'fetch' | 'session';
    readonly modelBytes?: Uint8Array<ArrayBuffer>;

    constructor(
        message: string,
        stage: 'fetch' | 'session',
        modelBytes?: Uint8Array<ArrayBuffer>,
    ) {
        super(message);
        this.name = 'ModelLoadError';
        this.stage = stage;
        this.modelBytes = modelBytes;
    }
}

export interface OnnxLoadOptions {
    wasmPaths?: string;
    numThreads?: number;
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
    /**
     * Reject (at the fetch stage) a file whose embedded metadata contradicts
     * this model's built-in STFT/chunk geometry.
     */
    model?: ModelType;
    /** Serve/fill this Cache Storage entry (see ``loadModelBytes``). */
    cache?: ModelCacheEntry;
    /** Reject a download of any other byte length (see ``loadModelBytes``). */
    expectedBytes?: number;
    /** Bytes to load instead of fetching; transferred to the worker. */
    modelBytes?: Uint8Array<ArrayBuffer>;
    /** Attach the fetched bytes to a session-stage ``ModelLoadError``. */
    returnBytesOnFailure?: boolean;
}

function asError(reason: unknown, fallback: string): Error {
    if (reason instanceof Error) return reason;
    return new Error(reason === undefined ? fallback : String(reason));
}

export class OnnxClient {
    private worker: Worker;
    private pendingResolve: ((value: WorkerResponse) => void) | null = null;
    private pendingReject: ((reason?: unknown) => void) | null = null;
    private requestCounter = 0;
    private pendingId = -1;
    private pendingProgress: ((msg: ProgressMessage) => void) | null = null;
    private terminated = false;

    constructor() {
        this.worker = new Worker(
            new URL('./workers/onnx-worker.js', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (event: MessageEvent<IncomingMessage>) => {
            if (this.terminated || event.data.requestId !== this.pendingId) return;
            if (event.data.type === 'progress') {
                this.pendingProgress?.(event.data);
                return;
            }
            const resolve = this.pendingResolve;
            this.clearPending();
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[onnx-worker] error:', error);
            this.terminate(error.message || 'ONNX worker failed');
        };

        this.worker.onmessageerror = (event) => {
            console.error('[onnx-worker] message error:', event);
            this.terminate('ONNX worker message deserialization failed');
        };
    }

    async load(
        modelUrl: string,
        backend: 'webgpu' | 'wasm',
        options: OnnxLoadOptions = {},
        onProgress?: LoadProgressCallback
    ): Promise<void> {
        const response = (await this.send(
            {
                type: 'load',
                modelUrl,
                model: options.model,
                cache: options.cache,
                expectedBytes: options.expectedBytes,
                modelBytes: options.modelBytes,
                returnBytesOnFailure: options.returnBytesOnFailure,
                backend,
                wasmPaths: options.wasmPaths,
                numThreads: options.numThreads,
                graphOptimizationLevel: options.graphOptimizationLevel,
            },
            options.modelBytes ? [options.modelBytes.buffer] : [],
            onProgress && (msg => onProgress(msg.phase, msg.loaded, msg.total, msg.source))
        )) as LoadResponse;

        if (!response.success) {
            throw new ModelLoadError(
                response.error || 'Model load failed',
                response.stage ?? 'session',
                response.modelBytes,
            );
        }
    }

    async runInference(
        specReal: Float32Array,
        specImag: Float32Array,
        specShape: number[],
        audio?: Float32Array,
        audioShape?: number[]
    ): Promise<InferenceResult> {
        // The spectrogram buffers are no longer read by the pipeline, so
        // transfer ownership instead of cloning their multi-megabyte payloads.
        // ``audio`` is the pipeline's reusable planar buffer, so it is cloned
        // (synchronously, by postMessage) rather than transferred.
        const response = (await this.send({
            type: 'run',
            specReal,
            specImag,
            audio,
            specShape,
            audioShape,
        }, [specReal.buffer, specImag.buffer])) as RunResponse;

        if (!response.success) {
            throw new Error(response.error || 'Inference failed');
        }

        return {
            outSpecReal: response.outSpecReal!,
            outSpecImag: response.outSpecImag!,
            outWave: response.outWave,
            outSpecShape: response.outSpecShape!,
            outWaveShape: response.outWaveShape,
        };
    }

    async unload(): Promise<void> {
        await this.send({ type: 'unload' });
    }

    terminate(reason?: unknown): void {
        if (this.terminated) return;
        this.terminated = true;
        this.rejectPending(asError(reason, 'ONNX worker terminated'));
        this.worker.onmessage = null;
        this.worker.onerror = null;
        this.worker.onmessageerror = null;
        this.worker.terminate();
    }

    private clearPending(): void {
        this.pendingResolve = null;
        this.pendingReject = null;
        this.pendingId = -1;
        this.pendingProgress = null;
    }

    private rejectPending(reason: unknown): void {
        const reject = this.pendingReject;
        this.clearPending();
        reject?.(reason);
    }

    private send(
        message: OutgoingMessage,
        transfer: Transferable[] = [],
        onProgress?: (msg: ProgressMessage) => void
    ): Promise<WorkerResponse> {
        if (this.terminated) {
            return Promise.reject(new Error('ONNX worker has been terminated'));
        }
        if (this.pendingReject !== null) {
            return Promise.reject(new Error('ONNX worker request already in progress'));
        }

        const requestId = ++this.requestCounter;
        this.pendingId = requestId;
        this.pendingProgress = onProgress ?? null;
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            try {
                this.worker.postMessage({ ...message, requestId }, transfer);
            } catch (error) {
                if (this.pendingId === requestId) this.clearPending();
                reject(error);
            }
        });
    }
}
