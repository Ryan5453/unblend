interface LoadModelMessage {
    type: 'load';
    requestId: number;
    modelUrl: string;
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
    /** Absent for models without an audio input (RoFormer). */
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

type WorkerResponse = LoadResponse | RunResponse | UnloadResponse;
type OutgoingMessage =
    | Omit<LoadModelMessage, 'requestId'>
    | Omit<RunInferenceMessage, 'requestId'>
    | Omit<UnloadMessage, 'requestId'>;

export interface InferenceResult {
    outSpecReal: Float32Array;
    outSpecImag: Float32Array;
    /** Present only for models with a time-domain branch (HTDemucs). */
    outWave?: Float32Array;
    outSpecShape: number[];
    outWaveShape?: number[];
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
    private terminated = false;

    constructor() {
        this.worker = new Worker(
            new URL('./workers/onnx-worker.js', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
            if (this.terminated || event.data.requestId !== this.pendingId) return;
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
        options: {
            wasmPaths?: string;
            numThreads?: number;
            graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
        } = {}
    ): Promise<void> {
        const response = (await this.send({
            type: 'load',
            modelUrl,
            backend,
            wasmPaths: options.wasmPaths,
            numThreads: options.numThreads,
            graphOptimizationLevel: options.graphOptimizationLevel,
        })) as LoadResponse;

        if (!response.success) {
            throw new Error(response.error || 'Model load failed');
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
        // ``audio`` remains one of the pipeline's reusable double-buffers.
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
    }

    private rejectPending(reason: unknown): void {
        const reject = this.pendingReject;
        this.clearPending();
        reject?.(reason);
    }

    private send(
        message: OutgoingMessage,
        transfer: Transferable[] = []
    ): Promise<WorkerResponse> {
        if (this.terminated) {
            return Promise.reject(new Error('ONNX worker has been terminated'));
        }
        if (this.pendingReject !== null) {
            return Promise.reject(new Error('ONNX worker request already in progress'));
        }

        const requestId = ++this.requestCounter;
        this.pendingId = requestId;
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
