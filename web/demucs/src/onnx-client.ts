interface LoadModelMessage {
    type: 'load';
    requestId: number;
    modelUrl: string;
    backend: 'webgpu' | 'wasm';
    wasmPaths?: string;
    numThreads?: number;
}

interface RunInferenceMessage {
    type: 'run';
    requestId: number;
    specReal: Float32Array;
    specImag: Float32Array;
    audio: Float32Array;
    specShape: number[];
    audioShape: number[];
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
// The caller supplies everything but the requestId; send() assigns it.
type OutgoingMessage =
    | Omit<LoadModelMessage, 'requestId'>
    | Omit<RunInferenceMessage, 'requestId'>
    | Omit<UnloadMessage, 'requestId'>;

export interface InferenceResult {
    outSpecReal: Float32Array;
    outSpecImag: Float32Array;
    outWave: Float32Array;
    outSpecShape: number[];
    outWaveShape: number[];
}

export class OnnxClient {
    private worker: Worker;
    private pendingResolve: ((value: WorkerResponse) => void) | null = null;
    private pendingReject: ((reason?: unknown) => void) | null = null;
    private requestCounter = 0;
    private pendingId = -1;

    constructor() {
        this.worker = new Worker(
            new URL('./workers/onnx-worker.js', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
            // Ignore late replies from a superseded request: a reused worker may
            // still post the result of an aborted run after the next request has
            // installed a new pending promise. Matching on requestId prevents
            // resolving the new request with the old run's stale payload.
            if (event.data.requestId !== this.pendingId) return;
            const resolve = this.pendingResolve;
            this.pendingResolve = null;
            this.pendingReject = null;
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[onnx-worker] error:', error);
            this.failPending(error.message || 'ONNX worker failed');
        };

        this.worker.onmessageerror = (event) => {
            console.error('[onnx-worker] message error:', event);
            this.failPending('ONNX worker message deserialization failed');
        };
    }

    async load(
        modelUrl: string,
        backend: 'webgpu' | 'wasm',
        options: { wasmPaths?: string; numThreads?: number } = {}
    ): Promise<void> {
        const response = (await this.send({
            type: 'load',
            modelUrl,
            backend,
            wasmPaths: options.wasmPaths,
            numThreads: options.numThreads,
        })) as LoadResponse;

        if (!response.success) {
            throw new Error(response.error || 'Model load failed');
        }
    }

    async runInference(
        specReal: Float32Array,
        specImag: Float32Array,
        audio: Float32Array,
        specShape: number[],
        audioShape: number[]
    ): Promise<InferenceResult> {
        // Transfer the spectrogram buffers instead of structured-cloning them
        // every segment (they're multi-MB). The caller (pipeline) never reads
        // specReal/specImag again after handing them off, so detaching them
        // here is safe. ``audio`` is deliberately NOT transferred: it's one of
        // the pipeline's two reused double-buffers, and transferring it would
        // detach the buffer the next segment writes into.
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
            outWave: response.outWave!,
            outSpecShape: response.outSpecShape!,
            outWaveShape: response.outWaveShape!,
        };
    }

    async unload(): Promise<void> {
        await this.send({ type: 'unload' });
    }

    terminate(): void {
        this.failPending('ONNX worker terminated');
        this.worker.terminate();
    }

    private failPending(message: string): void {
        const reject = this.pendingReject;
        this.pendingResolve = null;
        this.pendingReject = null;
        reject?.(new Error(message));
    }

    private send(
        message: OutgoingMessage,
        transfer: Transferable[] = []
    ): Promise<WorkerResponse> {
        // Single in-flight request by contract: the pipeline awaits each
        // response before issuing the next, so this failPending is normally a
        // no-op. Its real job is recovery: a Separator reuses its workers
        // across separate() calls, so if one run throws mid-pipeline it can
        // leave a never-awaited pending here — clearing it lets the next run
        // start clean instead of resolving against the stale promise. The
        // requestId on the posted message lets onmessage discard any late reply
        // from that aborted run.
        this.failPending('Superseded by a new ONNX request');
        const requestId = ++this.requestCounter;
        this.pendingId = requestId;
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            this.worker.postMessage({ ...message, requestId }, transfer);
        });
    }
}
