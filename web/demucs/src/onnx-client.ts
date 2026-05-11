interface LoadModelMessage {
    type: 'load';
    modelUrl: string;
    backend: 'webgpu' | 'wasm';
    wasmPaths?: string;
    numThreads?: number;
}

interface RunInferenceMessage {
    type: 'run';
    specReal: Float32Array;
    specImag: Float32Array;
    audio: Float32Array;
    specShape: number[];
    audioShape: number[];
}

interface UnloadMessage {
    type: 'unload';
}

interface LoadResponse {
    type: 'load';
    success: boolean;
    backend?: 'webgpu' | 'wasm';
    error?: string;
}

interface RunResponse {
    type: 'run';
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
    success: boolean;
}

type WorkerResponse = LoadResponse | RunResponse | UnloadResponse;
type Message = LoadModelMessage | RunInferenceMessage | UnloadMessage;

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

    constructor() {
        this.worker = new Worker(
            new URL('./workers/onnx-worker.ts', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
            const resolve = this.pendingResolve;
            this.pendingResolve = null;
            this.pendingReject = null;
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[onnx-worker] error:', error);
            this.failPending(error.message || 'ONNX worker failed');
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
        const response = (await this.send({
            type: 'run',
            specReal,
            specImag,
            audio,
            specShape,
            audioShape,
        })) as RunResponse;

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

    private send(message: Message): Promise<WorkerResponse> {
        this.failPending('Superseded by a new ONNX request');
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            this.worker.postMessage(message);
        });
    }
}
