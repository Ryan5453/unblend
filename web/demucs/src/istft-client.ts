interface ISTFTRequest {
    specReal: Float32Array;
    specImag: Float32Array;
    wave: Float32Array;
    numSources: number;
    numChannels: number;
    numBins: number;
    numFrames: number;
    segStart: number;
    segLength: number;
    seg: number;
    numSegments: number;
    numSamples: number;
    fadeIn: Float32Array;
    fadeOut: Float32Array;
    overlap: number;
}

export interface ISTFTResult {
    chunks: Float32Array[];
    segStart: number;
    segLength: number;
}

export class ISTFTClient {
    private worker: Worker;
    private pendingResolve: ((result: ISTFTResult) => void) | null = null;
    private pendingReject: ((error: Error) => void) | null = null;

    constructor() {
        this.worker = new Worker(
            new URL('./workers/istft-worker.ts', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (event: MessageEvent<ISTFTResult & { type: string }>) => {
            const resolve = this.pendingResolve;
            this.pendingResolve = null;
            this.pendingReject = null;
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[istft-worker] error:', error);
            this.failPending(error.message || 'iSTFT worker failed');
        };
    }

    process(request: ISTFTRequest): Promise<ISTFTResult> {
        this.failPending('Superseded by a new iSTFT request');
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            this.worker.postMessage({ type: 'process', ...request });
        });
    }

    terminate(): void {
        this.failPending('iSTFT worker terminated');
        this.worker.terminate();
    }

    private failPending(message: string): void {
        const reject = this.pendingReject;
        this.pendingResolve = null;
        this.pendingReject = null;
        reject?.(new Error(message));
    }
}
