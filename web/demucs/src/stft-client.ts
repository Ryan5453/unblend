import type { STFTResult } from './constants';

export class STFTClient {
    private worker: Worker;
    private pendingResolve: ((result: STFTResult) => void) | null = null;
    private pendingReject: ((error: Error) => void) | null = null;

    constructor() {
        this.worker = new Worker(
            new URL('./workers/stft-worker.ts', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (event: MessageEvent<STFTResult & { type: string }>) => {
            const resolve = this.pendingResolve;
            this.pendingResolve = null;
            this.pendingReject = null;
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[stft-worker] error:', error);
            this.failPending(error.message || 'STFT worker failed');
        };
    }

    process(segmentInterleaved: Float32Array): Promise<STFTResult> {
        this.failPending('Superseded by a new STFT request');
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            this.worker.postMessage(
                { type: 'process', segmentInterleaved },
                [segmentInterleaved.buffer] as unknown as Transferable[]
            );
        });
    }

    terminate(): void {
        this.failPending('STFT worker terminated');
        this.worker.terminate();
    }

    private failPending(message: string): void {
        const reject = this.pendingReject;
        this.pendingResolve = null;
        this.pendingReject = null;
        reject?.(new Error(message));
    }
}
