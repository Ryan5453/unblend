import type { DSPConfig, STFTResult } from './constants.js';

export class STFTClient {
    private worker: Worker;
    private pendingResolve: ((result: STFTResult) => void) | null = null;
    private pendingReject: ((error: Error) => void) | null = null;
    private requestCounter = 0;
    private pendingId = -1;

    constructor() {
        this.worker = new Worker(
            new URL('./workers/stft-worker.js', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (
            event: MessageEvent<
                STFTResult & { type: string; requestId: number; success: boolean; error?: string }
            >
        ) => {
            // Discard late replies from a superseded request (see OnnxClient).
            if (event.data.requestId !== this.pendingId) return;
            const resolve = this.pendingResolve;
            const reject = this.pendingReject;
            this.pendingResolve = null;
            this.pendingReject = null;
            if (event.data.success === false) {
                reject?.(new Error(event.data.error || 'STFT worker failed'));
                return;
            }
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[stft-worker] error:', error);
            this.failPending(error.message || 'STFT worker failed');
        };

        this.worker.onmessageerror = (event) => {
            console.error('[stft-worker] message error:', event);
            this.failPending('STFT worker message deserialization failed');
        };
    }

    /**
     * Install the model's DSP geometry in the worker. Must resolve before the
     * first process() call; unconfigured workers assume HTDemucs defaults.
     */
    configure(config: DSPConfig): Promise<void> {
        this.failPending('Superseded by an STFT configure request');
        const requestId = ++this.requestCounter;
        this.pendingId = requestId;
        return new Promise((resolve, reject) => {
            this.pendingResolve = () => resolve();
            this.pendingReject = reject;
            this.worker.postMessage({ type: 'configure', requestId, config });
        });
    }

    process(segmentInterleaved: Float32Array): Promise<STFTResult> {
        // Single in-flight request by contract (see OnnxClient.send): the
        // pipeline awaits each result before the next call, so this clears
        // only a stale pending left by a prior run that aborted mid-pipeline.
        this.failPending('Superseded by a new STFT request');
        const requestId = ++this.requestCounter;
        this.pendingId = requestId;
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            this.worker.postMessage(
                { type: 'process', requestId, segmentInterleaved },
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
