import type { DSPConfig, STFTResult } from './constants.js';

function asError(reason: unknown, fallback: string): Error {
    if (reason instanceof Error) return reason;
    return new Error(reason === undefined ? fallback : String(reason));
}

export class STFTClient {
    private worker: Worker;
    private pendingResolve: ((result: STFTResult) => void) | null = null;
    private pendingReject: ((error: Error) => void) | null = null;
    private requestCounter = 0;
    private pendingId = -1;
    private terminated = false;

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
            if (this.terminated || event.data.requestId !== this.pendingId) return;
            const resolve = this.pendingResolve;
            const reject = this.pendingReject;
            this.clearPending();
            if (event.data.success === false) {
                reject?.(new Error(event.data.error || 'STFT worker failed'));
                return;
            }
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[stft-worker] error:', error);
            this.terminate(error.message || 'STFT worker failed');
        };

        this.worker.onmessageerror = (event) => {
            console.error('[stft-worker] message error:', event);
            this.terminate('STFT worker message deserialization failed');
        };
    }

    /** Install the model's DSP geometry before the first process call. */
    configure(config: DSPConfig): Promise<void> {
        return this.send(
            { type: 'configure', config },
            [],
            () => undefined,
        );
    }

    process(segmentInterleaved: Float32Array): Promise<STFTResult> {
        return this.send(
            { type: 'process', segmentInterleaved },
            [segmentInterleaved.buffer],
            result => result,
        );
    }

    terminate(reason?: unknown): void {
        if (this.terminated) return;
        this.terminated = true;
        this.rejectPending(asError(reason, 'STFT worker terminated'));
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

    private rejectPending(reason: Error): void {
        const reject = this.pendingReject;
        this.clearPending();
        reject?.(reason);
    }

    private send<T>(
        message: { type: 'configure'; config: DSPConfig } | {
            type: 'process'; segmentInterleaved: Float32Array;
        },
        transfer: Transferable[],
        project: (result: STFTResult) => T,
    ): Promise<T> {
        if (this.terminated) {
            return Promise.reject(new Error('STFT worker has been terminated'));
        }
        if (this.pendingReject !== null) {
            return Promise.reject(new Error('STFT worker request already in progress'));
        }

        const requestId = ++this.requestCounter;
        this.pendingId = requestId;
        return new Promise((resolve, reject) => {
            this.pendingResolve = result => resolve(project(result));
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
