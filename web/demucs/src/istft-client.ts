import type { DSPConfig } from './constants.js';

interface ISTFTRequest {
    specReal: Float32Array;
    specImag: Float32Array;
    /** Absent for models without a time-domain branch (RoFormer). */
    wave?: Float32Array;
    numSources: number;
    numChannels: number;
    numBins: number;
    numFrames: number;
    segStart: number;
    segLength: number;
    /** Offset of the chunk's real samples inside the centered model window. */
    trimOffset: number;
}

export interface ISTFTResult {
    chunks: Float32Array[];
    segStart: number;
    segLength: number;
}

function asError(reason: unknown, fallback: string): Error {
    if (reason instanceof Error) return reason;
    return new Error(reason === undefined ? fallback : String(reason));
}

export class ISTFTClient {
    private worker: Worker;
    private pendingResolve: ((result: ISTFTResult) => void) | null = null;
    private pendingReject: ((error: Error) => void) | null = null;
    private requestCounter = 0;
    private pendingId = -1;
    private terminated = false;

    constructor() {
        this.worker = new Worker(
            new URL('./workers/istft-worker.js', import.meta.url),
            { type: 'module' }
        );

        this.worker.onmessage = (
            event: MessageEvent<
                ISTFTResult & { type: string; requestId: number; success: boolean; error?: string }
            >
        ) => {
            if (this.terminated || event.data.requestId !== this.pendingId) return;
            const resolve = this.pendingResolve;
            const reject = this.pendingReject;
            this.clearPending();
            if (event.data.success === false) {
                reject?.(new Error(event.data.error || 'iSTFT worker failed'));
                return;
            }
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[istft-worker] error:', error);
            this.terminate(error.message || 'iSTFT worker failed');
        };

        this.worker.onmessageerror = (event) => {
            console.error('[istft-worker] message error:', event);
            this.terminate('iSTFT worker message deserialization failed');
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

    process(request: ISTFTRequest): Promise<ISTFTResult> {
        const transfer: Transferable[] = [
            request.specReal.buffer,
            request.specImag.buffer,
        ];
        if (request.wave) transfer.push(request.wave.buffer);
        return this.send(
            { type: 'process', ...request },
            transfer,
            result => result,
        );
    }

    terminate(reason?: unknown): void {
        if (this.terminated) return;
        this.terminated = true;
        this.rejectPending(asError(reason, 'iSTFT worker terminated'));
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
        message: ({ type: 'configure'; config: DSPConfig } | ({ type: 'process' } & ISTFTRequest)),
        transfer: Transferable[],
        project: (result: ISTFTResult) => T,
    ): Promise<T> {
        if (this.terminated) {
            return Promise.reject(new Error('iSTFT worker has been terminated'));
        }
        if (this.pendingReject !== null) {
            return Promise.reject(new Error('iSTFT worker request already in progress'));
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
