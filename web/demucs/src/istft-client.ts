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
    /** Offset of the chunk's real samples inside the centered model window. */
    trimOffset: number;
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
    private requestCounter = 0;
    private pendingId = -1;

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
            // Discard late replies from a superseded request (see OnnxClient).
            if (event.data.requestId !== this.pendingId) return;
            const resolve = this.pendingResolve;
            const reject = this.pendingReject;
            this.pendingResolve = null;
            this.pendingReject = null;
            if (event.data.success === false) {
                reject?.(new Error(event.data.error || 'iSTFT worker failed'));
                return;
            }
            resolve?.(event.data);
        };

        this.worker.onerror = (error) => {
            console.error('[istft-worker] error:', error);
            this.failPending(error.message || 'iSTFT worker failed');
        };

        this.worker.onmessageerror = (event) => {
            console.error('[istft-worker] message error:', event);
            this.failPending('iSTFT worker message deserialization failed');
        };
    }

    process(request: ISTFTRequest): Promise<ISTFTResult> {
        // Single in-flight request by contract (see OnnxClient.send): the
        // pipeline awaits each result before the next call, so this clears
        // only a stale pending left by a prior run that aborted mid-pipeline.
        this.failPending('Superseded by a new iSTFT request');
        const requestId = ++this.requestCounter;
        this.pendingId = requestId;
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            // Transfer the multi-MB spec/wave buffers instead of cloning them;
            // the caller hands over fresh copies it never reads again.
            this.worker.postMessage(
                { type: 'process', requestId, ...request },
                [
                    request.specReal.buffer,
                    request.specImag.buffer,
                    request.wave.buffer,
                ] as unknown as Transferable[]
            );
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
