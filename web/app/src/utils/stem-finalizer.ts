interface FinalizedStem {
    source: string;
    blob: Blob;
    peaks: number[];
}

interface FinalizeResponse {
    source: string;
    wavBuffer?: ArrayBuffer;
    peaks?: number[];
    error?: string;
}

function abortError(): DOMException {
    return new DOMException('Stem finalization aborted', 'AbortError');
}

/** Encode player WAVs and waveform peaks away from the animation/UI thread. */
export async function finalizeStems(
    stems: Record<string, Float32Array>,
    sampleRate: number,
    signal: AbortSignal,
    onProgress?: (done: number, total: number, source: string) => void,
): Promise<FinalizedStem[]> {
    if (signal.aborted) throw abortError();

    const entries = Object.entries(stems);
    const worker = new Worker(
        new URL('../workers/stem-finalize-worker.ts', import.meta.url),
        { type: 'module' },
    );
    const finalized: FinalizedStem[] = [];

    try {
        for (let index = 0; index < entries.length; index++) {
            if (signal.aborted) throw abortError();
            const [source, audioData] = entries[index];

            const response = await new Promise<FinalizeResponse>((resolve, reject) => {
                const cleanup = () => {
                    signal.removeEventListener('abort', handleAbort);
                    worker.onmessage = null;
                    worker.onerror = null;
                };
                const handleAbort = () => {
                    cleanup();
                    reject(abortError());
                };
                signal.addEventListener('abort', handleAbort, { once: true });
                worker.onmessage = event => {
                    cleanup();
                    resolve(event.data as FinalizeResponse);
                };
                worker.onerror = event => {
                    cleanup();
                    reject(new Error(event.message || `Failed to finalize ${source}`));
                };
                worker.postMessage(
                    { source, audioData, numChannels: 2, sampleRate },
                    [audioData.buffer as ArrayBuffer],
                );
            });

            if (response.error || !response.wavBuffer || !response.peaks) {
                throw new Error(response.error || `Incomplete finalization result for ${source}`);
            }
            finalized.push({
                source,
                blob: new Blob([response.wavBuffer], { type: 'audio/wav' }),
                peaks: response.peaks,
            });
            onProgress?.(index + 1, entries.length, source);
        }
        return finalized;
    } finally {
        worker.terminate();
    }
}
