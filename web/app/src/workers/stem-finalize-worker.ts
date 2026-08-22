/// <reference lib="webworker" />

import { WAVE_BINS } from '../utils/peaks';

interface FinalizeRequest {
    source: string;
    audioData: Float32Array;
    numChannels: number;
    sampleRate: number;
}

interface FinalizeResponse {
    source: string;
    wavBuffer?: ArrayBuffer;
    peaks?: number[];
    error?: string;
}

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

function writeAscii(view: DataView, offset: number, value: string): void {
    for (let i = 0; i < value.length; i++) {
        view.setUint8(offset + i, value.charCodeAt(i));
    }
}

workerScope.onmessage = (event: MessageEvent<FinalizeRequest>) => {
    const { source, audioData, numChannels, sampleRate } = event.data;
    try {
        const frames = Math.floor(audioData.length / numChannels);
        const blockAlign = numChannels * 2;
        const dataSize = frames * blockAlign;
        const wavBuffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(wavBuffer);

        writeAscii(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeAscii(view, 8, 'WAVE');
        writeAscii(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, 16, true);
        writeAscii(view, 36, 'data');
        view.setUint32(40, dataSize, true);

        const perBin = Math.max(1, Math.floor(frames / WAVE_BINS));
        const peaks = new Array<number>(WAVE_BINS).fill(0);
        let globalMax = 1e-6;
        let byteOffset = 44;

        for (let frame = 0; frame < frames; frame++) {
            const bin = Math.floor(frame / perBin);
            let framePeak = 0;
            for (let channel = 0; channel < numChannels; channel++) {
                let sample = audioData[frame * numChannels + channel];
                sample = Math.max(-1, Math.min(1, sample));
                framePeak = Math.max(framePeak, Math.abs(sample));
                const scaled = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
                const quantized = Math.max(-32768, Math.min(32767, Math.round(scaled)));
                view.setInt16(byteOffset, quantized, true);
                byteOffset += 2;
            }
            // Match peaksFromInterleaved: any remainder after the final full
            // bin is deliberately ignored.
            if (bin < WAVE_BINS && framePeak > peaks[bin]) {
                peaks[bin] = framePeak;
                globalMax = Math.max(globalMax, framePeak);
            }
        }

        for (let i = 0; i < peaks.length; i++) {
            peaks[i] = Math.min(1, peaks[i] / globalMax);
        }

        const response: FinalizeResponse = { source, wavBuffer, peaks };
        workerScope.postMessage(response, [wavBuffer]);
    } catch (error) {
        const response: FinalizeResponse = {
            source,
            error: error instanceof Error ? error.message : String(error),
        };
        workerScope.postMessage(response);
    }
};
