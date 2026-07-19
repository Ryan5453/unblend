/**
 * Web Worker for ISTFT computation.
 * Runs ISTFT + freq/time branch combination + overlap-add weighting. Lives in
 * its own worker so it overlaps with the STFT and ONNX workers rather than
 * blocking the main thread.
 *
 * The client sends one 'configure' message (model DSP geometry) before the
 * first 'process'; unconfigured workers fall back to the HTDemucs defaults.
 * RoFormer models have no time branch: 'process' arrives without ``wave`` and
 * the chunk is the weighted iSTFT alone.
 */

import { createDSP, type DSP } from '../audio-processor.js';
import {
    NFFT, HOP_LENGTH, SEGMENT_SAMPLES, createSplitWeight,
    type DSPConfig,
} from '../constants.js';

let dsp: DSP | null = null;
let splitWeight: Float32Array | null = null;
let sourceReal: Float32Array | null = null;
let sourceImag: Float32Array | null = null;

interface ConfigureMessage {
    type: 'configure';
    requestId: number;
    config: DSPConfig;
}

interface ProcessMessage {
    type: 'process';
    requestId: number;
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
    /**
     * Offset of the chunk's real samples inside the model window. The window
     * is centered on the chunk (Python TensorChunk.padded), so a short final
     * chunk sits trimOffset samples into the segment; full chunks use 0.
     */
    trimOffset: number;
}

type ISTFTMessage = ConfigureMessage | ProcessMessage;

interface ISTFTResponse {
    type: 'result';
    requestId: number;
    success: true;
    // Per-source weighted interleaved audio chunks
    chunks: Float32Array[];
    segStart: number;
    segLength: number;
}

interface ConfigureResponse {
    type: 'result';
    requestId: number;
    success: true;
    chunks?: undefined;
}

interface ISTFTErrorResponse {
    type: 'result';
    requestId: number;
    success: false;
    error: string;
}

function setup(config: DSPConfig): void {
    dsp = createDSP(config);
    splitWeight = createSplitWeight(config.segmentSamples);
    // Largest per-channel spectrogram the ONNX model can emit for one
    // segment, ×2 channels. Derived from the DSP so it can't silently
    // undersize (the STFT producer uses the same geometry).
    const maxSpecSize = 2 * dsp.numBins * dsp.numFrames;
    sourceReal = new Float32Array(maxSpecSize);
    sourceImag = new Float32Array(maxSpecSize);
}

self.onmessage = (event: MessageEvent<ISTFTMessage>) => {
    const msg = event.data;

    if (msg.type === 'configure') {
        try {
            setup(msg.config);
            const response: ConfigureResponse = {
                type: 'result',
                requestId: msg.requestId,
                success: true,
            };
            self.postMessage(response);
        } catch (error) {
            console.error('[istft-worker] configure failed:', error);
            const response: ISTFTErrorResponse = {
                type: 'result',
                requestId: msg.requestId,
                success: false,
                error: (error as Error).message,
            };
            self.postMessage(response);
        }
        return;
    }

    try {
        if (!dsp) {
            setup({
                family: 'htdemucs',
                nfft: NFFT,
                hopLength: HOP_LENGTH,
                segmentSamples: SEGMENT_SAMPLES,
            });
        }

        const {
            specReal, specImag, wave,
            numSources, numChannels, numBins, numFrames,
            segStart, segLength, trimOffset,
        } = msg;
        const segmentSamples = dsp!.segmentSamples;

        const specSize = numChannels * numBins * numFrames;
        const chunks: Float32Array[] = [];

        for (let s = 0; s < numSources; s++) {
            const specOffset = s * specSize;
            sourceReal!.set(specReal.subarray(specOffset, specOffset + specSize));
            sourceImag!.set(specImag.subarray(specOffset, specOffset + specSize));

            const freqAudio = dsp!.computeISTFT(
                sourceReal!, sourceImag!,
                numChannels, numBins, numFrames
            );

            // Combine freq + time branches (when the model has one), apply the
            // triangular weight, write to the output chunk. Matches Python:
            // center_trim to the chunk, then weight[:chunk_length] (the
            // triangle indexed from 0, not centered).
            const sourceWaveOffset = s * numChannels * segmentSamples;
            const chunk = new Float32Array(segLength * numChannels);

            for (let i = 0; i < segLength; i++) {
                const srcIdx = trimOffset + i;
                let leftVal = freqAudio[srcIdx];
                let rightVal = freqAudio[segmentSamples + srcIdx];
                if (wave) {
                    leftVal += wave[sourceWaveOffset + srcIdx];
                    rightVal += wave[sourceWaveOffset + segmentSamples + srcIdx];
                }
                const w = splitWeight![i];

                chunk[i * numChannels] = leftVal * w;
                chunk[i * numChannels + 1] = rightVal * w;
            }

            chunks.push(chunk);
        }

        const response: ISTFTResponse = {
            type: 'result',
            requestId: msg.requestId,
            success: true,
            chunks,
            segStart,
            segLength,
        };

        // Transfer ownership of chunk buffers to avoid copying
        self.postMessage(response, { transfer: chunks.map(c => c.buffer) as unknown as Transferable[] });
    } catch (error) {
        console.error('[istft-worker] process failed:', error);
        const response: ISTFTErrorResponse = {
            type: 'result',
            requestId: msg.requestId,
            success: false,
            error: (error as Error).message,
        };
        self.postMessage(response);
    }
};
