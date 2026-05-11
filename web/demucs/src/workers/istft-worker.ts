/**
 * Web Worker for ISTFT computation.
 * Runs ISTFT + freq/time branch combination + overlap-add weighting
 * in parallel with GPU inference on the main thread.
 */

import { computeISTFT, createISTFTBuffers, type ISTFTBuffers } from '../audio-processor';
import { SEGMENT_SAMPLES } from '../constants';

let istftBuffers: ISTFTBuffers | null = null;
const sourceReal = new Float32Array(2 * (4096 / 2) * Math.ceil(SEGMENT_SAMPLES / (4096 / 4)));
const sourceImag = new Float32Array(sourceReal.length);

interface ISTFTMessage {
    type: 'process';
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

interface ISTFTResponse {
    type: 'result';
    // Per-source weighted interleaved audio chunks
    chunks: Float32Array[];
    segStart: number;
    segLength: number;
}

self.onmessage = (event: MessageEvent<ISTFTMessage>) => {
    const msg = event.data;

    if (!istftBuffers) {
        istftBuffers = createISTFTBuffers();
    }

    const {
        specReal, specImag, wave,
        numSources, numChannels, numBins, numFrames,
        segStart, segLength, seg, numSegments, numSamples,
        fadeIn, fadeOut, overlap,
    } = msg;

    const specSize = numChannels * numBins * numFrames;
    const chunks: Float32Array[] = [];

    for (let s = 0; s < numSources; s++) {
        const specOffset = s * specSize;
        sourceReal.set(specReal.subarray(specOffset, specOffset + specSize));
        sourceImag.set(specImag.subarray(specOffset, specOffset + specSize));

        const freqAudio = computeISTFT(
            sourceReal, sourceImag,
            numChannels, numBins, numFrames,
            SEGMENT_SAMPLES, istftBuffers
        );

        // Combine freq + time branches, apply fade weights, write to output chunk
        const sourceWaveOffset = s * numChannels * SEGMENT_SAMPLES;
        const chunk = new Float32Array(segLength * numChannels);

        for (let i = 0; i < segLength; i++) {
            const globalIdx = segStart + i;
            if (globalIdx >= numSamples) continue;

            const leftVal = freqAudio[i] + wave[sourceWaveOffset + i];
            const rightVal = freqAudio[SEGMENT_SAMPLES + i] + wave[sourceWaveOffset + SEGMENT_SAMPLES + i];

            let weight = 1.0;
            if (seg > 0 && i < overlap) {
                weight = fadeIn[i];
            }
            if (seg < numSegments - 1 && i >= SEGMENT_SAMPLES - overlap) {
                weight = fadeOut[i - (SEGMENT_SAMPLES - overlap)];
            }

            chunk[i * numChannels] = leftVal * weight;
            chunk[i * numChannels + 1] = rightVal * weight;
        }

        chunks.push(chunk);
    }

    const response: ISTFTResponse = {
        type: 'result',
        chunks,
        segStart,
        segLength,
    };

    // Transfer ownership of chunk buffers to avoid copying
    self.postMessage(response, { transfer: chunks.map(c => c.buffer) as unknown as Transferable[] });
};
