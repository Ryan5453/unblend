/**
 * Web Worker for ISTFT computation.
 * Runs ISTFT + freq/time branch combination + overlap-add weighting. Lives in
 * its own worker so it overlaps with the STFT and ONNX workers rather than
 * blocking the main thread.
 */

import { computeISTFT, createISTFTBuffers, type ISTFTBuffers } from '../audio-processor';
import { NFFT, HOP_LENGTH, SEGMENT_SAMPLES, createSplitWeight } from '../constants';

let istftBuffers: ISTFTBuffers | null = null;
const splitWeight = createSplitWeight();
// Largest per-channel spectrogram the ONNX model can emit for one segment:
// OUT_BINS (Nyquist dropped → NFFT/2) × OUT_FRAMES (ceil(SEGMENT/HOP)), ×2
// channels. Derived from the constants so it can't silently undersize if the
// FFT/hop/segment sizes change (the STFT producer uses the same constants).
const MAX_SPEC_SIZE = 2 * (NFFT / 2) * Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH);
const sourceReal = new Float32Array(MAX_SPEC_SIZE);
const sourceImag = new Float32Array(sourceReal.length);

interface ISTFTMessage {
    type: 'process';
    requestId: number;
    specReal: Float32Array;
    specImag: Float32Array;
    wave: Float32Array;
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

interface ISTFTResponse {
    type: 'result';
    requestId: number;
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
        segStart, segLength, trimOffset,
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

        // Combine freq + time branches, apply the triangular weight, write to
        // the output chunk. Matches Python: center_trim to the chunk, then
        // weight[:chunk_length] (the triangle indexed from 0, not centered).
        const sourceWaveOffset = s * numChannels * SEGMENT_SAMPLES;
        const chunk = new Float32Array(segLength * numChannels);

        for (let i = 0; i < segLength; i++) {
            const srcIdx = trimOffset + i;
            const leftVal = freqAudio[srcIdx] + wave[sourceWaveOffset + srcIdx];
            const rightVal = freqAudio[SEGMENT_SAMPLES + srcIdx] + wave[sourceWaveOffset + SEGMENT_SAMPLES + srcIdx];
            const w = splitWeight[i];

            chunk[i * numChannels] = leftVal * w;
            chunk[i * numChannels + 1] = rightVal * w;
        }

        chunks.push(chunk);
    }

    const response: ISTFTResponse = {
        type: 'result',
        requestId: msg.requestId,
        chunks,
        segStart,
        segLength,
    };

    // Transfer ownership of chunk buffers to avoid copying
    self.postMessage(response, { transfer: chunks.map(c => c.buffer) as unknown as Transferable[] });
};
