/**
 * Web Worker for STFT computation. Lives in its own worker so it overlaps with
 * the ONNX and iSTFT workers rather than blocking the main thread.
 */

import { computeSTFT, createSTFTBuffers, type STFTBuffers } from '../audio-processor';

let stftBuffers: STFTBuffers | null = null;

interface STFTMessage {
    type: 'process';
    requestId: number;
    segmentInterleaved: Float32Array;
}

interface STFTResponse {
    type: 'result';
    requestId: number;
    real: Float32Array;
    imag: Float32Array;
    numBins: number;
    numFrames: number;
}

self.onmessage = (event: MessageEvent<STFTMessage>) => {
    if (!stftBuffers) {
        stftBuffers = createSTFTBuffers();
    }

    const { requestId, segmentInterleaved } = event.data;
    const stft = computeSTFT(segmentInterleaved, stftBuffers);

    // Copy results since computeSTFT returns references to shared buffers
    const real = new Float32Array(stft.real);
    const imag = new Float32Array(stft.imag);

    const response: STFTResponse = {
        type: 'result',
        requestId,
        real,
        imag,
        numBins: stft.numBins,
        numFrames: stft.numFrames,
    };

    // Transfer ownership to avoid copying
    self.postMessage(response, { transfer: [real.buffer, imag.buffer] as unknown as Transferable[] });
};
