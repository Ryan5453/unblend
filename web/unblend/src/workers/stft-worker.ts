/**
 * Web Worker for STFT computation. Lives in its own worker so it overlaps with
 * the ONNX and iSTFT workers rather than blocking the main thread.
 *
 * The client sends one 'configure' message (model DSP geometry) before the
 * first 'process' (Separator.load always does); 'process' on an unconfigured
 * worker is an error.
 */

import { createDSP, type DSP } from '../audio-processor.js';
import type { DSPConfig } from '../constants.js';

let dsp: DSP | null = null;

interface ConfigureMessage {
    type: 'configure';
    requestId: number;
    config: DSPConfig;
}

interface ProcessMessage {
    type: 'process';
    requestId: number;
    segmentInterleaved: Float32Array;
}

type STFTMessage = ConfigureMessage | ProcessMessage;

interface STFTResponse {
    type: 'result';
    requestId: number;
    success: true;
    real: Float32Array;
    imag: Float32Array;
    numBins: number;
    numFrames: number;
}

interface ConfigureResponse {
    type: 'result';
    requestId: number;
    success: true;
}

interface STFTErrorResponse {
    type: 'result';
    requestId: number;
    success: false;
    error: string;
}

self.onmessage = (event: MessageEvent<STFTMessage>) => {
    const msg = event.data;

    if (msg.type === 'configure') {
        try {
            dsp = createDSP(msg.config);
            const response: ConfigureResponse = {
                type: 'result',
                requestId: msg.requestId,
                success: true,
            };
            self.postMessage(response);
        } catch (error) {
            console.error('[stft-worker] configure failed:', error);
            const response: STFTErrorResponse = {
                type: 'result',
                requestId: msg.requestId,
                success: false,
                error: error instanceof Error ? error.message : String(error),
            };
            self.postMessage(response);
        }
        return;
    }

    const { requestId, segmentInterleaved } = msg;
    try {
        if (!dsp) throw new Error('STFT worker used before configure');

        const stft = dsp.computeSTFT(segmentInterleaved);

        // Copy results since computeSTFT returns references to shared buffers
        const real = new Float32Array(stft.real);
        const imag = new Float32Array(stft.imag);

        const response: STFTResponse = {
            type: 'result',
            requestId,
            success: true,
            real,
            imag,
            numBins: stft.numBins,
            numFrames: stft.numFrames,
        };

        // Transfer ownership to avoid copying
        self.postMessage(response, { transfer: [real.buffer, imag.buffer] as unknown as Transferable[] });
    } catch (error) {
        console.error('[stft-worker] process failed:', error);
        const response: STFTErrorResponse = {
            type: 'result',
            requestId,
            success: false,
            error: error instanceof Error ? error.message : String(error),
        };
        self.postMessage(response);
    }
};
