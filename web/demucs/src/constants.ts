export const SAMPLE_RATE = 44100;
export const NFFT = 4096;
export const HOP_LENGTH = NFFT / 4;
/**
 * HTDemucs training length (7.8s @ 44.1kHz). The ONNX graph is traced at
 * exactly this size, so callers must feed segments of this length.
 */
export const SEGMENT_SAMPLES = 343980;
export const SEGMENT_SECONDS = SEGMENT_SAMPLES / SAMPLE_RATE;
export const SEGMENT_OVERLAP = 0.25;

export type ModelType = 'htdemucs' | 'htdemucs_6s';

export interface STFTResult {
    real: Float32Array;
    imag: Float32Array;
    numBins: number;
    numFrames: number;
}
