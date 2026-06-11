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

/**
 * Triangular cross-fade weight matching the Python pipeline's _split_weight
 * (demucs/apply.py, transition_power=1): rising 1..half, falling
 * (SEGMENT-half)..1, normalized by the peak. The iSTFT worker applies it to
 * each chunk; the pipeline accumulates the same values into a per-sample
 * weight sum and divides at the end.
 */
export function createSplitWeight(): Float32Array {
    const half = Math.floor(SEGMENT_SAMPLES / 2);
    const peak = Math.max(half, SEGMENT_SAMPLES - half);
    const weight = new Float32Array(SEGMENT_SAMPLES);
    for (let i = 0; i < SEGMENT_SAMPLES; i++) {
        weight[i] = (i < half ? i + 1 : SEGMENT_SAMPLES - i) / peak;
    }
    return weight;
}

export interface STFTResult {
    real: Float32Array;
    imag: Float32Array;
    numBins: number;
    numFrames: number;
}
