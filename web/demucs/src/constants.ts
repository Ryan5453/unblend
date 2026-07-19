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

export type ModelType =
    | 'htdemucs'
    | 'htdemucs_6s'
    | 'bs_roformer_sw'
    | 'melband_roformer_kim';

export type ModelFamily = 'htdemucs' | 'roformer';

/**
 * Everything the pipeline needs to know about one model. The two families
 * share the chunk/overlap-add machinery but differ in DSP:
 *
 * - `htdemucs`: Demucs pre-padding + frame trims around a √N-normalized STFT
 *   (Nyquist bin dropped), track-level mean/std input normalization, and a
 *   time-domain branch (`out_wave`) added to the iSTFT result.
 * - `roformer`: plain centered reflect-pad STFT (all `nfft/2 + 1` bins, no
 *   normalization — the checkpoints use `torch.stft(normalized=False)`), raw
 *   audio in (no normalization), spectrogram masking only (no time branch).
 */
export interface ModelConfig {
    family: ModelFamily;
    nfft: number;
    hopLength: number;
    /** The ONNX graph is traced at exactly this many samples per segment. */
    segmentSamples: number;
    /** Stems the ONNX graph emits, in output order. */
    modelSources: string[];
    /** Final stems returned to the caller (includes any complement stem). */
    sources: string[];
    /**
     * Single-mask models emit one stem; the second is computed client-side
     * as ``mixture - stem`` after separation.
     */
    complement?: { stem: string; name: string };
    /** Track-level mean/std normalization around the model (HTDemucs only). */
    normalizeInput: boolean;
    /** Whether the graph has the HTDemucs time-domain branch (``out_wave``). */
    hasTimeBranch: boolean;
    /** License of the model weights (shown so apps can surface it). */
    license: string;
}

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
    'htdemucs': {
        family: 'htdemucs',
        nfft: NFFT,
        hopLength: HOP_LENGTH,
        segmentSamples: SEGMENT_SAMPLES,
        modelSources: ['drums', 'bass', 'other', 'vocals'],
        sources: ['drums', 'bass', 'other', 'vocals'],
        normalizeInput: true,
        hasTimeBranch: true,
        license: 'unlicensed',
    },
    'htdemucs_6s': {
        family: 'htdemucs',
        nfft: NFFT,
        hopLength: HOP_LENGTH,
        segmentSamples: SEGMENT_SAMPLES,
        modelSources: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
        sources: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
        normalizeInput: true,
        hasTimeBranch: true,
        license: 'unlicensed',
    },
    'bs_roformer_sw': {
        family: 'roformer',
        nfft: 2048,
        hopLength: 512,
        segmentSamples: 588800, // 13.35s; traced chunk length of the checkpoint
        modelSources: ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano'],
        sources: ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano'],
        normalizeInput: false,
        hasTimeBranch: false,
        license: 'CC-BY-NC-SA-4.0',
    },
    'melband_roformer_kim': {
        family: 'roformer',
        nfft: 2048,
        hopLength: 441,
        segmentSamples: 352800, // 8s; traced chunk length of the checkpoint
        modelSources: ['vocals'],
        sources: ['vocals', 'other'],
        complement: { stem: 'vocals', name: 'other' },
        normalizeInput: false,
        hasTimeBranch: false,
        license: 'CC-BY-NC-SA-4.0',
    },
};

/**
 * Spectrogram dims the ONNX graph expects for one segment of ``config``.
 * HTDemucs drops the Nyquist bin and trims to ``ceil(segment / hop)`` frames
 * (its Demucs-specific padding); RoFormer keeps all bins and the standard
 * centered frame count.
 */
export function specDims(config: ModelConfig): { numBins: number; numFrames: number } {
    if (config.family === 'htdemucs') {
        return {
            numBins: config.nfft / 2,
            numFrames: Math.ceil(config.segmentSamples / config.hopLength),
        };
    }
    return {
        numBins: config.nfft / 2 + 1,
        numFrames: Math.floor(config.segmentSamples / config.hopLength) + 1,
    };
}

/**
 * Triangular cross-fade weight matching the Python pipeline's _split_weight
 * (unblend/apply.py, transition_power=1): rising 1..half, falling
 * (SEGMENT-half)..1, normalized by the peak. The iSTFT worker applies it to
 * each chunk; the pipeline accumulates the same values into a per-sample
 * weight sum and divides at the end.
 */
export function createSplitWeight(segmentSamples: number = SEGMENT_SAMPLES): Float32Array {
    const half = Math.floor(segmentSamples / 2);
    const peak = Math.max(half, segmentSamples - half);
    const weight = new Float32Array(segmentSamples);
    for (let i = 0; i < segmentSamples; i++) {
        weight[i] = (i < half ? i + 1 : segmentSamples - i) / peak;
    }
    return weight;
}

export interface STFTResult {
    real: Float32Array;
    imag: Float32Array;
    numBins: number;
    numFrames: number;
}

/** DSP parameters a worker needs to build its FFT state for one model. */
export interface DSPConfig {
    family: ModelFamily;
    nfft: number;
    hopLength: number;
    segmentSamples: number;
}

/** Extract the DSP subset of a model config (what the workers consume). */
export function dspConfig(config: ModelConfig): DSPConfig {
    return {
        family: config.family,
        nfft: config.nfft,
        hopLength: config.hopLength,
        segmentSamples: config.segmentSamples,
    };
}
