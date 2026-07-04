/**
 * Worker-only module. The module-level FFT/window state is private to each
 * worker; importing this file from the main thread would share that state
 * across the STFT worker, which is unsafe.
 */
import FFT from 'fft.js';
import type { STFTResult } from './constants';
import { NFFT, HOP_LENGTH, SEGMENT_SAMPLES } from './constants';

const fftInstance = new FFT(NFFT);

const hannWindow = new Float32Array(NFFT);
for (let i = 0; i < NFFT; i++) {
    hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / NFFT));
}

// Pre-calculate dimensions for buffer allocation
const NUM_CHANNELS = 2;
const LE = Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH);
const DEMUCS_PAD = Math.floor(HOP_LENGTH / 2) * 3;
const DEMUCS_PAD_RIGHT = DEMUCS_PAD + LE * HOP_LENGTH - SEGMENT_SAMPLES;
const DEMUCS_PADDED_LENGTH = DEMUCS_PAD + SEGMENT_SAMPLES + DEMUCS_PAD_RIGHT;
const CENTER_PAD = NFFT / 2;
const PADDED_LENGTH = DEMUCS_PADDED_LENGTH + 2 * CENTER_PAD;
const RAW_FRAMES = Math.floor((PADDED_LENGTH - NFFT) / HOP_LENGTH) + 1;
const NUM_BINS = NFFT / 2 + 1;
const OUT_BINS = NUM_BINS - 1;
const OUT_FRAMES = LE;

// ISTFT dimensions
const ISTFT_PAD = Math.floor(HOP_LENGTH / 2) * 3;
const ISTFT_LE = HOP_LENGTH * Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH) + 2 * ISTFT_PAD;

/**
 * Pre-allocated buffers for STFT computation to avoid repeated allocations
 */
export interface STFTBuffers {
    demucs_padded: [Float32Array, Float32Array];
    paddedChannels: [Float32Array, Float32Array];
    real: Float32Array;
    imag: Float32Array;
    outReal: Float32Array;
    outImag: Float32Array;
    fftInput: number[];
    fftOutput: number[];
}

/**
 * Pre-allocated buffers for ISTFT computation to avoid repeated allocations
 */
export interface ISTFTBuffers {
    output: Float32Array;
    windowSumReciprocal: Float32Array;
    finalOutput: Float32Array;
    ifftInput: number[];
    ifftOutput: number[];
}

/**
 * Create reusable STFT buffers - call once before processing loop
 */
export function createSTFTBuffers(): STFTBuffers {
    return {
        demucs_padded: [
            new Float32Array(DEMUCS_PADDED_LENGTH),
            new Float32Array(DEMUCS_PADDED_LENGTH)
        ],
        paddedChannels: [
            new Float32Array(PADDED_LENGTH),
            new Float32Array(PADDED_LENGTH)
        ],
        real: new Float32Array(NUM_CHANNELS * NUM_BINS * RAW_FRAMES),
        imag: new Float32Array(NUM_CHANNELS * NUM_BINS * RAW_FRAMES),
        outReal: new Float32Array(NUM_CHANNELS * OUT_BINS * OUT_FRAMES),
        outImag: new Float32Array(NUM_CHANNELS * OUT_BINS * OUT_FRAMES),
        fftInput: fftInstance.createComplexArray(),
        fftOutput: fftInstance.createComplexArray(),
    };
}

/**
 * Create reusable ISTFT buffers - call once before processing loop.
 * Precomputes the window sum reciprocal since it's identical for every ISTFT call.
 */
export function createISTFTBuffers(): ISTFTBuffers {
    const paddedFrames = OUT_FRAMES + 4;
    const hopLength = HOP_LENGTH;
    const nfft = NFFT;

    const windowSum = new Float32Array(ISTFT_LE);
    for (let fp = 0; fp < paddedFrames; fp++) {
        const frameStart = fp * hopLength;
        for (let i = 0; i < nfft; i++) {
            const outIdx = frameStart + i - nfft / 2;
            if (outIdx >= 0 && outIdx < ISTFT_LE) {
                windowSum[outIdx] += hannWindow[i] * hannWindow[i];
            }
        }
    }
    const windowSumReciprocal = new Float32Array(ISTFT_LE);
    for (let i = 0; i < ISTFT_LE; i++) {
        windowSumReciprocal[i] = windowSum[i] > 1e-8 ? 1.0 / windowSum[i] : 0;
    }

    return {
        output: new Float32Array(NUM_CHANNELS * ISTFT_LE),
        windowSumReciprocal,
        finalOutput: new Float32Array(NUM_CHANNELS * SEGMENT_SAMPLES),
        ifftInput: fftInstance.createComplexArray(),
        ifftOutput: fftInstance.createComplexArray(),
    };
}

function reflectIndex(i: number, len: number): number {
    if (len === 1) {
        return 0;
    }
    // Reflect into [0, len) in O(1). The reflection has period 2*(len-1)
    // (edge samples are not repeated), matching the iterative mirror below:
    //   i < 0      -> -i
    //   i >= len   -> 2*(len-1) - i
    const period = 2 * (len - 1);
    i = ((i % period) + period) % period;
    return i >= len ? period - i : i;
}

/**
 * Compute STFT using pre-allocated buffers to avoid memory allocations
 */
export function computeSTFT(audio: Float32Array, buffers: STFTBuffers): STFTResult {
    // The padding/frame constants above are compile-time functions of
    // SEGMENT_SAMPLES; any other input length would silently produce garbage.
    if (audio.length !== NUM_CHANNELS * SEGMENT_SAMPLES) {
        throw new Error(
            `computeSTFT expects ${NUM_CHANNELS * SEGMENT_SAMPLES} interleaved ` +
            `samples (${NUM_CHANNELS} ch × SEGMENT_SAMPLES), got ${audio.length}`
        );
    }
    const numSamples = audio.length / NUM_CHANNELS;
    const { demucs_padded, paddedChannels, real, imag, outReal, outImag, fftInput, fftOutput } = buffers;

    demucs_padded[0].fill(0);
    demucs_padded[1].fill(0);
    paddedChannels[0].fill(0);
    paddedChannels[1].fill(0);
    real.fill(0);
    imag.fill(0);
    outReal.fill(0);
    outImag.fill(0);

    for (let c = 0; c < NUM_CHANNELS; c++) {
        for (let i = 0; i < DEMUCS_PADDED_LENGTH; i++) {
            const origIdx = i - DEMUCS_PAD;
            const srcIdx = reflectIndex(origIdx, numSamples);
            demucs_padded[c][i] = audio[srcIdx * NUM_CHANNELS + c];
        }
    }

    for (let c = 0; c < NUM_CHANNELS; c++) {
        for (let i = 0; i < PADDED_LENGTH; i++) {
            const origIdx = i - CENTER_PAD;
            if (origIdx >= 0 && origIdx < DEMUCS_PADDED_LENGTH) {
                paddedChannels[c][i] = demucs_padded[c][origIdx];
            } else {
                const srcIdx = reflectIndex(origIdx, DEMUCS_PADDED_LENGTH);
                paddedChannels[c][i] = demucs_padded[c][srcIdx];
            }
        }
    }

    const norm = 1.0 / Math.sqrt(NFFT);

    for (let c = 0; c < NUM_CHANNELS; c++) {
        const channelData = paddedChannels[c];

        for (let f = 0; f < RAW_FRAMES; f++) {
            const frameStart = f * HOP_LENGTH;

            for (let i = 0; i < NFFT; i++) {
                const idx = frameStart + i;
                if (idx < PADDED_LENGTH) {
                    fftInput[i * 2] = channelData[idx] * hannWindow[i];
                } else {
                    fftInput[i * 2] = 0;
                }
                fftInput[i * 2 + 1] = 0;
            }

            fftInstance.transform(fftOutput, fftInput);

            const binOffset = (c * RAW_FRAMES + f) * NUM_BINS;
            for (let k = 0; k < NUM_BINS; k++) {
                real[binOffset + k] = fftOutput[k * 2] * norm;
                imag[binOffset + k] = fftOutput[k * 2 + 1] * norm;
            }
        }
    }

    for (let c = 0; c < NUM_CHANNELS; c++) {
        for (let f = 0; f < OUT_FRAMES; f++) {
            for (let b = 0; b < OUT_BINS; b++) {
                const srcIdx = (c * RAW_FRAMES + (f + 2)) * NUM_BINS + b;
                const dstIdx = c * OUT_BINS * OUT_FRAMES + b * OUT_FRAMES + f;
                outReal[dstIdx] = real[srcIdx];
                outImag[dstIdx] = imag[srcIdx];
            }
        }
    }

    return { real: outReal, imag: outImag, numBins: OUT_BINS, numFrames: OUT_FRAMES };
}

/**
 * Compute ISTFT using pre-allocated buffers to avoid memory allocations
 */
export function computeISTFT(
    real: Float32Array,
    imag: Float32Array,
    numChannels: number,
    numBins: number,
    numFrames: number,
    targetLength: number,
    buffers: ISTFTBuffers
): Float32Array {
    // The buffers (and the precomputed window sum) are sized from the
    // compile-time segment constants; reject anything else loudly.
    if (
        numChannels !== NUM_CHANNELS ||
        numBins !== OUT_BINS ||
        numFrames !== OUT_FRAMES ||
        targetLength !== SEGMENT_SAMPLES
    ) {
        throw new Error(
            `computeISTFT expects ${NUM_CHANNELS} ch × ${OUT_BINS} bins × ` +
            `${OUT_FRAMES} frames → ${SEGMENT_SAMPLES} samples, got ` +
            `${numChannels} ch × ${numBins} bins × ${numFrames} frames → ` +
            `${targetLength} samples`
        );
    }
    const paddedFrames = numFrames + 4;
    const { output, windowSumReciprocal, finalOutput, ifftInput, ifftOutput } = buffers;

    output.fill(0);

    const nfft = NFFT;
    const halfNfft = nfft / 2;
    const hopLength = HOP_LENGTH;
    const scale = Math.sqrt(nfft);
    const channelStride = numBins * numFrames;

    for (let c = 0; c < numChannels; c++) {
        const cBase = c * channelStride;
        const outBase = c * ISTFT_LE;

        for (let fp = 0; fp < paddedFrames; fp++) {
            const f = fp - 2;
            const frameStart = fp * hopLength;
            const overlapStart = Math.max(0, halfNfft - frameStart);
            const overlapEnd = Math.min(nfft, ISTFT_LE - frameStart + halfNfft);

            if (f >= 0 && f < numFrames) {
                const dcIdx = cBase + f;
                ifftInput[0] = real[dcIdx] * scale;
                ifftInput[1] = imag[dcIdx] * scale;

                for (let b = 1; b < numBins; b++) {
                    const srcIdx = cBase + b * numFrames + f;
                    const rv = real[srcIdx] * scale;
                    const iv = imag[srcIdx] * scale;
                    ifftInput[b * 2] = rv;
                    ifftInput[b * 2 + 1] = iv;
                    const negIdx = nfft - b;
                    ifftInput[negIdx * 2] = rv;
                    ifftInput[negIdx * 2 + 1] = -iv;
                }

                ifftInput[numBins * 2] = 0;
                ifftInput[numBins * 2 + 1] = 0;
            } else {
                for (let b = 0; b <= numBins; b++) {
                    ifftInput[b * 2] = 0;
                    ifftInput[b * 2 + 1] = 0;
                }
                for (let b = 1; b < numBins; b++) {
                    const negIdx = nfft - b;
                    ifftInput[negIdx * 2] = 0;
                    ifftInput[negIdx * 2 + 1] = 0;
                }
            }

            fftInstance.inverseTransform(ifftOutput, ifftInput);

            for (let i = overlapStart; i < overlapEnd; i++) {
                const outIdx = frameStart + i - halfNfft;
                output[outBase + outIdx] += ifftOutput[i * 2] * hannWindow[i];
            }
        }
    }

    for (let c = 0; c < numChannels; c++) {
        const outBase = c * ISTFT_LE;
        const finalBase = c * targetLength;
        for (let i = 0; i < targetLength; i++) {
            finalOutput[finalBase + i] = output[outBase + ISTFT_PAD + i] * windowSumReciprocal[ISTFT_PAD + i];
        }
    }

    return finalOutput;
}
