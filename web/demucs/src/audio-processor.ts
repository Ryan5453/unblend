/**
 * Worker-only module. Each worker builds its own DSP instance via
 * ``createDSP``; the FFT/window state inside is private to that instance.
 * Importing this file from the main thread would share that state across
 * the STFT worker, which is unsafe.
 *
 * Two DSP families:
 * - ``htdemucs``: Demucs pre-padding + frame trims around a √N-normalized
 *   STFT with the Nyquist bin dropped (matches ``HTDemucs._spec``).
 * - ``roformer``: plain centered reflect-pad STFT keeping all bins with no
 *   normalization (matches ``torch.stft(center=True, normalized=False)`` as
 *   the RoFormer checkpoints use it).
 */
import FFT from 'fft.js';
import type { DSPConfig, STFTResult } from './constants.js';

const NUM_CHANNELS = 2;

export interface DSP {
    /** Segment STFT: interleaved stereo in, planar [C][bin][frame] out. */
    computeSTFT(audio: Float32Array): STFTResult;
    /**
     * Segment iSTFT: planar [C][bin][frame] in, planar [C][sample] out of
     * length ``numChannels * segmentSamples`` (a reused internal buffer —
     * copy before the next call).
     */
    computeISTFT(
        real: Float32Array,
        imag: Float32Array,
        numChannels: number,
        numBins: number,
        numFrames: number,
    ): Float32Array;
    readonly numBins: number;
    readonly numFrames: number;
    readonly segmentSamples: number;
}

function reflectIndex(i: number, len: number): number {
    if (len === 1) {
        return 0;
    }
    // Reflect into [0, len) in O(1). The reflection has period 2*(len-1)
    // (edge samples are not repeated), matching torch's pad_mode='reflect':
    //   i < 0      -> -i
    //   i >= len   -> 2*(len-1) - i
    const period = 2 * (len - 1);
    i = ((i % period) + period) % period;
    return i >= len ? period - i : i;
}

/** Build a DSP instance for one model family/geometry. */
export function createDSP(config: DSPConfig): DSP {
    return config.family === 'htdemucs'
        ? createHTDemucsDSP(config)
        : createRoformerDSP(config);
}

function makeHannWindow(nfft: number): Float32Array {
    // Periodic Hann, matching torch.hann_window(nfft) (divide by N, not N-1).
    const window = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) {
        window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / nfft));
    }
    return window;
}

function createHTDemucsDSP(config: DSPConfig): DSP {
    const NFFT = config.nfft;
    const HOP_LENGTH = config.hopLength;
    const SEGMENT_SAMPLES = config.segmentSamples;

    const fftInstance = new FFT(NFFT);
    const hannWindow = makeHannWindow(NFFT);

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

    const ISTFT_PAD = Math.floor(HOP_LENGTH / 2) * 3;
    const ISTFT_LE = HOP_LENGTH * Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH) + 2 * ISTFT_PAD;

    // STFT buffers
    const demucs_padded = [
        new Float32Array(DEMUCS_PADDED_LENGTH),
        new Float32Array(DEMUCS_PADDED_LENGTH),
    ];
    const paddedChannels = [
        new Float32Array(PADDED_LENGTH),
        new Float32Array(PADDED_LENGTH),
    ];
    const rawReal = new Float32Array(NUM_CHANNELS * NUM_BINS * RAW_FRAMES);
    const rawImag = new Float32Array(NUM_CHANNELS * NUM_BINS * RAW_FRAMES);
    const outReal = new Float32Array(NUM_CHANNELS * OUT_BINS * OUT_FRAMES);
    const outImag = new Float32Array(NUM_CHANNELS * OUT_BINS * OUT_FRAMES);
    const fftInput = fftInstance.createComplexArray();
    const fftOutput = fftInstance.createComplexArray();

    // iSTFT buffers. The window-sum reciprocal is identical for every call,
    // so precompute it once.
    const istftPaddedFrames = OUT_FRAMES + 4;
    const windowSum = new Float32Array(ISTFT_LE);
    for (let fp = 0; fp < istftPaddedFrames; fp++) {
        const frameStart = fp * HOP_LENGTH;
        for (let i = 0; i < NFFT; i++) {
            const outIdx = frameStart + i - NFFT / 2;
            if (outIdx >= 0 && outIdx < ISTFT_LE) {
                windowSum[outIdx] += hannWindow[i] * hannWindow[i];
            }
        }
    }
    const windowSumReciprocal = new Float32Array(ISTFT_LE);
    for (let i = 0; i < ISTFT_LE; i++) {
        windowSumReciprocal[i] = windowSum[i] > 1e-8 ? 1.0 / windowSum[i] : 0;
    }
    const istftOutput = new Float32Array(NUM_CHANNELS * ISTFT_LE);
    const istftFinal = new Float32Array(NUM_CHANNELS * SEGMENT_SAMPLES);
    const ifftInput = fftInstance.createComplexArray();
    const ifftOutput = fftInstance.createComplexArray();

    function computeSTFT(audio: Float32Array): STFTResult {
        // The padding/frame constants above are functions of SEGMENT_SAMPLES;
        // any other input length would silently produce garbage.
        if (audio.length !== NUM_CHANNELS * SEGMENT_SAMPLES) {
            throw new Error(
                `computeSTFT expects ${NUM_CHANNELS * SEGMENT_SAMPLES} interleaved ` +
                `samples (${NUM_CHANNELS} ch × segmentSamples), got ${audio.length}`
            );
        }
        const numSamples = audio.length / NUM_CHANNELS;

        demucs_padded[0].fill(0);
        demucs_padded[1].fill(0);
        paddedChannels[0].fill(0);
        paddedChannels[1].fill(0);
        rawReal.fill(0);
        rawImag.fill(0);
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
                    rawReal[binOffset + k] = fftOutput[k * 2] * norm;
                    rawImag[binOffset + k] = fftOutput[k * 2 + 1] * norm;
                }
            }
        }

        for (let c = 0; c < NUM_CHANNELS; c++) {
            for (let f = 0; f < OUT_FRAMES; f++) {
                for (let b = 0; b < OUT_BINS; b++) {
                    const srcIdx = (c * RAW_FRAMES + (f + 2)) * NUM_BINS + b;
                    const dstIdx = c * OUT_BINS * OUT_FRAMES + b * OUT_FRAMES + f;
                    outReal[dstIdx] = rawReal[srcIdx];
                    outImag[dstIdx] = rawImag[srcIdx];
                }
            }
        }

        return { real: outReal, imag: outImag, numBins: OUT_BINS, numFrames: OUT_FRAMES };
    }

    function computeISTFT(
        real: Float32Array,
        imag: Float32Array,
        numChannels: number,
        numBins: number,
        numFrames: number,
    ): Float32Array {
        // The buffers (and the precomputed window sum) are sized from the
        // segment constants; reject anything else loudly.
        if (numChannels !== NUM_CHANNELS || numBins !== OUT_BINS || numFrames !== OUT_FRAMES) {
            throw new Error(
                `computeISTFT expects ${NUM_CHANNELS} ch × ${OUT_BINS} bins × ` +
                `${OUT_FRAMES} frames, got ${numChannels} ch × ${numBins} bins × ` +
                `${numFrames} frames`
            );
        }
        const paddedFrames = numFrames + 4;

        istftOutput.fill(0);

        const halfNfft = NFFT / 2;
        const scale = Math.sqrt(NFFT);
        const channelStride = numBins * numFrames;

        for (let c = 0; c < numChannels; c++) {
            const cBase = c * channelStride;
            const outBase = c * ISTFT_LE;

            for (let fp = 0; fp < paddedFrames; fp++) {
                const f = fp - 2;
                const frameStart = fp * HOP_LENGTH;
                const overlapStart = Math.max(0, halfNfft - frameStart);
                const overlapEnd = Math.min(NFFT, ISTFT_LE - frameStart + halfNfft);

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
                        const negIdx = NFFT - b;
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
                        const negIdx = NFFT - b;
                        ifftInput[negIdx * 2] = 0;
                        ifftInput[negIdx * 2 + 1] = 0;
                    }
                }

                fftInstance.inverseTransform(ifftOutput, ifftInput);

                for (let i = overlapStart; i < overlapEnd; i++) {
                    const outIdx = frameStart + i - halfNfft;
                    istftOutput[outBase + outIdx] += ifftOutput[i * 2] * hannWindow[i];
                }
            }
        }

        for (let c = 0; c < numChannels; c++) {
            const outBase = c * ISTFT_LE;
            const finalBase = c * SEGMENT_SAMPLES;
            for (let i = 0; i < SEGMENT_SAMPLES; i++) {
                istftFinal[finalBase + i] =
                    istftOutput[outBase + ISTFT_PAD + i] * windowSumReciprocal[ISTFT_PAD + i];
            }
        }

        return istftFinal;
    }

    return {
        computeSTFT,
        computeISTFT,
        numBins: OUT_BINS,
        numFrames: OUT_FRAMES,
        segmentSamples: SEGMENT_SAMPLES,
    };
}

function createRoformerDSP(config: DSPConfig): DSP {
    const NFFT = config.nfft;
    const HOP_LENGTH = config.hopLength;
    const SEGMENT_SAMPLES = config.segmentSamples;
    // The centered frame count below is exact only for hop-divisible segments
    // (true for every shipped checkpoint: chunk sizes are multiples of hop).
    if (SEGMENT_SAMPLES % HOP_LENGTH !== 0) {
        throw new Error(
            `roformer segmentSamples (${SEGMENT_SAMPLES}) must be divisible by ` +
            `hopLength (${HOP_LENGTH})`
        );
    }

    const fftInstance = new FFT(NFFT);
    const hannWindow = makeHannWindow(NFFT);

    const CENTER_PAD = NFFT / 2;
    const PADDED_LENGTH = SEGMENT_SAMPLES + 2 * CENTER_PAD;
    const NUM_BINS = NFFT / 2 + 1;
    const NUM_FRAMES = Math.floor(SEGMENT_SAMPLES / HOP_LENGTH) + 1;
    // Overlap-add span of all frames: last frame starts at (frames-1)*hop and
    // extends nfft — equals SEGMENT + NFFT for hop-divisible segments.
    const OUT_LE = (NUM_FRAMES - 1) * HOP_LENGTH + NFFT;

    const paddedChannels = [
        new Float32Array(PADDED_LENGTH),
        new Float32Array(PADDED_LENGTH),
    ];
    const outReal = new Float32Array(NUM_CHANNELS * NUM_BINS * NUM_FRAMES);
    const outImag = new Float32Array(NUM_CHANNELS * NUM_BINS * NUM_FRAMES);
    const fftInput = fftInstance.createComplexArray();
    const fftOutput = fftInstance.createComplexArray();

    // torch.istft envelope: sum of squared windows at each output position.
    const windowSum = new Float32Array(OUT_LE);
    for (let f = 0; f < NUM_FRAMES; f++) {
        const frameStart = f * HOP_LENGTH;
        for (let i = 0; i < NFFT; i++) {
            windowSum[frameStart + i] += hannWindow[i] * hannWindow[i];
        }
    }
    const windowSumReciprocal = new Float32Array(OUT_LE);
    for (let i = 0; i < OUT_LE; i++) {
        windowSumReciprocal[i] = windowSum[i] > 1e-8 ? 1.0 / windowSum[i] : 0;
    }
    const istftOutput = new Float32Array(NUM_CHANNELS * OUT_LE);
    const istftFinal = new Float32Array(NUM_CHANNELS * SEGMENT_SAMPLES);
    const ifftInput = fftInstance.createComplexArray();
    const ifftOutput = fftInstance.createComplexArray();

    function computeSTFT(audio: Float32Array): STFTResult {
        if (audio.length !== NUM_CHANNELS * SEGMENT_SAMPLES) {
            throw new Error(
                `computeSTFT expects ${NUM_CHANNELS * SEGMENT_SAMPLES} interleaved ` +
                `samples (${NUM_CHANNELS} ch × segmentSamples), got ${audio.length}`
            );
        }
        const numSamples = audio.length / NUM_CHANNELS;

        outReal.fill(0);
        outImag.fill(0);

        // torch.stft(center=True): reflect-pad nfft/2 on each side.
        for (let c = 0; c < NUM_CHANNELS; c++) {
            for (let i = 0; i < PADDED_LENGTH; i++) {
                const srcIdx = reflectIndex(i - CENTER_PAD, numSamples);
                paddedChannels[c][i] = audio[srcIdx * NUM_CHANNELS + c];
            }
        }

        for (let c = 0; c < NUM_CHANNELS; c++) {
            const channelData = paddedChannels[c];
            for (let f = 0; f < NUM_FRAMES; f++) {
                const frameStart = f * HOP_LENGTH;
                for (let i = 0; i < NFFT; i++) {
                    fftInput[i * 2] = channelData[frameStart + i] * hannWindow[i];
                    fftInput[i * 2 + 1] = 0;
                }
                fftInstance.transform(fftOutput, fftInput);
                // normalized=False: raw FFT sums, no 1/√N factor. Layout is
                // planar [C][bin][frame], matching the ONNX input (B,C,F,T).
                const base = c * NUM_BINS * NUM_FRAMES;
                for (let k = 0; k < NUM_BINS; k++) {
                    outReal[base + k * NUM_FRAMES + f] = fftOutput[k * 2];
                    outImag[base + k * NUM_FRAMES + f] = fftOutput[k * 2 + 1];
                }
            }
        }

        return { real: outReal, imag: outImag, numBins: NUM_BINS, numFrames: NUM_FRAMES };
    }

    function computeISTFT(
        real: Float32Array,
        imag: Float32Array,
        numChannels: number,
        numBins: number,
        numFrames: number,
    ): Float32Array {
        if (numChannels !== NUM_CHANNELS || numBins !== NUM_BINS || numFrames !== NUM_FRAMES) {
            throw new Error(
                `computeISTFT expects ${NUM_CHANNELS} ch × ${NUM_BINS} bins × ` +
                `${NUM_FRAMES} frames, got ${numChannels} ch × ${numBins} bins × ` +
                `${numFrames} frames`
            );
        }

        istftOutput.fill(0);
        const channelStride = numBins * numFrames;
        const nyquist = NFFT / 2;

        for (let c = 0; c < numChannels; c++) {
            const cBase = c * channelStride;
            const outBase = c * OUT_LE;

            for (let f = 0; f < numFrames; f++) {
                const frameStart = f * HOP_LENGTH;

                // Rebuild the full conjugate-symmetric spectrum. Bins 0 (DC)
                // and nfft/2 (Nyquist) have no mirror partner.
                ifftInput[0] = real[cBase + f];
                ifftInput[1] = imag[cBase + f];
                for (let b = 1; b < nyquist; b++) {
                    const srcIdx = cBase + b * numFrames + f;
                    const rv = real[srcIdx];
                    const iv = imag[srcIdx];
                    ifftInput[b * 2] = rv;
                    ifftInput[b * 2 + 1] = iv;
                    const negIdx = NFFT - b;
                    ifftInput[negIdx * 2] = rv;
                    ifftInput[negIdx * 2 + 1] = -iv;
                }
                const nyIdx = cBase + nyquist * numFrames + f;
                ifftInput[nyquist * 2] = real[nyIdx];
                ifftInput[nyquist * 2 + 1] = imag[nyIdx];

                // fft.js inverseTransform includes the 1/N factor;
                // normalized=False needs no further scaling.
                fftInstance.inverseTransform(ifftOutput, ifftInput);

                for (let i = 0; i < NFFT; i++) {
                    istftOutput[outBase + frameStart + i] += ifftOutput[i * 2] * hannWindow[i];
                }
            }
        }

        // Divide by the window envelope, trim the nfft/2 center padding, and
        // keep exactly segmentSamples (torch.istft(..., length=segment)).
        for (let c = 0; c < numChannels; c++) {
            const outBase = c * OUT_LE;
            const finalBase = c * SEGMENT_SAMPLES;
            for (let i = 0; i < SEGMENT_SAMPLES; i++) {
                istftFinal[finalBase + i] =
                    istftOutput[outBase + CENTER_PAD + i] * windowSumReciprocal[CENTER_PAD + i];
            }
        }

        return istftFinal;
    }

    return {
        computeSTFT,
        computeISTFT,
        numBins: NUM_BINS,
        numFrames: NUM_FRAMES,
        segmentSamples: SEGMENT_SAMPLES,
    };
}
