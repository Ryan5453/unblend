/**
 * Shared STFT → ONNX → iSTFT pipeline. Pure: takes the worker clients and
 * source list it needs, no module-level state.
 */

import {
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SEGMENT_OVERLAP,
    createSplitWeight,
} from './constants';
import type { OnnxClient } from './onnx-client';
import type { STFTClient } from './stft-client';
import { type ISTFTClient, type ISTFTResult } from './istft-client';

/** Maximum random shift in samples (Python: int(0.5 * model.samplerate)). */
const MAX_SHIFT = Math.floor(0.5 * SAMPLE_RATE);

export interface SeparationProgress {
    /** 1-based segment index that just finished (cumulative across shift rounds). */
    segIdx: number;
    /** Total number of segments for this track across all shift rounds. */
    totalSegs: number;
    /** Convenience: ``segIdx / totalSegs`` ∈ (0, 1]. */
    fraction: number;
}

export interface SeparationOptions {
    /** Fired after every segment completes (after iSTFT accumulation). */
    onProgress?: (progress: SeparationProgress) => void;
    /**
     * Number of random sub-second shifts to average (the Demucs "shift
     * trick"). Each extra shift reruns the whole separation on a randomly
     * shifted copy of the input, so runtime scales linearly. Integer in
     * [1, 20]; defaults to 1.
     */
    shifts?: number;
    /**
     * Optional integer seed for the shift-offset PRNG. With a fixed seed the
     * offsets — and therefore the output samples — are deterministic across
     * runs. Defaults to non-deterministic (``Math.random()``). The PRNG is
     * independent of Python's, so same-seed parity is within JS only. The
     * seed is reduced modulo 2³² (so e.g. ``2**32`` and ``0`` collide, as do
     * ``-1`` and ``2**32 - 1``).
     */
    seed?: number;
}

/** Mulberry32: a 32-bit deterministic PRNG seeded from a single integer. */
function mulberry32(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export interface SeparationResult {
    /** stem name → interleaved stereo Float32Array (L,R,L,R…) of length ``numSamples * 2``. */
    stems: Record<string, Float32Array>;
    /** Total wall time including STFT/iSTFT orchestration. */
    wallMs: number;
    /** Sum of ONNX inference times across all segments. */
    inferenceMs: number;
    /** Number of segments processed (summed across shift rounds). */
    numSegments: number;
}

export interface Pipeline {
    onnx: OnnxClient;
    stft: STFTClient;
    istft: ISTFTClient;
}

export async function runPipeline(
    pipeline: Pipeline,
    audioBuffer: AudioBuffer,
    sources: string[],
    options: SeparationOptions = {}
): Promise<SeparationResult> {
    const { onProgress } = options;
    const shifts = options.shifts ?? 1;
    if (!Number.isInteger(shifts) || shifts < 1 || shifts > 20) {
        throw new Error(
            `shifts must be an integer between 1 and 20 (inclusive), got ${shifts}`
        );
    }
    if (options.seed !== undefined && !Number.isInteger(options.seed)) {
        throw new Error(`seed must be an integer if provided, got ${options.seed}`);
    }
    const rand = options.seed !== undefined ? mulberry32(options.seed) : Math.random;
    const { onnx, stft, istft } = pipeline;
    const numChannels = 2;
    const numSamples = audioBuffer.length;

    const left = audioBuffer.getChannelData(0);
    // Mono is duplicated to both channels; for >2 channels we take the first
    // two. This matches the Python pipeline's convert_audio_channels, which
    // returns wav[:channels] when the source has more channels than the model.
    const right = audioBuffer.numberOfChannels > 1
        ? audioBuffer.getChannelData(1)
        : left;

    // Track-level normalization (Python demucs/api.py _normalize): mean/std
    // are scalars over the channel-mean reference signal, std is unbiased
    // (divide by N-1). Denormalized after separation with the same
    // ``1e-5 + std`` factor.
    let refSum = 0;
    for (let i = 0; i < numSamples; i++) {
        refSum += (left[i] + right[i]) / 2;
    }
    const mean = refSum / numSamples;
    let refVar = 0;
    for (let i = 0; i < numSamples; i++) {
        const d = (left[i] + right[i]) / 2 - mean;
        refVar += d * d;
    }
    const std = numSamples > 1 ? Math.sqrt(refVar / (numSamples - 1)) : 0;
    const norm = 1 / (1e-5 + std);

    // Normalized input zero-padded by MAX_SHIFT on each side for the shift
    // trick (Python apply.py: TensorChunk.padded(length + 2 * max_shift)).
    // Each round separates the view padded[offset : numSamples + MAX_SHIFT]
    // and the rounds are averaged.
    const paddedSamples = numSamples + 2 * MAX_SHIFT;
    const padded = new Float32Array(paddedSamples * numChannels);
    for (let i = 0; i < numSamples; i++) {
        const idx = (MAX_SHIFT + i) * 2;
        padded[idx] = (left[i] - mean) * norm;
        padded[idx + 1] = (right[i] - mean) * norm;
    }

    // Mirror the Python chunking exactly (demucs/apply.py): segments start at
    // every multiple of the stride, the window for a short final chunk is
    // centered on it (real left context from the padded track, zeros past the
    // end), and chunks are blended with the triangular weight normalized by
    // the per-sample weight sum.
    const STEP = Math.floor(SEGMENT_SAMPLES * (1 - SEGMENT_OVERLAP));
    const weight = createSplitWeight();

    // Final accumulators across shift rounds; divided by ``shifts`` and
    // denormalized at the end. These are the returned stem buffers.
    const outputs: Record<string, Float32Array> = {};
    for (const source of sources) {
        outputs[source] = new Float32Array(numSamples * numChannels);
    }

    // Per-round buffers, sized for the longest possible view (offset = 0)
    // and reused across rounds.
    const maxViewLength = numSamples + MAX_SHIFT;
    const roundOut = sources.map(() => new Float32Array(maxViewLength * numChannels));
    const sumWeight = new Float32Array(maxViewLength);

    // Draw all offsets up front so totalSegs is known for progress reporting.
    // Python: random.randint(0, max_shift) — inclusive on both ends.
    const offsets: number[] = [];
    let totalSegs = 0;
    for (let r = 0; r < shifts; r++) {
        const offset = Math.floor(rand() * (MAX_SHIFT + 1));
        offsets.push(offset);
        totalSegs += Math.ceil((numSamples + MAX_SHIFT - offset) / STEP);
    }

    // Double-buffer so we can prepare the next segment while inference reads the current one.
    const planarBuffers = [
        new Float32Array(SEGMENT_SAMPLES * numChannels),
        new Float32Array(SEGMENT_SAMPLES * numChannels),
    ];
    let pendingPlanarIndex = 0;

    const startTime = performance.now();
    let totalInferenceMs = 0;
    let segsDone = 0;

    for (let r = 0; r < shifts; r++) {
        const viewOffset = offsets[r];
        const viewLength = numSamples + MAX_SHIFT - viewOffset;
        const numSegments = Math.ceil(viewLength / STEP);

        for (const buf of roundOut) buf.fill(0);
        sumWeight.fill(0);

        function segmentWindow(seg: number) {
            const segStart = seg * STEP;
            const segLength = Math.min(SEGMENT_SAMPLES, viewLength - segStart);
            const trimOffset = (SEGMENT_SAMPLES - segLength) >> 1;
            return { segStart, segLength, trimOffset, windowStart: segStart - trimOffset };
        }

        function accumulate(result: ISTFTResult) {
            const { chunks, segStart, segLength } = result;
            for (let s = 0; s < sources.length; s++) {
                const chunk = chunks[s];
                const out = roundOut[s];
                for (let i = 0; i < segLength; i++) {
                    const outIdx = (segStart + i) * numChannels;
                    out[outIdx] += chunk[i * numChannels];
                    out[outIdx + 1] += chunk[i * numChannels + 1];
                }
            }
            // Chunks arrive pre-weighted from the iSTFT worker; track the matching
            // weight sum so the final normalization divides it back out.
            for (let i = 0; i < segLength; i++) {
                sumWeight[segStart + i] += weight[i];
            }
        }

        // ``windowStart`` is view-relative; reads index the underlying padded
        // track (Python TensorChunk.padded reads the full tensor), so a
        // centered window pulls real context across the view edges and zeros
        // outside the padded track.
        function prepareInterleaved(windowStart: number): Float32Array {
            const interleaved = new Float32Array(SEGMENT_SAMPLES * numChannels);
            const base = viewOffset + windowStart;
            const from = Math.max(0, -base);
            const to = Math.min(SEGMENT_SAMPLES, paddedSamples - base);
            for (let i = from; i < to; i++) {
                const srcIdx = (base + i) * 2;
                interleaved[i * 2] = padded[srcIdx];
                interleaved[i * 2 + 1] = padded[srcIdx + 1];
            }
            return interleaved;
        }

        function preparePlanar(buffer: Float32Array, windowStart: number): Float32Array {
            buffer.fill(0);
            const base = viewOffset + windowStart;
            const from = Math.max(0, -base);
            const to = Math.min(SEGMENT_SAMPLES, paddedSamples - base);
            for (let i = from; i < to; i++) {
                const srcIdx = (base + i) * 2;
                buffer[i] = padded[srcIdx];
                buffer[SEGMENT_SAMPLES + i] = padded[srcIdx + 1];
            }
            return buffer;
        }

        // ``.catch(() => {})`` silences unhandled-rejection warnings when an
        // exception aborts the loop before we await these promises.
        const seg0 = segmentWindow(0);
        let pendingStft = stft.process(prepareInterleaved(seg0.windowStart));
        pendingStft.catch(() => {});
        let pendingPlanar = preparePlanar(planarBuffers[pendingPlanarIndex], seg0.windowStart);
        let prevIstftPromise: Promise<ISTFTResult> | null = null;

        for (let seg = 0; seg < numSegments; seg++) {
            const { segStart, segLength, trimOffset } = segmentWindow(seg);

            // Yield to the event loop occasionally so the UI can repaint.
            // setTimeout rather than requestAnimationFrame: rAF never fires in
            // hidden tabs, which would stall the pipeline.
            if (seg % 5 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }

            const stftResult = await pendingStft;
            const currentPlanar = pendingPlanar;

            const specShape = [1, numChannels, stftResult.numBins, stftResult.numFrames];
            const audioShape = [1, numChannels, SEGMENT_SAMPLES];

            const inferenceStart = performance.now();
            const inferencePromise = onnx.runInference(
                stftResult.real, stftResult.imag, currentPlanar, specShape, audioShape
            );

            if (seg + 1 < numSegments) {
                const next = segmentWindow(seg + 1);
                pendingStft = stft.process(prepareInterleaved(next.windowStart));
                pendingStft.catch(() => {});
                pendingPlanarIndex = 1 - pendingPlanarIndex;
                pendingPlanar = preparePlanar(
                    planarBuffers[pendingPlanarIndex], next.windowStart
                );
            }

            const results = await inferencePromise;
            totalInferenceMs += performance.now() - inferenceStart;

            if (prevIstftPromise) {
                accumulate(await prevIstftPromise);
            }

            // The result buffers were transferred from the ONNX worker and are
            // exclusively owned here, so hand them straight to the iSTFT worker.
            prevIstftPromise = istft.process({
                specReal: results.outSpecReal,
                specImag: results.outSpecImag,
                wave: results.outWave,
                numSources: sources.length,
                numChannels,
                numBins: stftResult.numBins,
                numFrames: stftResult.numFrames,
                segStart, segLength, trimOffset,
            });
            prevIstftPromise.catch(() => {});

            segsDone++;
            onProgress?.({
                segIdx: segsDone,
                totalSegs,
                fraction: segsDone / totalSegs,
            });
        }

        if (prevIstftPromise) {
            accumulate(await prevIstftPromise);
        }

        // Normalize by the accumulated weight sum (Python: out / sum_weight),
        // then trim the shift back out: drop the first MAX_SHIFT - offset
        // samples and keep numSamples (Python: partial[..., max_shift - offset:]
        // [..., :length]). Every retained sample is covered by at least one
        // chunk, so sumWeight > 0.
        const trimStart = MAX_SHIFT - viewOffset;
        for (let s = 0; s < sources.length; s++) {
            const out = outputs[sources[s]];
            const round = roundOut[s];
            for (let i = 0; i < numSamples; i++) {
                const w = sumWeight[trimStart + i];
                const srcIdx = (trimStart + i) * 2;
                out[i * 2] += round[srcIdx] / w;
                out[i * 2 + 1] += round[srcIdx + 1] / w;
            }
        }
    }

    // Average the shift rounds and denormalize (Python: out * (1e-5 + std) + mean).
    const denormScale = (1e-5 + std) / shifts;
    for (const source of sources) {
        const out = outputs[source];
        for (let i = 0; i < numSamples * numChannels; i++) {
            out[i] = out[i] * denormScale + mean;
        }
    }

    return {
        stems: outputs,
        wallMs: performance.now() - startTime,
        inferenceMs: totalInferenceMs,
        numSegments: totalSegs,
    };
}
