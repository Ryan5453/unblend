/**
 * Shared STFT → ONNX → iSTFT pipeline. Pure: takes the worker clients and
 * model config it needs, no module-level state. Family differences are
 * config-driven: HTDemucs normalizes the input and combines a time-domain
 * branch; RoFormer feeds raw audio, has no time branch, and single-mask
 * checkpoints get a ``mixture - stem`` complement computed at the end.
 */

import {
    SAMPLE_RATE,
    createSplitWeight,
    specDims,
    type ModelConfig,
} from './constants.js';
import type { OnnxClient } from './onnx-client.js';
import type { STFTClient } from './stft-client.js';
import type { ISTFTClient, ISTFTResult } from './istft-client.js';
import { StreamingOverlapAccumulator } from './overlap-accumulator.js';

/** Maximum random shift in samples (Python: int(0.5 * model.samplerate)). */
const MAX_SHIFT = Math.floor(0.5 * SAMPLE_RATE);
const SEGMENT_OVERLAP = 0.25;

export interface SeparationProgress {
    /** 1-based segment index that just finished (cumulative across shift rounds). */
    segIdx: number;
    /** Total number of segments for this track across all shift rounds. */
    totalSegs: number;
    /** Convenience: ``segIdx / totalSegs`` ∈ (0, 1]. */
    fraction: number;
}

export interface SeparationOptions {
    /**
     * Fired per segment as its inference completes. Final stems are ready
     * only when the returned promise resolves.
     */
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

/** Throws if an ONNX output shape differs from the dims the pipeline expects. */
function assertShape(name: string, actual: number[], expected: number[]): void {
    if (actual.length !== expected.length || actual.some((d, i) => d !== expected[i])) {
        throw new Error(
            `Unexpected ONNX output shape for ${name}: got [${actual.join(', ')}], ` +
            `expected [${expected.join(', ')}]`
        );
    }
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
    config: ModelConfig,
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
    const SEGMENT_SAMPLES = config.segmentSamples;
    const modelSources = config.modelSources;
    const { numBins: expectBins, numFrames: expectFrames } = specDims(config);

    const left = audioBuffer.getChannelData(0);
    // Mono is duplicated to both channels; for >2 channels we take the first
    // two. This matches the Python pipeline's convert_audio_channels, which
    // returns wav[:channels] when the source has more channels than the model.
    const right = audioBuffer.numberOfChannels > 1
        ? audioBuffer.getChannelData(1)
        : left;

    // Track-level normalization (Python unblend/api.py _normalize): mean/std
    // are scalars over the channel-mean reference signal, std is unbiased
    // (divide by N-1). Denormalized after separation with the same
    // ``1e-5 + std`` factor. RoFormer checkpoints are trained on raw audio,
    // so the whole normalize/denormalize pair is skipped for them (matching
    // the Python Separator's external_normalization gate).
    let mean = 0;
    let norm = 1;
    let denormStd = 0;
    if (config.normalizeInput) {
        let refSum = 0;
        for (let i = 0; i < numSamples; i++) {
            refSum += (left[i] + right[i]) / 2;
        }
        mean = refSum / numSamples;
        let refVar = 0;
        for (let i = 0; i < numSamples; i++) {
            const d = (left[i] + right[i]) / 2 - mean;
            refVar += d * d;
        }
        // Deliberate guard: the reference's unbiased std is NaN for 1-sample input.
        const std = numSamples > 1 ? Math.sqrt(refVar / (numSamples - 1)) : 0;
        norm = 1 / (1e-5 + std);
        denormStd = std;
    }

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

    // Mirror the Python chunking exactly (unblend/apply.py): segments start at
    // every multiple of the stride, the window for a short final chunk is
    // centered on it (real left context from the padded track, zeros past the
    // end), and chunks are blended with the triangular weight normalized by
    // the per-sample weight sum.
    const STEP = Math.floor(SEGMENT_SAMPLES * (1 - SEGMENT_OVERLAP));
    const weight = createSplitWeight(SEGMENT_SAMPLES);

    // Final accumulators across shift rounds; divided by ``shifts`` and
    // denormalized at the end. These are the returned stem buffers.
    const outputs: Record<string, Float32Array> = {};
    for (const source of modelSources) {
        outputs[source] = new Float32Array(numSamples * numChannels);
    }

    // Keep at most one segment of unfinished overlap-add data per source (and
    // only the shifted-view length for shorter clips). Samples are normalized
    // into final outputs as soon as no future segment can touch them, avoiding
    // a second set of full-track source buffers.
    const overlapBufferSamples = Math.min(
        SEGMENT_SAMPLES,
        numSamples + MAX_SHIFT,
    );
    const roundAccumulator = new StreamingOverlapAccumulator(
        outputs,
        modelSources,
        overlapBufferSamples,
        numChannels,
        STEP,
    );

    // Draw all offsets up front so totalSegs is known for progress reporting.
    // Python: random.randint(0, max_shift) — inclusive on both ends.
    const offsets: number[] = [];
    let totalSegs = 0;
    for (let r = 0; r < shifts; r++) {
        const offset = Math.floor(rand() * (MAX_SHIFT + 1));
        offsets.push(offset);
        totalSegs += Math.ceil((numSamples + MAX_SHIFT - offset) / STEP);
    }

    // Double-buffer so we can prepare the next segment while inference reads
    // the current one (HTDemucs only — RoFormer graphs take no audio input).
    const planarBuffers = config.hasTimeBranch
        ? [
            new Float32Array(SEGMENT_SAMPLES * numChannels),
            new Float32Array(SEGMENT_SAMPLES * numChannels),
        ]
        : null;
    let pendingPlanarIndex = 0;

    const startTime = performance.now();
    let totalInferenceMs = 0;
    let segsDone = 0;

    for (let r = 0; r < shifts; r++) {
        const viewOffset = offsets[r];
        const viewLength = numSamples + MAX_SHIFT - viewOffset;
        const numSegments = Math.ceil(viewLength / STEP);
        const trimStart = MAX_SHIFT - viewOffset;
        roundAccumulator.startRound(trimStart, viewLength);

        function segmentWindow(seg: number) {
            const segStart = seg * STEP;
            const segLength = Math.min(SEGMENT_SAMPLES, viewLength - segStart);
            const trimOffset = (SEGMENT_SAMPLES - segLength) >> 1;
            return { segStart, segLength, trimOffset, windowStart: segStart - trimOffset };
        }

        function accumulate(result: ISTFTResult) {
            const { chunks, segStart, segLength } = result;
            // Chunks arrive preweighted from the iSTFT worker. The streaming
            // accumulator preserves the old add-then-divide order exactly.
            roundAccumulator.add(chunks, segStart, segLength, weight);
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
        let pendingPlanar = planarBuffers
            ? preparePlanar(planarBuffers[pendingPlanarIndex], seg0.windowStart)
            : undefined;
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
                stftResult.real, stftResult.imag, specShape,
                currentPlanar, currentPlanar ? audioShape : undefined
            );

            if (seg + 1 < numSegments) {
                const next = segmentWindow(seg + 1);
                pendingStft = stft.process(prepareInterleaved(next.windowStart));
                pendingStft.catch(() => {});
                if (planarBuffers) {
                    pendingPlanarIndex = 1 - pendingPlanarIndex;
                    pendingPlanar = preparePlanar(
                        planarBuffers[pendingPlanarIndex], next.windowStart
                    );
                }
            }

            const results = await inferencePromise;
            totalInferenceMs += performance.now() - inferenceStart;

            // The iSTFT worker slices these buffers with subarray, which
            // clamps silently, so a model with unexpected dims would produce
            // wrong/zero audio. Fail loudly instead (like the ONNX worker's
            // dtype check).
            assertShape('out_spec', results.outSpecShape,
                [1, modelSources.length, numChannels, expectBins, expectFrames]);
            if (config.hasTimeBranch) {
                if (!results.outWave || !results.outWaveShape) {
                    throw new Error("Model produced no 'out_wave' output");
                }
                assertShape('out_wave', results.outWaveShape,
                    [1, modelSources.length, numChannels, SEGMENT_SAMPLES]);
            }

            if (prevIstftPromise) {
                accumulate(await prevIstftPromise);
            }

            // The result buffers were transferred from the ONNX worker and are
            // exclusively owned here, so hand them straight to the iSTFT worker.
            prevIstftPromise = istft.process({
                specReal: results.outSpecReal,
                specImag: results.outSpecImag,
                wave: config.hasTimeBranch ? results.outWave : undefined,
                numSources: modelSources.length,
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
        roundAccumulator.finishRound();
    }

    // Average the shift rounds and denormalize (Python: out * (1e-5 + std) + mean).
    // Without input normalization this reduces to the plain shift average.
    const denormScale = config.normalizeInput
        ? (1e-5 + denormStd) / shifts
        : 1 / shifts;
    for (const source of modelSources) {
        const out = outputs[source];
        for (let i = 0; i < numSamples * numChannels; i++) {
            out[i] = out[i] * denormScale + mean;
        }
    }

    // Single-mask models: the second stem is mixture - stem. All the steps
    // above are linear, so the full-track subtraction equals the Python
    // backend's per-chunk complement.
    if (config.complement) {
        const stem = outputs[config.complement.stem];
        const complement = new Float32Array(numSamples * numChannels);
        for (let i = 0; i < numSamples; i++) {
            complement[i * 2] = left[i] - stem[i * 2];
            complement[i * 2 + 1] = right[i] - stem[i * 2 + 1];
        }
        outputs[config.complement.name] = complement;
    }

    return {
        stems: outputs,
        wallMs: performance.now() - startTime,
        inferenceMs: totalInferenceMs,
        numSegments: totalSegs,
    };
}
