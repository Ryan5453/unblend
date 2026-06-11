/**
 * Shared STFT → ONNX → iSTFT pipeline. Pure: takes the worker clients and
 * source list it needs, no module-level state.
 */

import { SEGMENT_SAMPLES, SEGMENT_OVERLAP, createSplitWeight } from './constants';
import type { OnnxClient } from './onnx-client';
import type { STFTClient } from './stft-client';
import { type ISTFTClient, type ISTFTResult } from './istft-client';

export interface SeparationProgress {
    /** 1-based segment index that just finished. */
    segIdx: number;
    /** Total number of segments for this track. */
    totalSegs: number;
    /** Convenience: ``segIdx / totalSegs`` ∈ (0, 1]. */
    fraction: number;
}

export interface SeparationOptions {
    /** Fired after every segment completes (after iSTFT accumulation). */
    onProgress?: (progress: SeparationProgress) => void;
}

export interface SeparationResult {
    /** stem name → interleaved stereo Float32Array (L,R,L,R…) of length ``numSamples * 2``. */
    stems: Record<string, Float32Array>;
    /** Total wall time including STFT/iSTFT orchestration. */
    wallMs: number;
    /** Sum of ONNX inference times across all segments. */
    inferenceMs: number;
    /** Number of segments processed. */
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
    const { onnx, stft, istft } = pipeline;
    const numChannels = 2;
    const numSamples = audioBuffer.length;

    const audio = new Float32Array(numSamples * numChannels);
    const left = audioBuffer.getChannelData(0);
    // Mono is duplicated to both channels; for >2 channels we take the first
    // two. This matches the Python pipeline's convert_audio_channels, which
    // returns wav[:channels] when the source has more channels than the model.
    const right = audioBuffer.numberOfChannels > 1
        ? audioBuffer.getChannelData(1)
        : left;
    for (let i = 0; i < numSamples; i++) {
        audio[i * 2] = left[i];
        audio[i * 2 + 1] = right[i];
    }

    // Mirror the Python chunking exactly (demucs/apply.py): segments start at
    // every multiple of the stride, the window for a short final chunk is
    // centered on it (real left context, zeros past the end), and chunks are
    // blended with the triangular weight normalized by the per-sample weight
    // sum.
    const STEP = Math.floor(SEGMENT_SAMPLES * (1 - SEGMENT_OVERLAP));
    const numSegments = Math.ceil(numSamples / STEP);

    const outputs: Record<string, Float32Array> = {};
    for (const source of sources) {
        outputs[source] = new Float32Array(numSamples * numChannels);
    }

    const weight = createSplitWeight();
    const sumWeight = new Float32Array(numSamples);

    function segmentWindow(seg: number) {
        const segStart = seg * STEP;
        const segLength = Math.min(SEGMENT_SAMPLES, numSamples - segStart);
        const trimOffset = (SEGMENT_SAMPLES - segLength) >> 1;
        return { segStart, segLength, trimOffset, windowStart: segStart - trimOffset };
    }

    // Double-buffer so we can prepare the next segment while inference reads the current one.
    const planarBuffers = [
        new Float32Array(SEGMENT_SAMPLES * numChannels),
        new Float32Array(SEGMENT_SAMPLES * numChannels),
    ];
    let pendingPlanarIndex = 0;

    function accumulate(result: ISTFTResult) {
        const { chunks, segStart, segLength } = result;
        for (let s = 0; s < sources.length; s++) {
            const chunk = chunks[s];
            for (let i = 0; i < segLength; i++) {
                const outIdx = (segStart + i) * numChannels;
                outputs[sources[s]][outIdx] += chunk[i * numChannels];
                outputs[sources[s]][outIdx + 1] += chunk[i * numChannels + 1];
            }
        }
        // Chunks arrive pre-weighted from the iSTFT worker; track the matching
        // weight sum so the final normalization divides it back out.
        for (let i = 0; i < segLength; i++) {
            sumWeight[segStart + i] += weight[i];
        }
    }

    // ``windowStart`` may be negative (track shorter than one segment) or run
    // past the end of the audio; out-of-range samples stay zero.
    function prepareInterleaved(windowStart: number): Float32Array {
        const interleaved = new Float32Array(SEGMENT_SAMPLES * numChannels);
        const from = Math.max(0, -windowStart);
        const to = Math.min(SEGMENT_SAMPLES, numSamples - windowStart);
        for (let i = from; i < to; i++) {
            const srcIdx = (windowStart + i) * numChannels;
            interleaved[i * 2] = audio[srcIdx];
            interleaved[i * 2 + 1] = audio[srcIdx + 1];
        }
        return interleaved;
    }

    function preparePlanar(buffer: Float32Array, windowStart: number): Float32Array {
        buffer.fill(0);
        const from = Math.max(0, -windowStart);
        const to = Math.min(SEGMENT_SAMPLES, numSamples - windowStart);
        for (let i = from; i < to; i++) {
            const srcIdx = (windowStart + i) * numChannels;
            buffer[i] = audio[srcIdx];
            buffer[SEGMENT_SAMPLES + i] = audio[srcIdx + 1];
        }
        return buffer;
    }

    const startTime = performance.now();
    let totalInferenceMs = 0;

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
        if (seg % 5 === 0) {
            await new Promise(resolve => requestAnimationFrame(resolve));
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

        prevIstftPromise = istft.process({
            specReal: new Float32Array(results.outSpecReal),
            specImag: new Float32Array(results.outSpecImag),
            wave: new Float32Array(results.outWave),
            numSources: sources.length,
            numChannels,
            numBins: stftResult.numBins,
            numFrames: stftResult.numFrames,
            segStart, segLength, trimOffset,
        });
        prevIstftPromise.catch(() => {});

        onProgress?.({
            segIdx: seg + 1,
            totalSegs: numSegments,
            fraction: (seg + 1) / numSegments,
        });
    }

    if (prevIstftPromise) {
        accumulate(await prevIstftPromise);
    }

    // Normalize by the accumulated weight sum (Python: out / sum_weight).
    // Every sample is covered by at least one chunk, so sumWeight > 0.
    for (const source of sources) {
        const out = outputs[source];
        for (let i = 0; i < numSamples; i++) {
            const w = sumWeight[i];
            out[i * 2] /= w;
            out[i * 2 + 1] /= w;
        }
    }

    return {
        stems: outputs,
        wallMs: performance.now() - startTime,
        inferenceMs: totalInferenceMs,
        numSegments,
    };
}
