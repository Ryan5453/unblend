/**
 * Shared STFT → ONNX → iSTFT pipeline. Pure: takes the worker clients and
 * source list it needs, no module-level state.
 */

import { SEGMENT_SAMPLES, SEGMENT_OVERLAP } from './constants';
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
    /** stem name → planar interleaved Float32Array of length ``numSamples * 2``. */
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
    const right = audioBuffer.numberOfChannels > 1
        ? audioBuffer.getChannelData(1)
        : left;
    for (let i = 0; i < numSamples; i++) {
        audio[i * 2] = left[i];
        audio[i * 2 + 1] = right[i];
    }

    const OVERLAP = Math.floor(SEGMENT_SAMPLES * SEGMENT_OVERLAP);
    const STEP = SEGMENT_SAMPLES - OVERLAP;
    const numSegments = Math.ceil((numSamples - OVERLAP) / STEP);

    const outputs: Record<string, Float32Array> = {};
    for (const source of sources) {
        outputs[source] = new Float32Array(numSamples * numChannels);
    }

    const fadeIn = new Float32Array(OVERLAP);
    const fadeOut = new Float32Array(OVERLAP);
    for (let i = 0; i < OVERLAP; i++) {
        fadeIn[i] = i / OVERLAP;
        fadeOut[i] = 1 - i / OVERLAP;
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
                const globalIdx = segStart + i;
                if (globalIdx >= numSamples) continue;
                const outIdx = globalIdx * numChannels;
                outputs[sources[s]][outIdx] += chunk[i * numChannels];
                outputs[sources[s]][outIdx + 1] += chunk[i * numChannels + 1];
            }
        }
    }

    function prepareInterleaved(segStart: number, segLength: number): Float32Array {
        const interleaved = new Float32Array(SEGMENT_SAMPLES * numChannels);
        for (let i = 0; i < segLength; i++) {
            const srcIdx = (segStart + i) * numChannels;
            interleaved[i * 2] = audio[srcIdx];
            interleaved[i * 2 + 1] = audio[srcIdx + 1];
        }
        return interleaved;
    }

    function preparePlanar(buffer: Float32Array, segStart: number, segLength: number): Float32Array {
        buffer.fill(0);
        for (let i = 0; i < segLength; i++) {
            const srcIdx = (segStart + i) * numChannels;
            buffer[i] = audio[srcIdx];
            buffer[SEGMENT_SAMPLES + i] = audio[srcIdx + 1];
        }
        return buffer;
    }

    const startTime = performance.now();
    let totalInferenceMs = 0;

    // ``.catch(() => {})`` silences unhandled-rejection warnings when an
    // exception aborts the loop before we await these promises.
    const seg0End = Math.min(SEGMENT_SAMPLES, numSamples);
    let pendingStft = stft.process(prepareInterleaved(0, seg0End));
    pendingStft.catch(() => {});
    let pendingPlanar = preparePlanar(planarBuffers[pendingPlanarIndex], 0, seg0End);
    let prevIstftPromise: Promise<ISTFTResult> | null = null;

    for (let seg = 0; seg < numSegments; seg++) {
        const segStart = seg * STEP;
        const segEnd = Math.min(segStart + SEGMENT_SAMPLES, numSamples);
        const segLength = segEnd - segStart;

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
            const nextSegStart = (seg + 1) * STEP;
            const nextSegEnd = Math.min(nextSegStart + SEGMENT_SAMPLES, numSamples);
            const nextSegLength = nextSegEnd - nextSegStart;
            pendingStft = stft.process(prepareInterleaved(nextSegStart, nextSegLength));
            pendingStft.catch(() => {});
            pendingPlanarIndex = 1 - pendingPlanarIndex;
            pendingPlanar = preparePlanar(
                planarBuffers[pendingPlanarIndex], nextSegStart, nextSegLength
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
            segStart, segLength, seg,
            numSegments, numSamples,
            fadeIn, fadeOut,
            overlap: OVERLAP,
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

    return {
        stems: outputs,
        wallMs: performance.now() - startTime,
        inferenceMs: totalInferenceMs,
        numSegments,
    };
}
