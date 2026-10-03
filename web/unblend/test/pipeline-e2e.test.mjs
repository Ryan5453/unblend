/**
 * End-to-end ``runPipeline`` through the real STFT/iSTFT workers (run on
 * Node worker threads behind a Web Worker shim) with a fake ONNX model that
 * splits its input spectrogram (and, for HTDemucs, the time-domain input)
 * into fixed fractions per stem. Every step after the model is linear, so
 * each stem must reconstruct its fraction of the mixture for every family,
 * across segment boundaries, the overlap-add, shifts and normalization.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { Worker as NodeWorker } from 'node:worker_threads';

import { MODEL_CONFIGS, SEGMENT_OVERLAP, dspConfig } from '../dist/constants.js';
import { runPipeline, validateSeparationOptions } from '../dist/pipeline.js';
import { STFTClient } from '../dist/stft-client.js';
import { ISTFTClient } from '../dist/istft-client.js';

// Runs a dedicated-worker module on a worker thread: ``self`` is the global,
// messages are queued until the module has installed ``self.onmessage``.
const SHIM = `
const { parentPort, workerData } = require('node:worker_threads');
globalThis.self = globalThis;
globalThis.postMessage = (message, transfer) => parentPort.postMessage(
    message, Array.isArray(transfer) ? transfer : transfer?.transfer,
);
const queued = [];
let ready = false;
parentPort.on('message', data => ready ? globalThis.onmessage({ data }) : queued.push(data));
import(workerData).then(() => {
    ready = true;
    for (const data of queued.splice(0)) globalThis.onmessage({ data });
});
`;

class ThreadWorker {
    onmessage = null;
    onerror = null;
    onmessageerror = null;

    constructor(url) {
        this.thread = new NodeWorker(SHIM, { eval: true, workerData: String(url) });
        this.thread.on('message', data => this.onmessage?.({ data }));
        this.thread.on('error', error => this.onerror?.(error));
    }

    postMessage(message, transfer = []) {
        this.thread.postMessage(message, transfer);
    }

    terminate() {
        void this.thread.terminate();
    }
}

const originalWorker = globalThis.Worker;
test.before(() => {
    globalThis.Worker = ThreadWorker;
});
test.after(() => {
    if (originalWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = originalWorker;
});

/** Stem s gets spectrogram fraction ``spec[s]`` and time fraction ``wave[s]``. */
function fractions(config) {
    const n = config.modelSources.length;
    const total = (n * (n + 1)) / 2;
    const share = config.modelSources.map((_, s) => (s + 1) / total);
    if (config.complement) share[0] = 0.3;
    return {
        spec: share.map(w => (config.hasTimeBranch ? w * 0.75 : w)),
        wave: share.map(w => (config.hasTimeBranch ? w * 0.25 : 0)),
        total: share,
    };
}

function fakeModel(config) {
    const { spec, wave } = fractions(config);
    const S = config.modelSources.length;
    const scaled = (input, weights) => {
        const out = new Float32Array(input.length * S);
        weights.forEach((w, s) => {
            for (let i = 0; i < input.length; i++) out[s * input.length + i] = input[i] * w;
        });
        return out;
    };
    return {
        calls: 0,
        async runInference(specReal, specImag, specShape, audio, audioShape) {
            this.calls++;
            const [, C, F, T] = specShape;
            return {
                outSpecReal: scaled(specReal, spec),
                outSpecImag: scaled(specImag, spec),
                outSpecShape: [1, S, C, F, T],
                outWave: audio ? scaled(audio, wave) : undefined,
                outWaveShape: audio ? [1, S, ...audioShape.slice(1)] : undefined,
            };
        },
    };
}

/** Band-limited stereo test signal with a DC offset (exercises denormalization). */
function testAudio(numSamples) {
    const left = new Float32Array(numSamples);
    const right = new Float32Array(numSamples);
    for (let n = 0; n < numSamples; n++) {
        left[n] = 0.1 + 0.4 * Math.sin(n * 0.013) + 0.2 * Math.sin(n * 0.31 + 1);
        right[n] = -0.05 + 0.3 * Math.cos(n * 0.021) + 0.1 * Math.sin(n * 0.7);
    }
    return {
        sampleRate: 44100,
        length: numSamples,
        numberOfChannels: 2,
        getChannelData: c => (c === 0 ? left : right),
    };
}

for (const [model, config] of Object.entries(MODEL_CONFIGS)) {
    test(`${model}: runPipeline reconstructs per-stem fractions of the mixture`, async () => {
        const stft = new STFTClient();
        const istft = new ISTFTClient();
        try {
            await Promise.all([
                stft.configure(dspConfig(config)),
                istft.configure(dspConfig(config)),
            ]);
            // Long enough for several segments per round.
            const numSamples = Math.round(config.segmentSamples * 1.6);
            const audio = testAudio(numSamples);
            const onnx = fakeModel(config);
            const result = await runPipeline(
                { onnx, stft, istft }, audio, config, { shifts: 2, seed: 7 },
            );
            assert.ok(onnx.calls >= 4, `only ${onnx.calls} segments ran`);
            assert.deepEqual(Object.keys(result.stems).sort(), [...config.sources].sort());

            const left = audio.getChannelData(0);
            const right = audio.getChannelData(1);
            // Denormalization adds the track mean back to every model stem.
            let mean = 0;
            if (config.normalizeInput) {
                for (let i = 0; i < numSamples; i++) mean += (left[i] + right[i]) / 2;
                mean /= numSamples;
            }
            const { total } = fractions(config);
            const expected = {};
            config.modelSources.forEach((name, s) => {
                expected[name] = x => total[s] * (x - mean) + mean;
            });
            if (config.complement) {
                const w = total[0];
                expected[config.complement.name] = x => x - (w * (x - mean) + mean);
            }

            for (const [name, stem] of Object.entries(result.stems)) {
                assert.equal(stem.length, numSamples * 2);
                let diff = 0;
                for (let i = 0; i < numSamples; i++) {
                    diff = Math.max(
                        diff,
                        Math.abs(stem[i * 2] - expected[name](left[i])),
                        Math.abs(stem[i * 2 + 1] - expected[name](right[i])),
                    );
                }
                assert.ok(diff < 1e-3, `${name}: max |diff| = ${diff}`);
            }
        } finally {
            stft.terminate();
            istft.terminate();
        }
    });
}

test('validateSeparationOptions accepts shifts 0-20 and rejects the rest', () => {
    for (const shifts of [0, 1, 20, undefined]) {
        assert.doesNotThrow(() => validateSeparationOptions({ shifts }));
    }
    for (const shifts of [-1, 21, 1.5, NaN]) {
        assert.throws(() => validateSeparationOptions({ shifts }), /between 0 and 20/);
    }
});

test('shifts=0 runs one unshifted, deterministic pass', async () => {
    const config = MODEL_CONFIGS.htdemucs;
    const stft = new STFTClient();
    const istft = new ISTFTClient();
    try {
        await Promise.all([
            stft.configure(dspConfig(config)),
            istft.configure(dspConfig(config)),
        ]);
        const numSamples = Math.round(config.segmentSamples * 1.6);
        const audio = testAudio(numSamples);
        const step = Math.floor(config.segmentSamples * (1 - SEGMENT_OVERLAP));
        const run = async () => {
            const onnx = fakeModel(config);
            const segs = [];
            const result = await runPipeline(
                { onnx, stft, istft }, audio, config,
                { shifts: 0, onProgress: p => { if (p.stage === 'completed') segs.push(p.totalSegs); } },
            );
            return { onnx, result, totalSegs: segs.at(-1) };
        };
        const a = await run();
        const b = await run();
        // One pass over exactly the unshifted track (no MAX_SHIFT padding).
        const expectedSegs = Math.ceil(numSamples / step);
        assert.equal(a.onnx.calls, expectedSegs);
        assert.equal(a.totalSegs, expectedSegs);
        assert.equal(a.result.numSegments, expectedSegs);
        for (const name of Object.keys(a.result.stems)) {
            assert.deepEqual(a.result.stems[name], b.result.stems[name], `${name} differs between runs`);
        }
    } finally {
        stft.terminate();
        istft.terminate();
    }
});
