/**
 * Chunking / overlap-add parity with Python's ``apply_model_multi``. A fake
 * HTDemucs-shaped model scales its time-domain input by a function of the
 * sample's position inside the model window (and zeroes the spectrogram
 * branch), so the output depends on every chunk start, the centering of the
 * short final chunk, the triangular blend weights and the shift offsets. The
 * expected output comes from the Python package with the same fake model
 * (scripts/gen-dsp-fixtures.py).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { Worker as NodeWorker } from 'node:worker_threads';

import { dspConfig } from '../dist/constants.js';
import { runPipeline } from '../dist/pipeline.js';
import { STFTClient } from '../dist/stft-client.js';
import { ISTFTClient } from '../dist/istft-client.js';

const fixture = JSON.parse(
    fs.readFileSync(new URL('./chunking-fixture.json', import.meta.url), 'utf8'),
);

// Runs a dedicated-worker module on a worker thread (see pipeline-e2e.test.mjs).
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

const SEG = fixture.segmentSamples;
const N = fixture.numSamples;
const MAX_SHIFT = 22050;
const SOURCES = ['a', 'b'];
const config = {
    family: 'htdemucs',
    nfft: 32,
    hopLength: 8,
    segmentSamples: SEG,
    modelSources: SOURCES,
    sources: SOURCES,
    normalizeInput: false,
    hasTimeBranch: true,
    license: 'test',
};

/** Python PositionModel: stem a = x * g, stem b = x * g², g = 1 + i / L. */
const positionModel = {
    async runInference(specReal, specImag, specShape, audio, audioShape) {
        const [, C, F, T] = specShape;
        const L = audioShape[2];
        const wave = new Float32Array(SOURCES.length * C * L);
        for (let c = 0; c < C; c++) {
            for (let i = 0; i < L; i++) {
                const g = 1 + i / L;
                const x = audio[c * L + i];
                wave[(0 * C + c) * L + i] = x * g;
                wave[(1 * C + c) * L + i] = x * g * g;
            }
        }
        const specSize = SOURCES.length * C * F * T;
        return {
            outSpecReal: new Float32Array(specSize),
            outSpecImag: new Float32Array(specSize),
            outSpecShape: [1, SOURCES.length, C, F, T],
            outWave: wave,
            outWaveShape: [1, SOURCES.length, C, L],
        };
    },
};

const left = Float32Array.from(fixture.mix[0]);
const right = Float32Array.from(fixture.mix[1]);
const audio = {
    sampleRate: 44100,
    length: N,
    numberOfChannels: 2,
    getChannelData: c => (c === 0 ? left : right),
};

/** Run with ``Math.random`` stubbed so the pipeline draws exactly ``offsets``. */
async function runWithOffsets(pipeline, shifts, offsets) {
    const original = Math.random;
    let k = 0;
    Math.random = () => (offsets[k++] + 0.5) / (MAX_SHIFT + 1);
    try {
        return await runPipeline(pipeline, audio, config, { shifts });
    } finally {
        Math.random = original;
    }
}

const cases = [
    ['shifts=0 (single unshifted pass, short final chunk)', 'shifts0', 0, []],
    [`shifts=3 at offsets ${fixture.shiftOffsets.join(', ')}`, 'shifts3',
        fixture.shiftOffsets.length, fixture.shiftOffsets],
];

for (const [title, key, shifts, offsets] of cases) {
    test(`runPipeline chunking matches Python apply_model_multi: ${title}`, async () => {
        const stft = new STFTClient();
        const istft = new ISTFTClient();
        try {
            await Promise.all([
                stft.configure(dspConfig(config)),
                istft.configure(dspConfig(config)),
            ]);
            const result = await runWithOffsets({ onnx: positionModel, stft, istft }, shifts, offsets);
            const expected = fixture.outputs[key];
            SOURCES.forEach((source, s) => {
                const stem = result.stems[source];
                assert.equal(stem.length, N * 2);
                for (let c = 0; c < 2; c++) {
                    const ref = expected[s * 2 + c];
                    let diff = 0;
                    for (let i = 0; i < N; i++) {
                        diff = Math.max(diff, Math.abs(stem[i * 2 + c] - ref[i]));
                    }
                    assert.ok(diff < 1e-5, `${source}[${c}]: max |JS - Python| = ${diff}`);
                }
            });
        } finally {
            stft.terminate();
            istft.terminate();
        }
    });
}
