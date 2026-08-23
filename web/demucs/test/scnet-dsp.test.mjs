/**
 * The SCNet STFT differs from the RoFormer one in two ways that are easy to
 * get silently wrong: it is unwindowed (the reference passes no `window` to
 * torch.stft) and it is sqrt(N)-normalized. Both are handled by the shared
 * roformer DSP via config flags, so this pins them against a fixture produced
 * by the Python implementation.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import { createDSP } from '../dist/audio-processor.js';
import { MODEL_CONFIGS, dspConfig, specDims } from '../dist/constants.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(
    readFileSync(join(here, 'scnet-stft-fixture.json'), 'utf8'),
);

test('scnet DSP matches the Python STFT (unwindowed, normalized)', () => {
    const dsp = createDSP({
        family: 'scnet',
        nfft: fixture.nfft,
        hopLength: fixture.hopLength,
        segmentSamples: fixture.segmentSamples,
        window: 'rectangular',
        normalized: true,
    });

    const { real, imag, numBins, numFrames } = dsp.computeSTFT(
        Float32Array.from(fixture.audio),
    );
    assert.equal(numBins, fixture.numBins);
    assert.equal(numFrames, fixture.numFrames);

    let maxDiff = 0;
    for (let i = 0; i < fixture.real.length; i++) {
        maxDiff = Math.max(
            maxDiff,
            Math.abs(real[i] - fixture.real[i]),
            Math.abs(imag[i] - fixture.imag[i]),
        );
    }
    assert.ok(maxDiff < 1e-3, `max |diff| = ${maxDiff}`);
});

test('scnet DSP round-trips through its own iSTFT', () => {
    const dsp = createDSP({
        family: 'scnet',
        nfft: fixture.nfft,
        hopLength: fixture.hopLength,
        segmentSamples: fixture.segmentSamples,
        window: 'rectangular',
        normalized: true,
    });

    const audio = Float32Array.from(fixture.audio);
    const { real, imag, numBins, numFrames } = dsp.computeSTFT(audio);
    const out = dsp.computeISTFT(real, imag, 2, numBins, numFrames);

    // Interior samples only: the centred padding makes the first and last
    // frames' overlap-add incomplete at the very edges.
    const seg = fixture.segmentSamples;
    let maxDiff = 0;
    for (let c = 0; c < 2; c++) {
        for (let n = fixture.nfft; n < seg - fixture.nfft; n++) {
            maxDiff = Math.max(
                maxDiff,
                Math.abs(out[c * seg + n] - audio[n * 2 + c]),
            );
        }
    }
    assert.ok(maxDiff < 1e-3, `round-trip max |diff| = ${maxDiff}`);
});

test('scnet registry preserves Python normalization and internal zero padding', () => {
    for (const [name, window, normalizeInput, webgpuRequired] of [
        ['scnet_small', 'hann', false, false],
        ['scnet_xl_wide_v5', 'rectangular', false, true],
    ]) {
        const config = MODEL_CONFIGS[name];
        assert.equal(config.normalizeInput, normalizeInput);
        assert.equal(Boolean(config.webgpuRequired), webgpuRequired);
        assert.equal(config.segmentSamples, 485100);
        assert.equal(config.modelInputSamples, 486400);
        assert.deepEqual(specDims(config), { numBins: 2049, numFrames: 476 });
        assert.deepEqual(dspConfig(config), {
            family: 'scnet',
            nfft: 4096,
            hopLength: 1024,
            segmentSamples: 486400,
            chunkSamples: 485100,
            window,
            normalized: true,
        });
    }
});
