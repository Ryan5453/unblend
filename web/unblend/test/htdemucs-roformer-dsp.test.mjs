/**
 * Parity of the HTDemucs and RoFormer segment STFT/iSTFT against the Python
 * package. The fixtures are small-geometry runs of
 * ``unblend.onnx.compute_stft_for_export`` + ``HTDemucs._ispec`` (HTDemucs)
 * and ``compute_roformer_stft_for_export`` + ``torch.istft`` (RoFormer). The
 * iSTFT input is the fixture STFT times a deterministic real mask, so it is
 * not a consistent spectrogram: this pins the window-envelope division and
 * trims, not just perfect reconstruction.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

import { createDSP } from '../dist/audio-processor.js';

const here = dirname(fileURLToPath(import.meta.url));
const load = name => JSON.parse(readFileSync(join(here, name), 'utf8'));

/** The mask the fixture script applied: 0.5 + 0.5 cos(0.3 b + 0.7 f + c). */
function masked(values, numBins, numFrames) {
    const out = new Float32Array(values.length);
    for (let c = 0; c < 2; c++) {
        for (let b = 0; b < numBins; b++) {
            for (let f = 0; f < numFrames; f++) {
                const i = (c * numBins + b) * numFrames + f;
                out[i] = values[i] * (0.5 + 0.5 * Math.cos(0.3 * b + 0.7 * f + c));
            }
        }
    }
    return out;
}

function maxAbsDiff(actual, expected) {
    assert.equal(actual.length, expected.length);
    let max = 0;
    for (let i = 0; i < expected.length; i++) {
        max = Math.max(max, Math.abs(actual[i] - expected[i]));
    }
    return max;
}

for (const [family, fixtureName] of [
    ['htdemucs', 'htdemucs-stft-fixture.json'],
    ['roformer', 'roformer-stft-fixture.json'],
]) {
    const fixture = load(fixtureName);
    const makeDSP = () => createDSP({
        family,
        nfft: fixture.nfft,
        hopLength: fixture.hopLength,
        segmentSamples: fixture.segmentSamples,
    });

    test(`${family} DSP STFT matches the Python STFT`, () => {
        const { real, imag, numBins, numFrames } = makeDSP().computeSTFT(
            Float32Array.from(fixture.audio),
        );
        assert.equal(numBins, fixture.numBins);
        assert.equal(numFrames, fixture.numFrames);
        const diff = Math.max(
            maxAbsDiff(real, fixture.real),
            maxAbsDiff(imag, fixture.imag),
        );
        assert.ok(diff < 1e-4, `max |diff| = ${diff}`);
    });

    test(`${family} DSP iSTFT matches the Python iSTFT on a masked spectrum`, () => {
        const { numBins, numFrames } = fixture;
        const out = makeDSP().computeISTFT(
            masked(fixture.real, numBins, numFrames),
            masked(fixture.imag, numBins, numFrames),
            2, numBins, numFrames,
        );
        const diff = maxAbsDiff(out, fixture.istft);
        assert.ok(diff < 1e-4, `max |diff| = ${diff}`);
    });

    // HTDemucs drops the Nyquist bin and trims its edge frames, so only the
    // interior of a band-limited signal survives its round trip exactly (as
    // in Python); check that region for both families.
    test(`${family} DSP round-trips a band-limited signal through its own iSTFT`, () => {
        const dsp = makeDSP();
        const seg = fixture.segmentSamples;
        const audio = new Float32Array(seg * 2);
        for (let n = 0; n < seg; n++) {
            audio[n * 2] = Math.sin(0.3 * n) + 0.5 * Math.cos(0.9 * n + 0.2);
            audio[n * 2 + 1] = 0.7 * Math.sin(0.05 * n + 1) - 0.2 * Math.sin(1.3 * n);
        }
        const { real, imag, numBins, numFrames } = dsp.computeSTFT(audio);
        const out = dsp.computeISTFT(real, imag, 2, numBins, numFrames);
        let diff = 0;
        for (let c = 0; c < 2; c++) {
            for (let n = fixture.nfft; n < seg - fixture.nfft; n++) {
                diff = Math.max(diff, Math.abs(out[c * seg + n] - audio[n * 2 + c]));
            }
        }
        assert.ok(diff < 1e-4, `round-trip max |diff| = ${diff}`);
    });
}
