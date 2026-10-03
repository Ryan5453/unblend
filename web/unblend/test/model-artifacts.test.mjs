import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { MODEL_CONFIGS } from '../dist/constants.js';
import { MODEL_ARTIFACTS } from '../dist/model-artifacts.js';

const MODELS = [
    'htdemucs',
    'htdemucs_6s',
    'bs_roformer_sw',
    'melband_roformer_kim',
    'scnet_small',
    'scnet_xl_wide_v5',
];
const PRECISIONS = ['fp32', 'fp16'];
const REVISIONS = {
    htdemucs: '30b749fd691312b1e4b3f8fe79df0b51cde836cd',
    htdemucs_6s: '30b749fd691312b1e4b3f8fe79df0b51cde836cd',
    bs_roformer_sw: '30b749fd691312b1e4b3f8fe79df0b51cde836cd',
    melband_roformer_kim: '30b749fd691312b1e4b3f8fe79df0b51cde836cd',
    scnet_small: '30b749fd691312b1e4b3f8fe79df0b51cde836cd',
    scnet_xl_wide_v5: '30b749fd691312b1e4b3f8fe79df0b51cde836cd',
};

test('browser SCNet catalog matches the Python registry', () => {
    // The Python registry is YAML; scan it for model entries whose
    // `architecture` is an SCNet variant rather than pulling in a YAML parser.
    // The registry declares architecture only -- the backend is derived from it
    // in repo.py -- so matching on `backend:` finds nothing.
    const yaml = readFileSync(new URL('../../../unblend/metadata.yaml', import.meta.url), 'utf8');
    let current = null;
    const pythonModels = [];
    for (const line of yaml.split('\n')) {
        const name = line.match(/^  ([A-Za-z0-9_]+):\s*$/);
        if (name) current = name[1];
        if (current && /^    architecture: scnet[A-Za-z0-9_]*\s*$/.test(line)) {
            pythonModels.push(current);
        }
    }
    pythonModels.sort();
    const browserModels = Object.entries(MODEL_CONFIGS)
        .filter(([, info]) => info.family === 'scnet')
        .map(([name]) => name)
        .sort();

    // A scan that matches nothing yields an empty list, which would compare
    // equal to an equally-empty browser side and pass -- so the drift check
    // would silently stop working. Fail loudly if the registry format moves.
    assert.ok(
        pythonModels.length > 0,
        'found no SCNet models in metadata.yaml; the registry format changed',
    );

    assert.deepEqual(browserModels, pythonModels);
});

test('browser model settings match the Python registry', () => {
    // Line scan of each entry's top-level fields, as above (no YAML parser).
    const yaml = readFileSync(new URL('../../../unblend/metadata.yaml', import.meta.url), 'utf8');
    const entries = {};
    let current = null;
    for (const line of yaml.split('\n')) {
        const name = line.match(/^  ([A-Za-z0-9_]+):\s*$/);
        if (name) {
            current = name[1];
            entries[current] = {};
            continue;
        }
        const field = line.match(/^    (sources|license|segment_samples): (.+?)\s*$/);
        if (current && field) entries[current][field[1]] = field[2];
    }
    for (const [model, config] of Object.entries(MODEL_CONFIGS)) {
        const entry = entries[model];
        assert.ok(entry, `${model} is not in metadata.yaml`);
        assert.equal(`[${config.sources.join(', ')}]`, entry.sources, `${model} sources`);
        assert.equal(config.license, entry.license, `${model} license`);
        // HTDemucs declares its segment in seconds inside its config.
        if (entry.segment_samples !== undefined) {
            assert.equal(config.segmentSamples, Number(entry.segment_samples), `${model} segment`);
        }
    }
});

test('model artifact registry is complete, immutable, and well-formed', () => {
    assert.deepEqual(Object.keys(MODEL_ARTIFACTS).sort(), [...MODELS].sort());

    const urls = new Set();
    for (const model of MODELS) {
        assert.deepEqual(Object.keys(MODEL_ARTIFACTS[model]).sort(), [...PRECISIONS].sort());
        for (const precision of PRECISIONS) {
            const artifact = MODEL_ARTIFACTS[model][precision];
            assert.match(artifact.url, new RegExp(`/resolve/${REVISIONS[model]}/`));
            assert.ok(!artifact.url.includes('/resolve/main/'));
            assert.ok(artifact.url.endsWith(`/${model}_${precision}.onnx`));
            assert.match(artifact.sha256, /^[0-9a-f]{64}$/);
            assert.ok(Number.isSafeInteger(artifact.sizeBytes));
            assert.ok(artifact.sizeBytes > 0);
            assert.ok(!urls.has(artifact.url));
            urls.add(artifact.url);
        }
    }
});
