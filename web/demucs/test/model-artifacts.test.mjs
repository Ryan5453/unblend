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
    htdemucs: 'eda32466a76dc81c5e66af6577dbc20fb219e959',
    htdemucs_6s: 'eda32466a76dc81c5e66af6577dbc20fb219e959',
    bs_roformer_sw: 'a80a71b41face40edc91178c07edfedeca4cbb19',
    melband_roformer_kim: 'a80a71b41face40edc91178c07edfedeca4cbb19',
    scnet_small: 'ac4b06164d974e1242bd9fc7585305e5ea022d0f',
    scnet_xl_wide_v5: '396e6583cea8e5104f35c05d87cf60883794a58e',
};

test('browser SCNet catalog matches the Python registry', () => {
    // The Python registry is YAML; scan it for model entries declaring
    // `backend: scnet` rather than pulling in a YAML parser.
    const yaml = readFileSync(new URL('../../../unblend/metadata.yaml', import.meta.url), 'utf8');
    let current = null;
    const pythonModels = [];
    for (const line of yaml.split('\n')) {
        const name = line.match(/^  ([A-Za-z0-9_]+):\s*$/);
        if (name) current = name[1];
        if (current && /^    backend: scnet\s*$/.test(line)) pythonModels.push(current);
    }
    pythonModels.sort();
    const browserModels = Object.entries(MODEL_CONFIGS)
        .filter(([, info]) => info.family === 'scnet')
        .map(([name]) => name)
        .sort();

    assert.deepEqual(browserModels, pythonModels);
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
