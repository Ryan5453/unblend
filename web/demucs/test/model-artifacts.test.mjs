import assert from 'node:assert/strict';
import test from 'node:test';

import { MODEL_ARTIFACTS } from '../dist/model-artifacts.js';

const MODELS = [
    'htdemucs',
    'htdemucs_6s',
    'bs_roformer_sw',
    'melband_roformer_kim',
];
const PRECISIONS = ['fp32', 'fp16'];
const REVISION = 'eda32466a76dc81c5e66af6577dbc20fb219e959';

test('model artifact registry is complete, immutable, and well-formed', () => {
    assert.deepEqual(Object.keys(MODEL_ARTIFACTS).sort(), [...MODELS].sort());

    const urls = new Set();
    for (const model of MODELS) {
        assert.deepEqual(Object.keys(MODEL_ARTIFACTS[model]).sort(), [...PRECISIONS].sort());
        for (const precision of PRECISIONS) {
            const artifact = MODEL_ARTIFACTS[model][precision];
            assert.match(artifact.url, new RegExp(`/resolve/${REVISION}/`));
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
