import assert from 'node:assert/strict';
import test from 'node:test';

import { checkShaderF16, withGpuErrorScopes } from '../dist/webgpu-checks.js';

function device(features = []) {
    return { features: new Set(features) };
}

/**
 * A GPUDevice stand-in whose error scopes behave like the spec's: each scope
 * keeps the first error matching its filter, and an error goes to the
 * innermost matching scope.
 */
function scopedDevice() {
    const stack = [];
    return {
        stack,
        pushErrorScope(filter) {
            stack.push({ filter, error: null });
        },
        popErrorScope() {
            const scope = stack.pop();
            return scope ? Promise.resolve(scope.error) : Promise.reject(new Error('empty stack'));
        },
        raise(filter, message) {
            for (let i = stack.length - 1; i >= 0; i--) {
                if (stack[i].filter === filter) {
                    stack[i].error ??= { message };
                    return;
                }
            }
        },
    };
}

test('fp16-compute files need shader-f16; everything else passes', () => {
    assert.throws(
        () => checkShaderF16({ compute_precision: 'fp16' }, device()),
        /needs the WebGPU 'shader-f16' feature.*precision: 'fp32'/,
    );
    checkShaderF16({ compute_precision: 'fp16' }, device(['shader-f16']));
    // fp16 storage with fp32 compute (HTDemucs, SCNet) and older exports.
    checkShaderF16({ compute_precision: 'fp32' }, device());
    checkShaderF16({}, device());
});

test('a run that raised no GPU error resolves with its value', async () => {
    const gpu = scopedDevice();
    assert.equal(await withGpuErrorScopes(gpu, async () => 'outputs'), 'outputs');
    assert.equal(gpu.stack.length, 0);
});

test('a GPU error fails the run after it resolved', async () => {
    const gpu = scopedDevice();
    let ran = false;
    await assert.rejects(
        withGpuErrorScopes(gpu, async () => {
            gpu.raise('validation', "'f16' type used without 'f16' extension enabled");
            ran = true;
            return 'garbage';
        }),
        /^Error: WebGPU validation error: 'f16' type used/,
    );
    assert.ok(ran);
    assert.equal(gpu.stack.length, 0);
});

test('an out-of-memory error is reported ahead of the validation errors it causes', async () => {
    const gpu = scopedDevice();
    await assert.rejects(
        withGpuErrorScopes(gpu, async () => {
            gpu.raise('out-of-memory', 'Out of memory');
            gpu.raise('validation', 'Buffer is invalid due to a previous error');
        }),
        /^Error: WebGPU out-of-memory error: Out of memory$/,
    );
});

test("the run's own error wins, and every scope is still popped", async () => {
    const gpu = scopedDevice();
    await assert.rejects(
        withGpuErrorScopes(gpu, async () => {
            gpu.raise('validation', 'shader failed');
            throw new Error('Failed to download data from buffer');
        }),
        /Failed to download data/,
    );
    assert.equal(gpu.stack.length, 0);
});

test('a rejected pop (a lost device) does not mask the run', async () => {
    const gpu = scopedDevice();
    gpu.popErrorScope = () => Promise.reject(new Error('device lost'));
    assert.equal(await withGpuErrorScopes(gpu, async () => 'outputs'), 'outputs');
});

test('without a device the run is called directly', async () => {
    assert.equal(await withGpuErrorScopes(null, async () => 'outputs'), 'outputs');
});
