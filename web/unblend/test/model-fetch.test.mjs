import assert from 'node:assert/strict';
import test from 'node:test';

import { fetchModelBytes } from '../dist/model-fetch.js';

const originalFetch = globalThis.fetch;

test.afterEach(() => {
    globalThis.fetch = originalFetch;
});

test('known-length response resolves after the final byte without waiting for EOF', async () => {
    let cancelled = false;
    const body = new ReadableStream({
        start(controller) {
            controller.enqueue(new Uint8Array([1, 2]));
            controller.enqueue(new Uint8Array([3, 4]));
            // Deliberately never close: this reproduces a CDN connection that
            // has delivered Content-Length bytes but does not promptly emit EOF.
        },
        cancel() {
            cancelled = true;
        },
    });
    globalThis.fetch = async () => new Response(body, {
        headers: { 'Content-Length': '4' },
    });

    const progress = [];
    const bytes = await Promise.race([
        fetchModelBytes('https://example.test/model.onnx', (loaded, total) => {
            progress.push([loaded, total]);
        }),
        new Promise((_, reject) => {
            setTimeout(() => reject(new Error('fetch waited for EOF')), 100);
        }),
    ]);

    assert.deepEqual([...bytes], [1, 2, 3, 4]);
    assert.equal(cancelled, true);
    assert.deepEqual(progress.at(-1), [4, 4]);
});

test('known-length response still rejects if EOF arrives early', async () => {
    const body = new ReadableStream({
        start(controller) {
            controller.enqueue(new Uint8Array([1, 2, 3]));
            controller.close();
        },
    });
    globalThis.fetch = async () => new Response(body, {
        headers: { 'Content-Length': '4' },
    });

    await assert.rejects(
        fetchModelBytes('https://example.test/truncated.onnx', () => {}),
        /ended at 3 bytes; expected 4/,
    );
});
