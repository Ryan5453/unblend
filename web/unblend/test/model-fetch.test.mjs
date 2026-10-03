import assert from 'node:assert/strict';
import test from 'node:test';

import {
    MODEL_CACHE_NAME,
    clearModelCache,
    fetchModelBytes,
    loadModelBytes,
} from '../dist/model-fetch.js';

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

class FakeCache {
    entries = new Map();
    putError = null;

    async match(key) {
        const bytes = this.entries.get(String(key));
        return bytes && new Response(bytes, {
            headers: { 'Content-Length': String(bytes.byteLength) },
        });
    }

    async put(key, response) {
        if (this.putError) throw this.putError;
        this.entries.set(String(key), new Uint8Array(await response.arrayBuffer()));
    }

    async delete(key) {
        return this.entries.delete(typeof key === 'string' ? key : key.url);
    }

    async keys() {
        return [...this.entries.keys()].map(url => ({ url }));
    }
}

function installCaches(cache) {
    Object.defineProperty(globalThis, 'caches', {
        configurable: true,
        value: { open: async name => (assert.equal(name, MODEL_CACHE_NAME), cache) },
    });
}

function countingFetch(payload) {
    const calls = [];
    globalThis.fetch = async url => {
        calls.push(String(url));
        return new Response(payload, {
            headers: { 'Content-Length': String(payload.byteLength) },
        });
    };
    return calls;
}

const KEY = 'https://huggingface.co/x/resolve/rev2/model_fp16.onnx';

test.afterEach(() => {
    delete globalThis.caches;
});

test('complete downloads are cached by artifact URL and served from cache', async () => {
    const cache = new FakeCache();
    installCaches(cache);
    const calls = countingFetch(new Uint8Array([1, 2, 3]));
    const entry = { key: KEY, expectedBytes: 3 };

    const firstSources = new Set();
    const first = await loadModelBytes(KEY, (_l, _t, source) => firstSources.add(source), entry);
    assert.deepEqual([...firstSources], ['network']);
    assert.deepEqual([...first], [1, 2, 3]);
    assert.deepEqual([...cache.entries.keys()], [KEY]);

    const progress = [];
    const second = await loadModelBytes(KEY, (loaded, total, source) => progress.push([loaded, total, source]), entry);
    assert.deepEqual([...second], [1, 2, 3]);
    assert.equal(calls.length, 1);
    assert.deepEqual(progress.at(-1), [3, 3, 'cache']);
});

test('responses of the wrong size are neither cached nor served', async () => {
    const cache = new FakeCache();
    installCaches(cache);
    countingFetch(new Uint8Array([1, 2]));
    await assert.rejects(
        loadModelBytes(KEY, () => {}, { key: KEY, expectedBytes: 3 }),
        /Downloaded model is 2 bytes; expected 3/,
    );
    assert.equal(cache.entries.size, 0);

    cache.entries.set(KEY, new Uint8Array([9]));
    const calls = countingFetch(new Uint8Array([1, 2, 3]));
    const bytes = await loadModelBytes(KEY, () => {}, { key: KEY, expectedBytes: 3 });
    assert.deepEqual([...bytes], [1, 2, 3]);
    assert.equal(calls.length, 1);
    assert.deepEqual([...cache.entries.get(KEY)], [1, 2, 3]);
});

test('storing a new revision drops older revisions of the same file', async () => {
    const cache = new FakeCache();
    installCaches(cache);
    const stale = 'https://huggingface.co/x/resolve/rev1/model_fp16.onnx';
    const other = 'https://huggingface.co/x/resolve/rev1/other_fp16.onnx';
    cache.entries.set(stale, new Uint8Array([0]));
    cache.entries.set(other, new Uint8Array([0]));
    countingFetch(new Uint8Array([1, 2, 3]));

    await loadModelBytes(KEY, () => {}, { key: KEY, expectedBytes: 3 });
    assert.deepEqual([...cache.entries.keys()].sort(), [KEY, other].sort());
});

test('a stale revision is dropped before storing, so it cannot block the new one', async () => {
    // Room for one model only: with the old revision still stored, the new
    // one would not fit.
    class OneSlotCache extends FakeCache {
        async put(key, response) {
            if (this.entries.size >= 1) throw new DOMException('full', 'QuotaExceededError');
            return super.put(key, response);
        }
    }
    const cache = new OneSlotCache();
    installCaches(cache);
    cache.entries.set('https://huggingface.co/x/resolve/rev1/model_fp16.onnx', new Uint8Array([0]));
    const calls = countingFetch(new Uint8Array([1, 2, 3]));
    const entry = { key: KEY, expectedBytes: 3 };

    await loadModelBytes(KEY, () => {}, entry);
    await loadModelBytes(KEY, () => {}, entry);
    assert.deepEqual([...cache.entries.keys()], [KEY]);
    assert.equal(calls.length, 1);
});

test('quota errors and missing Cache Storage fall back to plain downloads', async () => {
    const cache = new FakeCache();
    cache.putError = new DOMException('full', 'QuotaExceededError');
    installCaches(cache);
    countingFetch(new Uint8Array([1, 2, 3]));
    const originalWarn = console.warn;
    console.warn = () => {};
    try {
        const bytes = await loadModelBytes(KEY, () => {}, { key: KEY, expectedBytes: 3 });
        assert.deepEqual([...bytes], [1, 2, 3]);
    } finally {
        console.warn = originalWarn;
    }
    assert.equal(cache.entries.size, 0);

    delete globalThis.caches;
    const calls = countingFetch(new Uint8Array([4, 5, 6]));
    const bytes = await loadModelBytes(KEY, () => {}, { key: KEY, expectedBytes: 3 });
    assert.deepEqual([...bytes], [4, 5, 6]);
    assert.equal(calls.length, 1);
});

test('a wrong-size download is rejected even when caching is off', async () => {
    countingFetch(new Uint8Array([1, 2]));
    await assert.rejects(
        loadModelBytes(KEY, () => {}, undefined, 3),
        /Downloaded model is 2 bytes; expected 3/,
    );
    // No expected size (an overridden modelUrl): any length is accepted.
    const bytes = await loadModelBytes(KEY, () => {}, undefined, undefined);
    assert.deepEqual([...bytes], [1, 2]);
});

test('clearModelCache deletes the model bucket and tolerates missing Cache Storage', async () => {
    const deleted = [];
    Object.defineProperty(globalThis, 'caches', {
        configurable: true,
        value: { delete: async name => (deleted.push(name), true) },
    });
    assert.equal(await clearModelCache(), true);
    assert.deepEqual(deleted, [MODEL_CACHE_NAME]);
    delete globalThis.caches;
    assert.equal(await clearModelCache(), false);
});

test('the package entry point exports the cache and load-error API', async () => {
    const index = await import('../dist/index.js');
    assert.equal(index.clearModelCache, clearModelCache);
    assert.equal(index.MODEL_CACHE_NAME, MODEL_CACHE_NAME);
    assert.equal(new index.ModelLoadError('x', 'fetch').stage, 'fetch');
});
