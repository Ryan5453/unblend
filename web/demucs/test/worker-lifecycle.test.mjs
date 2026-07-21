import assert from 'node:assert/strict';
import test from 'node:test';

import { ISTFTClient } from '../dist/istft-client.js';
import { OnnxClient } from '../dist/onnx-client.js';
import { Separator } from '../dist/separator.js';
import { STFTClient } from '../dist/stft-client.js';

class FakeWorker {
    static instances = [];
    static handler = () => {};

    onmessage = null;
    onerror = null;
    onmessageerror = null;
    messages = [];
    transfers = [];
    terminateCalls = 0;

    constructor(url) {
        this.url = String(url);
        FakeWorker.instances.push(this);
    }

    postMessage(message, transfer = []) {
        this.messages.push(message);
        this.transfers.push(transfer);
        FakeWorker.handler(this, message);
    }

    terminate() {
        this.terminateCalls++;
    }

    respond(data) {
        queueMicrotask(() => this.onmessage?.({ data }));
    }
}

const originalWorker = globalThis.Worker;
const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
globalThis.Worker = FakeWorker;

function resetFakes() {
    FakeWorker.instances = [];
    FakeWorker.handler = () => {};
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: { gpu: undefined },
    });
}

function successfulLoadHandler(worker, message) {
    if (message.type === 'load') {
        worker.respond({ type: 'load', requestId: message.requestId, success: true });
    } else if (message.type === 'configure') {
        worker.respond({ type: 'configure', requestId: message.requestId, success: true });
    } else if (message.type === 'unload') {
        worker.respond({ type: 'unload', requestId: message.requestId, success: true });
    }
}

async function loadedSeparator() {
    FakeWorker.handler = successfulLoadHandler;
    return Separator.load('htdemucs', { backend: 'wasm' });
}

function tinyAudioBuffer() {
    const mono = new Float32Array([0.25]);
    return {
        sampleRate: 44100,
        length: mono.length,
        numberOfChannels: 1,
        getChannelData: () => mono,
    };
}

test.beforeEach(resetFakes);
test.after(() => {
    if (originalWorker === undefined) delete globalThis.Worker;
    else globalThis.Worker = originalWorker;
    if (originalNavigatorDescriptor) {
        Object.defineProperty(globalThis, 'navigator', originalNavigatorDescriptor);
    } else {
        delete globalThis.navigator;
    }
});

test('OnnxClient rejects concurrent/post-terminal requests and terminates once', async () => {
    const client = new OnnxClient();
    const worker = FakeWorker.instances[0];
    const pending = client.load('model.onnx', 'wasm');

    await assert.rejects(
        client.unload(),
        /ONNX worker request already in progress/,
    );
    assert.equal(worker.messages.length, 1);

    const reason = new Error('cancelled');
    client.terminate(reason);
    client.terminate(new Error('ignored'));
    await assert.rejects(pending, error => error === reason);
    assert.equal(worker.terminateCalls, 1);

    await assert.rejects(client.unload(), /has been terminated/);
    assert.equal(worker.messages.length, 1);
});

test('STFT/iSTFT clients clear synchronous post errors and stay terminal', async () => {
    for (const [Client, invoke] of [
        [STFTClient, client => client.configure({})],
        [ISTFTClient, client => client.configure({})],
    ]) {
        FakeWorker.handler = () => {
            throw new Error('synthetic post failure');
        };
        const client = new Client();
        await assert.rejects(invoke(client), /synthetic post failure/);

        // Clearing the failed request permits another request; terminating it
        // rejects exactly that pending request and is idempotent.
        FakeWorker.handler = () => {};
        const pending = invoke(client);
        const worker = FakeWorker.instances.at(-1);
        const reason = new Error('terminal');
        client.terminate(reason);
        client.terminate();
        await assert.rejects(pending, error => error === reason);
        assert.equal(worker.terminateCalls, 1);
        const messageCount = worker.messages.length;
        await assert.rejects(invoke(client), /has been terminated/);
        assert.equal(worker.messages.length, messageCount);
    }
});

test('pre-aborted load constructs no workers', async () => {
    const controller = new AbortController();
    const reason = new DOMException('stop', 'AbortError');
    controller.abort(reason);

    await assert.rejects(
        Separator.load('htdemucs', { backend: 'wasm', signal: controller.signal }),
        error => error === reason,
    );
    assert.equal(FakeWorker.instances.length, 0);
});

test('abort during WebGPU load terminates promptly without WASM fallback', async () => {
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: { gpu: { requestAdapter: async () => ({}) } },
    });
    // Hold the ONNX load request forever; abort must terminate/reject it.
    FakeWorker.handler = () => {};
    const controller = new AbortController();
    const reason = new DOMException('cancel load', 'AbortError');
    const loading = Separator.load('htdemucs', {
        backend: 'webgpu',
        signal: controller.signal,
    });
    await new Promise(resolve => setTimeout(resolve, 0));
    controller.abort(reason);

    await assert.rejects(loading, error => error === reason);
    const onnxWorkers = FakeWorker.instances.filter(worker =>
        worker.url.includes('onnx-worker.js')
    );
    assert.equal(onnxWorkers.length, 1);
    assert.equal(onnxWorkers[0].terminateCalls, 1);
});

test('WebGPU load failure retries once on a fresh WASM worker', async () => {
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: { gpu: { requestAdapter: async () => ({}) } },
    });
    let loadCount = 0;
    FakeWorker.handler = (worker, message) => {
        if (message.type === 'load') {
            loadCount++;
            worker.respond({
                type: 'load',
                requestId: message.requestId,
                success: loadCount > 1,
                error: loadCount === 1 ? 'WebGPU session failed' : undefined,
            });
            return;
        }
        successfulLoadHandler(worker, message);
    };

    const separator = await Separator.load('htdemucs', { backend: 'webgpu' });
    assert.equal(separator.backend, 'wasm');
    const onnxWorkers = FakeWorker.instances.filter(worker =>
        worker.url.includes('onnx-worker.js')
    );
    assert.equal(onnxWorkers.length, 2);
    assert.equal(onnxWorkers[0].terminateCalls, 1);
    assert.equal(onnxWorkers[1].terminateCalls, 0);
    await separator.unload();
    assert.equal(onnxWorkers[1].terminateCalls, 1);
});

test('idle unload is graceful, bounded, and idempotent', async () => {
    const separator = await loadedSeparator();
    const workers = [...FakeWorker.instances];

    await separator.unload();
    assert.equal(
        workers[0].messages.filter(message => message.type === 'unload').length,
        1,
    );
    assert.deepEqual(workers.map(worker => worker.terminateCalls), [1, 1, 1]);

    await separator.unload();
    assert.deepEqual(workers.map(worker => worker.terminateCalls), [1, 1, 1]);
});

test('active unload cancels destructively without posting graceful unload', async () => {
    const separator = await loadedSeparator();
    FakeWorker.handler = (worker, message) => {
        if (message.type === 'process' && worker.url.includes('stft-worker.js')) return;
        successfulLoadHandler(worker, message);
    };
    const separating = separator.separate(tinyAudioBuffer());
    await new Promise(resolve => setTimeout(resolve, 0));

    await separator.unload();
    await assert.rejects(separating, /unloaded during active separation/);
    assert.equal(
        FakeWorker.instances[0].messages.filter(message => message.type === 'unload').length,
        0,
    );
    assert.deepEqual(
        FakeWorker.instances.map(worker => worker.terminateCalls),
        [1, 1, 1],
    );
});

test('active separation rejects concurrency; abort invalidates the instance', async () => {
    const separator = await loadedSeparator();
    // Hold STFT processing so the first separation remains active.
    FakeWorker.handler = (worker, message) => {
        if (message.type === 'process' && worker.url.includes('stft-worker.js')) return;
        successfulLoadHandler(worker, message);
    };
    const controller = new AbortController();
    const reason = new DOMException('cancel run', 'AbortError');
    const first = separator.separate(tinyAudioBuffer(), { signal: controller.signal });
    await new Promise(resolve => setTimeout(resolve, 0));

    await assert.rejects(
        separator.separate(tinyAudioBuffer()),
        /Separation already in progress/,
    );
    controller.abort(reason);
    await assert.rejects(first, error => error === reason);
    await assert.rejects(separator.separate(tinyAudioBuffer()), /has been unloaded/);
    assert.deepEqual(
        FakeWorker.instances.map(worker => worker.terminateCalls),
        [1, 1, 1],
    );
});

test('worker-backed separation failure invalidates the instance', async () => {
    const separator = await loadedSeparator();
    FakeWorker.handler = (worker, message) => {
        if (message.type === 'process' && worker.url.includes('stft-worker.js')) {
            worker.respond({
                type: 'process',
                requestId: message.requestId,
                success: false,
                error: 'synthetic STFT failure',
            });
            return;
        }
        successfulLoadHandler(worker, message);
    };

    await assert.rejects(separator.separate(tinyAudioBuffer()), /synthetic STFT failure/);
    await assert.rejects(separator.separate(tinyAudioBuffer()), /has been unloaded/);
    assert.deepEqual(
        FakeWorker.instances.map(worker => worker.terminateCalls),
        [1, 1, 1],
    );
});

test('invalid options reject before marking the separator active', async () => {
    const separator = await loadedSeparator();
    await assert.rejects(
        separator.separate(tinyAudioBuffer(), { shifts: 0 }),
        /shifts must be an integer/,
    );

    // The same instance remains valid after argument validation. Hold and
    // abort a real run to prove it can still enter the active state.
    FakeWorker.handler = (worker, message) => {
        if (message.type === 'process' && worker.url.includes('stft-worker.js')) return;
        successfulLoadHandler(worker, message);
    };
    const controller = new AbortController();
    const pending = separator.separate(tinyAudioBuffer(), { signal: controller.signal });
    await new Promise(resolve => setTimeout(resolve, 0));
    controller.abort();
    await assert.rejects(pending, error => error?.name === 'AbortError');
});
