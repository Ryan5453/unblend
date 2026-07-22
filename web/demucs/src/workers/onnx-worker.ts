/**
 * ONNX Runtime worker. Handles both WebGPU and WASM backends — same code, the
 * caller picks via the ``backend`` field on the load message. Inputs, outputs,
 * activations, and compute are always float32. An "fp16" artifact uses fp16
 * only for weight storage; the exporter inserts a Cast(fp16 -> fp32) after
 * each converted weight, allowing ORT to fold the constant cast at load time.
 */

import * as onnx from 'onnxruntime-web';

let session: onnx.InferenceSession | null = null;

/**
 * Fetch the model with incremental progress. This trades the previous
 * URL-based load (ORT streams the file itself, never holding a second
 * full-size buffer in JS) for real download progress: the bytes are held
 * here as one contiguous buffer so `InferenceSession.create` can report
 * something better than a stalled bar for the ~100ms-950MB fetch. Peak
 * memory is briefly ~2x model size (this buffer + ORT's parsed copy).
 */
async function fetchModelBytes(
    url: string,
    onProgress: (loaded: number, total: number) => void
): Promise<Uint8Array> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }
    const totalHeader = response.headers.get('Content-Length');
    const total = totalHeader ? Number(totalHeader) : 0;

    if (!response.body) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        onProgress(bytes.byteLength, total || bytes.byteLength);
        return bytes;
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    let lastReport = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        const now = performance.now();
        if (now - lastReport >= 100) {
            onProgress(loaded, total);
            lastReport = now;
        }
    }
    onProgress(loaded, total || loaded);

    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
    }
    return bytes;
}

interface LoadMessage {
    type: 'load';
    requestId: number;
    modelUrl: string;
    backend: 'webgpu' | 'wasm';
    wasmPaths?: string;
    numThreads?: number;
    /** Defaults to 'all'; exposed for diagnosing EP-specific optimizer bugs. */
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
}

interface RunMessage {
    type: 'run';
    requestId: number;
    specReal: Float32Array;
    specImag: Float32Array;
    /** Absent for models without an audio input (RoFormer). */
    audio?: Float32Array;
    specShape: number[];
    audioShape?: number[];
}

interface UnloadMessage {
    type: 'unload';
    requestId: number;
}

type Message = LoadMessage | RunMessage | UnloadMessage;

interface ProgressResponse {
    type: 'progress';
    requestId: number;
    phase: 'download' | 'compile';
    loaded: number;
    total: number;
}

interface LoadResponse {
    type: 'load';
    requestId: number;
    success: boolean;
    backend?: 'webgpu' | 'wasm';
    error?: string;
}

interface RunResponse {
    type: 'run';
    requestId: number;
    success: boolean;
    outSpecReal?: Float32Array;
    outSpecImag?: Float32Array;
    outWave?: Float32Array;
    outSpecShape?: number[];
    outWaveShape?: number[];
    error?: string;
}

interface UnloadResponse {
    type: 'unload';
    requestId: number;
    success: boolean;
}

self.onmessage = async (event: MessageEvent<Message>) => {
    const msg = event.data;

    if (msg.type === 'load') {
        try {
            // Only override wasmPaths if the caller asked us to. Otherwise let
            // ORT resolve .wasm files via the bundler's default (next to the
            // bundled JS, via import.meta.url).
            if (msg.wasmPaths !== undefined) {
                onnx.env.wasm.wasmPaths = msg.wasmPaths;
            }
            // numThreads only affects the WASM backend. Setting >1 on a page
            // that isn't cross-origin isolated can fail/warn at session create,
            // so only apply it when WASM is actually the execution provider.
            if (msg.backend === 'wasm') {
                onnx.env.wasm.numThreads = msg.numThreads ?? 4;
            }
            onnx.env.logLevel = 'warning';

            const modelBytes = await fetchModelBytes(msg.modelUrl, (loaded, total) => {
                const progress: ProgressResponse = {
                    type: 'progress',
                    requestId: msg.requestId,
                    phase: 'download',
                    loaded,
                    total,
                };
                self.postMessage(progress);
            });
            const compiling: ProgressResponse = {
                type: 'progress',
                requestId: msg.requestId,
                phase: 'compile',
                loaded: 0,
                total: 0,
            };
            self.postMessage(compiling);

            session = await onnx.InferenceSession.create(modelBytes, {
                executionProviders: [msg.backend],
                graphOptimizationLevel: msg.graphOptimizationLevel ?? 'all',
            });

            const response: LoadResponse = {
                type: 'load',
                requestId: msg.requestId,
                success: true,
                backend: msg.backend,
            };
            self.postMessage(response);
        } catch (error) {
            console.error('[onnx-worker] load failed:', error);
            const response: LoadResponse = {
                type: 'load',
                requestId: msg.requestId,
                success: false,
                error: (error as Error).message,
            };
            self.postMessage(response);
        }
        return;
    }

    if (msg.type === 'run') {
        if (!session) {
            const response: RunResponse = {
                type: 'run',
                requestId: msg.requestId,
                success: false,
                error: 'No session loaded',
            };
            self.postMessage(response);
            return;
        }

        // Track every tensor created in this run for the finally block —
        // otherwise a failed run leaks WASM-heap/GPU buffers per segment
        // until the worker is torn down.
        const owned: { dispose(): void }[] = [];
        try {
            // Push each tensor as soon as it exists — if a later constructor
            // throws, the earlier ones must still reach the finally block.
            const specReal = new onnx.Tensor('float32', msg.specReal, msg.specShape);
            owned.push(specReal);
            const specImag = new onnx.Tensor('float32', msg.specImag, msg.specShape);
            owned.push(specImag);

            const feeds: Record<string, onnx.Tensor> = {
                spec_real: specReal,
                spec_imag: specImag,
            };
            // RoFormer graphs have no audio input / time branch.
            if (msg.audio !== undefined) {
                const audio = new onnx.Tensor('float32', msg.audio, msg.audioShape!);
                owned.push(audio);
                feeds.audio = audio;
            }

            const results = await session.run(feeds);
            for (const tensor of Object.values(results)) {
                owned.push(tensor);
            }

            const outSpecReal = results.out_spec_real;
            const outSpecImag = results.out_spec_imag;
            const outWave = results.out_wave;

            // The model's IO is float32 by design (Cast nodes bracket fp16
            // graphs). The .data casts below are unchecked, so a model whose
            // outputs are some other dtype would be silently mis-wrapped as
            // Float32Array. Verify the dtype up front so a mismatched model
            // fails loudly instead. out_wave exists only on HTDemucs graphs;
            // it is checked when present and its absence is reported to the
            // client (which knows whether the model should have one).
            const expected: [string, onnx.Tensor | undefined][] = [
                ['out_spec_real', outSpecReal],
                ['out_spec_imag', outSpecImag],
            ];
            if (outWave !== undefined) {
                expected.push(['out_wave', outWave]);
            }
            for (const [name, tensor] of expected) {
                if (tensor === undefined) {
                    throw new Error(`Model produced no '${name}' output`);
                }
                if (tensor.type !== 'float32') {
                    throw new Error(
                        `Expected output '${name}' to be float32, got '${tensor.type}'`
                    );
                }
            }
            // Only out_spec_real's dims are forwarded as outSpecShape, so
            // make sure the imag plane actually shares them.
            if (outSpecImag.dims.join(',') !== outSpecReal.dims.join(',')) {
                throw new Error(
                    `out_spec_imag dims [${outSpecImag.dims}] differ from ` +
                        `out_spec_real dims [${outSpecReal.dims}]`
                );
            }

            // Copy each output's bytes into fresh buffers so nothing posted
            // depends on the tensors' backing storage; the finally block
            // then disposes the tensors (WASM heap / GPU) — otherwise they'd
            // only be reclaimed by GC finalizers and accumulate across every
            // segment of every track.
            const outSpecRealData = new Float32Array(outSpecReal.data as Float32Array);
            const outSpecImagData = new Float32Array(outSpecImag.data as Float32Array);
            const outWaveData = outWave
                ? new Float32Array(outWave.data as Float32Array)
                : undefined;
            const outSpecShape = outSpecReal.dims as number[];
            const outWaveShape = outWave ? (outWave.dims as number[]) : undefined;

            const response: RunResponse = {
                type: 'run',
                requestId: msg.requestId,
                success: true,
                outSpecReal: outSpecRealData,
                outSpecImag: outSpecImagData,
                outWave: outWaveData,
                outSpecShape,
                outWaveShape,
            };

            // Transfer the fresh buffers — they're owned by this scope and no
            // longer needed here, so handing ownership to the client avoids a
            // structured-clone copy.
            const transfer: Transferable[] = [
                outSpecRealData.buffer,
                outSpecImagData.buffer,
            ];
            if (outWaveData) {
                transfer.push(outWaveData.buffer);
            }
            self.postMessage(response, transfer);
        } catch (error) {
            console.error('[onnx-worker] run failed:', error);
            const response: RunResponse = {
                type: 'run',
                requestId: msg.requestId,
                success: false,
                error: (error as Error).message,
            };
            self.postMessage(response);
        } finally {
            for (const tensor of owned.splice(0)) {
                try {
                    tensor.dispose();
                } catch {
                    // Best-effort cleanup; the response already went out.
                }
            }
        }
        return;
    }

    if (msg.type === 'unload') {
        // Always post the response, even if release() throws — otherwise the
        // client's unload() promise never settles and the caller hangs.
        try {
            if (session) {
                await session.release();
            }
        } catch (error) {
            console.error('[onnx-worker] release failed:', error);
        } finally {
            session = null;
            const response: UnloadResponse = {
                type: 'unload',
                requestId: msg.requestId,
                success: true,
            };
            self.postMessage(response);
        }
        return;
    }
};
