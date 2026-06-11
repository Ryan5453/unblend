/**
 * ONNX Runtime worker. Handles both WebGPU and WASM backends — same code, the
 * caller picks via the ``backend`` field on the load message. Inputs and
 * outputs are always float32; for fp16 models the weights and activations are
 * fp16 internally but the converter inserts Cast nodes at the IO boundary so
 * the JS pipeline stays fp32 end-to-end.
 */

import * as onnx from 'onnxruntime-web';

let session: onnx.InferenceSession | null = null;

interface LoadMessage {
    type: 'load';
    requestId: number;
    modelUrl: string;
    backend: 'webgpu' | 'wasm';
    wasmPaths?: string;
    numThreads?: number;
}

interface RunMessage {
    type: 'run';
    requestId: number;
    specReal: Float32Array;
    specImag: Float32Array;
    audio: Float32Array;
    specShape: number[];
    audioShape: number[];
}

interface UnloadMessage {
    type: 'unload';
    requestId: number;
}

type Message = LoadMessage | RunMessage | UnloadMessage;

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

            session = await onnx.InferenceSession.create(msg.modelUrl, {
                executionProviders: [msg.backend],
                graphOptimizationLevel: 'all',
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

        try {
            const specReal = new onnx.Tensor('float32', msg.specReal, msg.specShape);
            const specImag = new onnx.Tensor('float32', msg.specImag, msg.specShape);
            const audio = new onnx.Tensor('float32', msg.audio, msg.audioShape);

            const results = await session.run({
                spec_real: specReal,
                spec_imag: specImag,
                audio,
            });

            const outSpecReal = results.out_spec_real;
            const outSpecImag = results.out_spec_imag;
            const outWave = results.out_wave;

            // The model's IO is float32 by design (Cast nodes bracket fp16
            // graphs). The .data casts below are unchecked, so a model whose
            // outputs are some other dtype would be silently mis-wrapped as
            // Float32Array. Verify the dtype up front so a mismatched model
            // fails loudly instead.
            for (const [name, tensor] of [
                ['out_spec_real', outSpecReal],
                ['out_spec_imag', outSpecImag],
                ['out_wave', outWave],
            ] as const) {
                if (tensor.type !== 'float32') {
                    throw new Error(
                        `Expected output '${name}' to be float32, got '${tensor.type}'`
                    );
                }
            }

            // Copy each output's bytes into fresh buffers so we no longer
            // depend on dispose() running after postMessage. Dispose the
            // tensors right away to free their backing buffers (WASM heap /
            // GPU) — otherwise they'd only be reclaimed by GC finalizers and
            // accumulate across every segment of every track.
            const outSpecRealData = new Float32Array(outSpecReal.data as Float32Array);
            const outSpecImagData = new Float32Array(outSpecImag.data as Float32Array);
            const outWaveData = new Float32Array(outWave.data as Float32Array);
            const outSpecShape = outSpecReal.dims as number[];
            const outWaveShape = outWave.dims as number[];

            specReal.dispose();
            specImag.dispose();
            audio.dispose();
            outSpecReal.dispose();
            outSpecImag.dispose();
            outWave.dispose();

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
            self.postMessage(response, [
                outSpecRealData.buffer,
                outSpecImagData.buffer,
                outWaveData.buffer,
            ]);
        } catch (error) {
            console.error('[onnx-worker] run failed:', error);
            const response: RunResponse = {
                type: 'run',
                requestId: msg.requestId,
                success: false,
                error: (error as Error).message,
            };
            self.postMessage(response);
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
