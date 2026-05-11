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
    modelUrl: string;
    backend: 'webgpu' | 'wasm';
    wasmPaths?: string;
    numThreads?: number;
}

interface RunMessage {
    type: 'run';
    specReal: Float32Array;
    specImag: Float32Array;
    audio: Float32Array;
    specShape: number[];
    audioShape: number[];
}

interface UnloadMessage {
    type: 'unload';
}

type Message = LoadMessage | RunMessage | UnloadMessage;

interface LoadResponse {
    type: 'load';
    success: boolean;
    backend?: 'webgpu' | 'wasm';
    error?: string;
}

interface RunResponse {
    type: 'run';
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
            onnx.env.wasm.numThreads = msg.numThreads ?? 4;
            onnx.env.logLevel = 'warning';

            session = await onnx.InferenceSession.create(msg.modelUrl, {
                executionProviders: [msg.backend],
                graphOptimizationLevel: 'all',
            });

            const response: LoadResponse = {
                type: 'load',
                success: true,
                backend: msg.backend,
            };
            self.postMessage(response);
        } catch (error) {
            console.error('[onnx-worker] load failed:', error);
            const response: LoadResponse = {
                type: 'load',
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

            const response: RunResponse = {
                type: 'run',
                success: true,
                outSpecReal: outSpecReal.data as Float32Array,
                outSpecImag: outSpecImag.data as Float32Array,
                outWave: outWave.data as Float32Array,
                outSpecShape: outSpecReal.dims as number[],
                outWaveShape: outWave.dims as number[],
            };

            specReal.dispose();
            specImag.dispose();
            audio.dispose();

            self.postMessage(response);
        } catch (error) {
            console.error('[onnx-worker] run failed:', error);
            const response: RunResponse = {
                type: 'run',
                success: false,
                error: (error as Error).message,
            };
            self.postMessage(response);
        }
        return;
    }

    if (msg.type === 'unload') {
        if (session) {
            await session.release();
            session = null;
        }
        const response: UnloadResponse = { type: 'unload', success: true };
        self.postMessage(response);
        return;
    }
};
