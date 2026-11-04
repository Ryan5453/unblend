import { MODEL_CONFIGS, type ModelPrecision, type ModelType } from 'unblend';

let shaderF16: Promise<boolean | null> | undefined;

/** Whether the default WebGPU adapter has shader-f16; null without one. */
export function adapterHasShaderF16(): Promise<boolean | null> {
    shaderF16 ??= (async () => {
        try {
            const adapter = await navigator.gpu?.requestAdapter();
            return adapter ? adapter.features.has('shader-f16') : null;
        } catch {
            return null;
        }
    })();
    return shaderF16;
}

/**
 * The app loads the fp16 files, about half the download. The RoFormers' fp16
 * files also compute in fp16, which WebGPU can only run with shader-f16, so
 * an adapter without it gets their fp32 files instead.
 */
export function appPrecision(model: ModelType, hasShaderF16: boolean | null): ModelPrecision {
    return hasShaderF16 === false && MODEL_CONFIGS[model].family === 'roformer' ? 'fp32' : 'fp16';
}
