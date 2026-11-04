/**
 * WebGPU checks for the onnx worker. onnxruntime-web neither checks that a
 * device can run a model's shaders nor reports GPU errors to the caller: its
 * handler only logs uncaptured validation errors (and drops out-of-memory
 * ones), so a run whose shaders failed to compile or whose buffers failed to
 * allocate still resolves, with garbage outputs.
 */

/**
 * Throw if an export that computes in fp16 is about to run on a device
 * without ``shader-f16``. ORT would still assign its fp16 kernels to WebGPU
 * and generate f16 shaders the device rejects.
 */
export function checkShaderF16(
    metadata: Record<string, string>,
    device: Pick<GPUDevice, 'features'>,
): void {
    if (metadata.compute_precision !== 'fp16' || device.features.has('shader-f16')) return;
    throw new Error(
        `This file computes in fp16, which needs the WebGPU 'shader-f16' feature, `
        + `and this GPU or browser does not provide it. Load the model with `
        + `precision: 'fp32'.`
    );
}

// Popped in reverse, so an out-of-memory or internal error is reported ahead
// of the validation errors it causes downstream.
const ERROR_FILTERS: readonly GPUErrorFilter[] = ['validation', 'out-of-memory', 'internal'];

/**
 * Run ``fn`` inside WebGPU error scopes and throw the first GPU error it
 * raised. ``fn`` has already resolved by then, so anything it allocates must
 * stay reachable for the caller to free. Without a device, just runs ``fn``.
 */
export async function withGpuErrorScopes<T>(
    device: Pick<GPUDevice, 'pushErrorScope' | 'popErrorScope'> | null,
    fn: () => Promise<T>,
): Promise<T> {
    if (device === null) return fn();
    for (const filter of ERROR_FILTERS) device.pushErrorScope(filter);
    let gpuError: string | undefined;
    let value: T;
    try {
        value = await fn();
    } finally {
        // Pop every scope even when fn threw, or later runs would nest inside
        // stale ones. An error from fn itself takes precedence.
        for (let i = ERROR_FILTERS.length - 1; i >= 0; i--) {
            const error = await device.popErrorScope().catch(() => null);
            if (error && gpuError === undefined) {
                gpuError = `WebGPU ${ERROR_FILTERS[i]} error: ${error.message}`;
            }
        }
    }
    if (gpuError !== undefined) throw new Error(gpuError);
    return value;
}
