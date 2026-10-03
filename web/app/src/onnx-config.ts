// Cloudflare Pages caps files at 25 MB, so the 26 MB ORT .wasm is stripped from
// the build (vite.config.ts) and fetched from jsDelivr instead. The version must
// match the resolved onnxruntime-web package: the JS API and .wasm are
// versioned together.
export const ORT_WASM_PATHS =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
