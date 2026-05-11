// We deploy to Cloudflare Pages which has a 25MB per-file limit, so we
// can't ship onnxruntime-web's 26MB .wasm in our build output. The matching
// strip-ort-assets plugin in vite.config.ts drops the would-be-bundled file;
// here we tell ORT to fetch it from jsDelivr at runtime instead. Other
// deployments without size limits can omit this entirely and let the
// bundler bake the .wasm into dist/.
export const ORT_WASM_PATHS =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/';
