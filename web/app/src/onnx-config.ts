// We deploy to Cloudflare Pages which has a 25MB per-file limit, so we
// can't ship onnxruntime-web's 26MB .wasm in our build output. The matching
// strip-ort-assets plugin in vite.config.ts drops the would-be-bundled file;
// here we tell ORT to fetch it from jsDelivr at runtime instead. Other
// deployments without size limits can omit this entirely and let the
// bundler bake the .wasm into dist/.
//
// IMPORTANT: this version MUST match the onnxruntime-web JS version resolved
// in package.json (the JS API and the .wasm binary are versioned together and
// ORT will fail if they diverge).
export const ORT_WASM_PATHS =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
