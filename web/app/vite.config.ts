import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// onnxruntime-web statically references its .wasm/.mjs assets, so Vite
// bundles them. We don't want them: the worker sets wasmPaths to a CDN at
// runtime (see demucs/src/workers/onnx-worker.ts), and the .wasm file alone
// is 26MB — over Cloudflare Pages' 25MB per-file limit.
const stripOrtAssets: Plugin = {
  name: 'strip-ort-assets',
  apply: 'build',
  generateBundle(_, bundle) {
    for (const key of Object.keys(bundle)) {
      if (/ort-.*\.(wasm|mjs)$/.test(key)) {
        delete bundle[key];
      }
    }
  },
};

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    stripOrtAssets,
  ],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    // demucs-next ships tsc-transpiled JS whose workers are referenced with
    // `new Worker(new URL('./workers/*.js', import.meta.url))`. Excluding it
    // from esbuild dep pre-bundling lets Vite process those workers on demand
    // (resolving onnxruntime-web, emitting strippable ort-*.wasm), exactly as
    // it did when consuming the lib's source.
    exclude: ['@ffmpeg/ffmpeg', '@ffmpeg/util', 'demucs-next'],
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
