import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// onnxruntime-web statically references its .wasm/.mjs assets, so Vite
// bundles them. We don't want them: the app passes wasmPaths (src/onnx-config.ts)
// through Separator.load so ORT fetches them from a CDN, and the .wasm file
// alone is 26MB — over Cloudflare Pages' 25MB per-file limit.
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
    // unblend ships tsc-transpiled JS whose workers are referenced with
    // `new Worker(new URL('./workers/*.js', import.meta.url))`. Excluding it
    // from esbuild dep pre-bundling lets Vite process those workers on demand
    // (resolving onnxruntime-web and emitting strippable ort-*.wasm).
    exclude: ['@ffmpeg/ffmpeg', '@ffmpeg/util', 'unblend'],
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
    // Third-party notices for everything bundled (e.g. Mediabunny, MPL-2.0),
    // linked from the About page. A .txt name so browsers display it inline.
    license: { fileName: 'third-party-licenses.txt' },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
