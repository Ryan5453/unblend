export function About() {
    return (
        <div className="content-page">
            <h1 className="content-title">About</h1>

            <div className="content-body">
                <p>
                    <strong>un/blend</strong> is a free, open-source audio stem separation tool.
                    Everything runs entirely in your browser, so your audio files
                    never leave your device.
                </p>

                <p>
                    Model weights are converted to ONNX format and run in-browser via
                    onnxruntime-web. When a model is loaded, its weights are downloaded along with
                    a runtime binary matched to your browser: WebGPU if it's supported, WASM otherwise.
                    BS-RoFormer SW and SCNet XL need WebGPU: they use more working memory than WASM can provide.
                </p>

                <p>
                    Audio files are decoded with <a href="https://mediabunny.dev/">Mediabunny</a>, which
                    uses your browser's native decoders where possible. For formats that can't be decoded
                    natively, the app falls back to <a href="https://ffmpegwasm.netlify.app/">ffmpeg.wasm</a>.
                </p>

                <p>
                    Because the model itself runs locally on your machine's CPU/GPU rather than a server, it's a heavy tab:
                    you should expect high memory and
                    power use for the duration of a separation. Browsers with stricter tab memory limits,
                    Safari in particular, may reload or kill the tab on longer tracks.
                    If that happens, a Chromium-based browser will likely perform better.
                </p>

                <p>
                    un/blend is built on open-source software. See
                    the <a href="/third-party-licenses.txt">third-party licenses</a> for
                    the notices of the libraries bundled into this site. The web workers also
                    bundle <a href="https://github.com/microsoft/onnxruntime">ONNX Runtime Web</a> (MIT, with some bundled Apache-2.0
                    code) and <a href="https://github.com/indutny/fft.js">fft.js</a> (MIT), whose
                    notices are kept in those scripts.
                </p>
            </div>
        </div>
    );
}
