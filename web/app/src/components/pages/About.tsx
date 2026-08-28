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
                    onnxruntime-web. When a model is loaded, the model weights and a matching
                    runtime binary are downloaded depending whether or not your browser supports WebGPU.
                </p>

                <p>
                    Audio files are decoded with <a href="https://mediabunny.dev/">MediaBunny</a>, which
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

            </div>
        </div>
    );
}
