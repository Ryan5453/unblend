export function About() {
    return (
        <div className="content-page">
            <h1 className="content-title">About</h1>

            <div className="content-body">
                    <p>
                        <strong>un/blend</strong> is a free, open-source audio stem separation tool powered by
                        Meta AI's Demucs model. Everything runs entirely in your browser, so your audio files
                        never leave your device.
                    </p>

                    <p>
                        Demucs separates a mixed track into individual stems such as drums, bass, vocals,
                        and other instruments. The model is converted to ONNX format and runs in-browser via
                        onnxruntime-web. When a model is loaded, the runtime binary (~26MB) and model weights (~90MB) are
                        downloaded. Inference uses WebGPU when your browser supports it, falling back to
                        WebAssembly otherwise.
                    </p>

                    <p>
                        Audio files are decoded with <a href="https://mediabunny.dev/">MediaBunny</a>, which
                        uses your browser's native decoders where possible. For formats that can't be decoded
                        natively, the app falls back to <a href="https://ffmpegwasm.netlify.app/">ffmpeg.wasm</a>.
                    </p>
            </div>
        </div>
    );
}
