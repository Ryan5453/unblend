export function Privacy() {
    return (
        <div className="w-full max-w-3xl mx-auto px-6 py-12 flex-1">
            {/* Content */}
            <div className="content-card">
                <h1 className="content-title">Privacy Policy</h1>

                <div className="content-body">
                    <p>
                        <strong>Effective Date:</strong> June 2026
                    </p>

                    <h2>Data Collection</h2>
                    <p>
                        <strong>We don't collect any data.</strong> This application runs entirely in your browser.
                        Your audio files are processed locally on your device and are <strong>never</strong> uploaded to any server.
                    </p>

                    <h2>Local Processing</h2>
                    <p>
                        All audio separation is performed using WebGPU or WebAssembly directly in your browser.
                        The model is downloaded from HuggingFace when loaded. No audio data ever leaves your device.
                    </p>

                    <h2>Cookies & Analytics</h2>
                    <p>
                        We do not use cookies or any in-app tracking technologies. This site is hosted on
                        Cloudflare Pages, which collects basic, privacy-respecting web analytics (e.g., page views)
                        at the infrastructure level. No personal data is collected or stored by us.
                    </p>

                    <h2>Third-Party Services</h2>
                    <p>
                        We do not upload any data to any third-party services.
                        However, some resources are loaded from third-party services:
                    </p>
                    <ul>
                        <li>Hugging Face: ONNX model files</li>
                        <li>jsDelivr: ONNX Web Runtime, ffmpeg.wasm</li>
                    </ul>

                    <h2>Open Source</h2>
                    <p>
                        You can verify all privacy claims by viewing the <a href="https://github.com/Ryan5453/demucs-next" target="_blank" rel="noopener noreferrer">source code</a> yourself.
                    </p>

                </div>
            </div>
        </div>
    );
}
