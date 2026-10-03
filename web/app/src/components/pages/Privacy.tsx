import { useState } from 'react';
import { clearModelCache } from 'unblend';

function ClearModelCache() {
    const [status, setStatus] = useState<'idle' | 'busy' | 'cleared' | 'empty'>('idle');
    const clear = async () => {
        setStatus('busy');
        setStatus((await clearModelCache()) ? 'cleared' : 'empty');
    };
    return (
        <p>
            <button className="spec" onClick={() => void clear()} disabled={status === 'busy'}>
                CLEAR CACHED MODELS
            </button>{' '}
            <span role="status">
                {status === 'cleared' && 'Cached models deleted.'}
                {status === 'empty' && 'No cached models to delete.'}
            </span>
        </p>
    );
}

export function Privacy() {
    return (
        <div className="content-page">
            <h1 className="content-title">Privacy Policy</h1>

            <div className="content-body">
                    <p>
                        <strong>Effective Date:</strong> September 28, 2026
                    </p>

                    <h2>Data Collection</h2>
                    <p>
                        <strong>We don't collect any data.</strong> un/blend runs entirely in your browser.
                        Your audio files are processed locally on your device and are <strong>never</strong> uploaded to any server.
                    </p>

                    <h2>Local Processing</h2>
                    <p>
                        All audio separation is performed using WebGPU or WebAssembly directly in your browser.
                        The model is downloaded from Hugging Face when loaded and cached in your browser's storage so later visits can skip the download. No audio data ever leaves your device.
                    </p>
                    <ClearModelCache />

                    <h2>Cookies & Analytics</h2>
                    <p>
                        We do not use cookies, analytics, or any other tracking technologies. This site is
                        hosted on Cloudflare Pages as static files; no personal data is collected or stored by us.
                    </p>

                    <h2>Third-Party Services</h2>
                    <p>
                        We do not upload any data to any third-party services.
                        However, some resources are loaded from third-party services:
                    </p>
                    <ul>
                        <li>Hugging Face: ONNX model files</li>
                        <li>jsDelivr: ONNX Web Runtime, ffmpeg.wasm</li>
                        <li>Google Fonts: web fonts</li>
                    </ul>

                    <h2>Open Source</h2>
                    <p>
                        You can verify all privacy claims by viewing the <a href="https://github.com/Ryan5453/unblend" target="_blank" rel="noopener noreferrer">source code</a> yourself.
                    </p>

            </div>
        </div>
    );
}
