import { useState, useCallback, useRef, useEffect } from 'react';
import type { DemucsState } from '../types';
import { SAMPLE_RATE, Separator, type ModelType, type ModelPrecision } from 'demucs-next';
import { createWavBlob } from '../utils/wav-utils';
import { decodeAudioFile } from '../utils/audio-decoder';
import { ORT_WASM_PATHS } from '../onnx-config';

const initialState: DemucsState = {
    modelLoaded: false,
    modelLoading: false,
    audioLoaded: false,
    audioBuffer: null,
    audioFile: null,
    separating: false,
    progress: 0,
    status: 'Ready',
};

export function useDemucs() {
    const [state, setState] = useState<DemucsState>(initialState);
    const [audioError, setAudioError] = useState<string | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const separatorRef = useRef<Separator | null>(null);
    const modelLoadInFlightRef = useRef(false);

    // Store pre-created blob URLs
    const [stemUrls, setStemUrls] = useState<Record<string, string>>({});
    // Store artwork URL (album art from audio file)
    const [artworkUrl, setArtworkUrl] = useState<string | null>(null);
    // Mirror the latest object URLs into refs so the unmount cleanup can
    // revoke them without reading stale state from its empty-deps closure.
    const stemUrlsRef = useRef<Record<string, string>>({});
    const artworkUrlRef = useRef<string | null>(null);
    // Store track metadata from audio file
    const [trackTitle, setTrackTitle] = useState<string | null>(null);
    const [trackArtist, setTrackArtist] = useState<string | null>(null);

    // Route diagnostics to the console. These were previously accumulated in an
    // unbounded state array that no component ever rendered.
    const addLog = useCallback((message: string, type: 'info' | 'success' | 'error' = 'info') => {
        if (type === 'error') {
            console.error(`[demucs] ${message}`);
        } else {
            console.log(`[demucs] ${message}`);
        }
    }, []);

    const setStatus = useCallback((status: string) => {
        setState(prev => ({ ...prev, status }));
    }, []);

    const setProgress = useCallback((progress: number) => {
        setState(prev => ({ ...prev, progress }));
    }, []);

    const getAudioContext = useCallback(() => {
        if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
        }
        return audioContextRef.current;
    }, []);

    const loadModel = useCallback(async (
        model: ModelType,
        backend: 'webgpu' | 'wasm' = 'webgpu',
        precision: ModelPrecision = 'fp16',
    ) => {
        // Reject concurrent loads: a second call racing the first would tear
        // down or leak the separator the first one is still creating.
        if (modelLoadInFlightRef.current) {
            addLog('A model load is already in progress', 'error');
            return false;
        }
        modelLoadInFlightRef.current = true;

        try {
            // If a model is already loaded, tear it down before loading another.
            if (separatorRef.current) {
                await separatorRef.current.unload();
                separatorRef.current = null;
            }

            setState(prev => ({ ...prev, modelLoading: true, modelLoaded: false }));
            addLog(`Loading ${model} (${precision})...`, 'info');
            const start = performance.now();

            const separator = await Separator.load(model, {
                backend,
                precision,
                wasmPaths: ORT_WASM_PATHS,
            });
            separatorRef.current = separator;

            const elapsed = ((performance.now() - start) / 1000).toFixed(2);
            if (backend === 'webgpu' && separator.backend === 'wasm') {
                addLog('WebGPU unavailable, fell back to WASM', 'info');
            }
            addLog(
                `Loaded ${separator.backend}/${separator.precision} in ${elapsed}s (${separator.sources.join(', ')})`,
                'success'
            );

            setState(prev => ({ ...prev, modelLoading: false, modelLoaded: true }));
            return true;
        } catch (err) {
            addLog(`Failed to load ${model}: ${(err as Error).message}`, 'error');
            setState(prev => ({ ...prev, modelLoading: false, modelLoaded: false }));
            return false;
        } finally {
            modelLoadInFlightRef.current = false;
        }
    }, [addLog]);

    const clearAudioError = useCallback(() => {
        setAudioError(null);
    }, []);

    const loadAudio = useCallback(async (file: File) => {
        try {
            // Revoke object URLs from the previous track before it is replaced.
            setStemUrls(prev => {
                Object.values(prev).forEach(url => URL.revokeObjectURL(url));
                return {};
            });
            setArtworkUrl(prev => {
                if (prev) URL.revokeObjectURL(prev);
                return null;
            });
            // Clear previous track metadata up front so an untagged track
            // does not keep showing the previous track's title/artist.
            setTrackTitle(null);
            setTrackArtist(null);

            setAudioError(null);
            addLog(`Loading audio: ${file.name}`, 'info');
            const ctx = getAudioContext();

            const { buffer: audioBuffer, artwork, title, artist, usedFallback } = await decodeAudioFile(file, ctx);

            if (usedFallback === 'ffmpeg') {
                addLog('Audio decoded using fallback decoder (ffmpeg.wasm)', 'info');
            } else {
                addLog('Audio decoded with Mediabunny', 'info');
            }

            // Store artwork if present
            if (artwork) {
                setArtworkUrl(artwork);
                addLog('Album artwork extracted', 'info');
            }

            // Store track metadata if present
            if (title) {
                setTrackTitle(title);
                addLog(`Track title: ${title}`, 'info');
            }
            if (artist) {
                setTrackArtist(artist);
                addLog(`Artist: ${artist}`, 'info');
            }

            addLog('Audio loaded successfully.', 'success');

            setState(prev => ({
                ...prev,
                audioLoaded: true,
                audioBuffer,
                audioFile: file,
            }));
        } catch (error) {
            const errorMessage = (error as Error).message;
            addLog(`Failed to load audio: ${errorMessage}`, 'error');
            setAudioError(errorMessage);
        }
    }, [addLog, getAudioContext]);

    const separateAudio = useCallback(async () => {
        const separator = separatorRef.current;
        if (!separator) {
            addLog('Model not loaded', 'error');
            return;
        }
        if (!state.audioBuffer) {
            addLog('Audio not loaded', 'error');
            return;
        }

        try {
            setState(prev => ({ ...prev, separating: true }));
            setStemUrls({});
            setStatus('Preparing audio...');
            setProgress(0);

            // Yield once so React paints the "separating" UI before the
            // pipeline starts hammering the main thread.
            await new Promise(resolve => setTimeout(resolve, 0));
            addLog('Starting separation...', 'info');

            const result = await separator.separate(state.audioBuffer, {
                onProgress: ({ segIdx, totalSegs, fraction }) => {
                    setStatus(`Separating segment ${segIdx} of ${totalSegs}...`);
                    setProgress(fraction * 95);
                },
            });

            // Build blob URLs for the player UI.
            setStatus('Finalizing...');
            setProgress(98);

            const urls: Record<string, string> = {};

            for (const [source, samples] of Object.entries(result.stems)) {
                const blob = createWavBlob(samples, 2, SAMPLE_RATE);
                urls[source] = URL.createObjectURL(blob);
            }

            setStemUrls(urls);

            setStatus('Complete!');
            setProgress(100);
            addLog(`Finished separation in ${(result.wallMs / 1000).toFixed(2)}s.`, 'success');
            setState(prev => ({ ...prev, separating: false }));
        } catch (error) {
            addLog(`Separation failed: ${(error as Error).message}`, 'error');
            setStatus('Error during separation');
            setProgress(0);
            setState(prev => ({ ...prev, separating: false }));
        }
    }, [state.audioBuffer, addLog, setStatus, setProgress]);

    // Keep refs in sync with the latest object URLs for the unmount cleanup.
    useEffect(() => {
        stemUrlsRef.current = stemUrls;
    }, [stemUrls]);
    useEffect(() => {
        artworkUrlRef.current = artworkUrl;
    }, [artworkUrl]);

    // Terminate the separator's workers, close the AudioContext, and revoke
    // outstanding object URLs when the hook unmounts to free audio resources.
    useEffect(() => {
        return () => {
            if (separatorRef.current) {
                void separatorRef.current.unload();
                separatorRef.current = null;
            }
            if (audioContextRef.current) {
                void audioContextRef.current.close();
                audioContextRef.current = null;
            }
            Object.values(stemUrlsRef.current).forEach(url => URL.revokeObjectURL(url));
            if (artworkUrlRef.current) URL.revokeObjectURL(artworkUrlRef.current);
        };
    }, []);

    return {
        ...state,
        stemUrls,
        artworkUrl,
        trackTitle,
        trackArtist,
        audioError,
        loadModel,
        loadAudio,
        clearAudioError,
        separateAudio,
    };
}
