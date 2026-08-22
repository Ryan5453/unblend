import { useState, useCallback, useRef, useEffect } from 'react';
import type { DemucsState, ProgressPhase } from '../types';
import { SAMPLE_RATE, Separator, type ModelType, type ModelPrecision } from 'unblend';
import { decodeAudioFile } from '../utils/audio-decoder';
import { ORT_WASM_PATHS } from '../onnx-config';
import { finalizeStems } from '../utils/stem-finalizer';

function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}

const initialState: DemucsState = {
    modelLoaded: false,
    modelLoading: false,
    audioLoaded: false,
    audioBuffer: null,
    audioFile: null,
    separating: false,
    progressDeterminate: false,
    progressPhase: 'idle',
    progress: 0,
    segmentsDone: 0,
    segmentsTotal: 0,
    segmentStartedAtMs: 0,
    segmentExpectedMs: 0,
    status: 'Ready',
};

const DEFAULT_SEGMENT_MS: Record<ModelType, number> = {
    htdemucs: 1_500,
    htdemucs_6s: 2_000,
    bs_roformer_sw: 20_000,
    melband_roformer_kim: 7_000,
    // Estimate only: SCNet has not been timed in a browser yet. Its twelve
    // sequential bidirectional LSTMs are the bulk of the work and parallelise
    // poorly, so it is slower per second of audio than its size suggests.
    scnet_small: 8_000,
};

export function useDemucs() {
    const [state, setState] = useState<DemucsState>(initialState);
    const [loadedModel, setLoadedModel] = useState<ModelType | null>(null);
    const [audioError, setAudioError] = useState<string | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const separatorRef = useRef<Separator | null>(null);
    const mountedRef = useRef(true);
    const modelLoadInFlightRef = useRef(false);
    const separateInFlightRef = useRef(false);
    const loadAudioInFlightRef = useRef(false);
    const modelLoadAbortRef = useRef<AbortController | null>(null);
    const separationAbortRef = useRef<AbortController | null>(null);
    // Read at separation time (not captured at click time) so a click-time
    // closure that survives an await — e.g. Home's handleSeparate awaiting a
    // long model download while the user swaps tracks — separates the track
    // the UI is actually showing.
    const audioBufferRef = useRef<AudioBuffer | null>(null);

    // Terminal-style log lines surfaced to the processing view.
    const [logs, setLogs] = useState<string[]>([]);
    // Store pre-created blob URLs.
    const [originalUrl, setOriginalUrl] = useState<string | null>(null);
    const [stemUrls, setStemUrls] = useState<Record<string, string>>({});
    // Precomputed waveform peaks per stem (0..1), for the studio lanes.
    const [stemPeaks, setStemPeaks] = useState<Record<string, number[]>>({});
    // Store artwork URL (album art from audio file)
    const [artworkUrl, setArtworkUrl] = useState<string | null>(null);
    // Mirror the latest object URLs into refs so the unmount cleanup can
    // revoke them without reading stale state from its empty-deps closure.
    const originalUrlRef = useRef<string | null>(null);
    const stemUrlsRef = useRef<Record<string, string>>({});
    const artworkUrlRef = useRef<string | null>(null);
    // Store track metadata from audio file
    const [trackTitle, setTrackTitle] = useState<string | null>(null);
    const [trackArtist, setTrackArtist] = useState<string | null>(null);

    // Route diagnostics to the console. These were previously accumulated in an
    // unbounded state array that no component ever rendered.
    const addLog = useCallback((message: string, type: 'info' | 'success' | 'error' = 'info') => {
        if (type === 'error') {
            console.error(`[unblend] ${message}`);
        } else {
            console.log(`[unblend] ${message}`);
        }
        setLogs(prev => {
            const next = [...prev, message];
            return next.length > 200 ? next.slice(next.length - 200) : next;
        });
    }, []);

    const clearLogs = useCallback(() => setLogs([]), []);

    const setStatus = useCallback((status: string) => {
        setState(prev => ({ ...prev, status }));
    }, []);

    const setProgress = useCallback((
        progress: number,
        determinate = true,
        progressPhase?: ProgressPhase,
    ) => {
        setState(prev => ({
            ...prev,
            progress,
            progressDeterminate: determinate,
            ...(progressPhase ? { progressPhase } : {}),
            ...(progressPhase && progressPhase !== 'separate'
                ? {
                    segmentsDone: 0,
                    segmentsTotal: 0,
                    segmentStartedAtMs: 0,
                    segmentExpectedMs: 0,
                }
                : {}),
        }));
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
        if (modelLoadInFlightRef.current) {
            const message = 'A model load is already in progress';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        if (separateInFlightRef.current) {
            const message = 'Cannot switch models while separation is in progress';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }

        const controller = new AbortController();
        modelLoadAbortRef.current = controller;
        modelLoadInFlightRef.current = true;
        setAudioError(null);

        try {
            if (separatorRef.current) {
                await separatorRef.current.unload();
                separatorRef.current = null;
                setLoadedModel(null);
            }
            if (!mountedRef.current) return false;

            setState(prev => ({
                ...prev,
                modelLoading: true,
                modelLoaded: false,
                progress: 0,
                progressDeterminate: false,
                progressPhase: 'initialize',
            }));
            addLog(`Loading ${model} (${precision})...`, 'info');
            setStatus('Connecting...');
            const start = performance.now();

            const separator = await Separator.load(model, {
                backend,
                precision,
                wasmPaths: ORT_WASM_PATHS,
                signal: controller.signal,
                onProgress: (phase, loaded, total) => {
                    if (!mountedRef.current) return;
                    if (phase === 'download') {
                        const loadedMiB = (loaded / (1024 * 1024)).toFixed(1);
                        if (total > 0) {
                            const totalMiB = (total / (1024 * 1024)).toFixed(1);
                            setStatus(`Downloading model... ${loadedMiB} / ${totalMiB} MiB`);
                            setProgress((loaded / total) * 100, true, 'download');
                        } else {
                            setStatus(`Downloading model... ${loadedMiB} MiB`);
                            setProgress(0, false, 'download');
                        }
                    } else {
                        // ORT exposes no progress for runtime setup or graph
                        // compilation, so do not fabricate a percentage.
                        setStatus('Initializing ONNX runtime...');
                        setProgress(0, false, 'initialize');
                    }
                },
            });
            if (!mountedRef.current || controller.signal.aborted) {
                await separator.unload();
                return false;
            }
            separatorRef.current = separator;
            setLoadedModel(model);

            const elapsed = ((performance.now() - start) / 1000).toFixed(2);
            if (backend === 'webgpu' && separator.backend === 'wasm') {
                addLog('WebGPU unavailable, fell back to WASM', 'info');
            }
            addLog(
                `Loaded ${separator.backend}/${separator.precision} in ${elapsed}s (${separator.sources.join(', ')})`,
                'success'
            );

            setState(prev => ({
                ...prev,
                modelLoading: false,
                modelLoaded: true,
                progress: 0,
                progressDeterminate: false,
                progressPhase: 'idle',
            }));
            return true;
        } catch (err) {
            if (controller.signal.aborted || isAbortError(err)) {
                if (mountedRef.current) {
                    setState(prev => ({
                        ...prev,
                        modelLoading: false,
                        modelLoaded: false,
                        progress: 0,
                        progressDeterminate: false,
                        progressPhase: 'idle',
                    }));
                }
                return false;
            }
            if (!mountedRef.current) return false;
            const detail = err instanceof Error ? err.message : String(err);
            const message = `Failed to load ${model}: ${detail}`;
            setLoadedModel(null);
            addLog(message, 'error');
            setAudioError(message);
            setState(prev => ({
                ...prev,
                modelLoading: false,
                modelLoaded: false,
                progress: 0,
                progressDeterminate: false,
                progressPhase: 'idle',
            }));
            return false;
        } finally {
            if (modelLoadAbortRef.current === controller) {
                modelLoadAbortRef.current = null;
                modelLoadInFlightRef.current = false;
            }
        }
    }, [addLog, setStatus, setProgress]);

    const clearAudioError = useCallback(() => {
        setAudioError(null);
    }, []);

    const loadAudio = useCallback(async (file: File): Promise<boolean> => {
        // Swapping tracks mid-separation would publish the old track's stems
        // under the new track's metadata when the run finishes.
        if (separateInFlightRef.current) {
            const message = 'Cannot load a new track while separation is in progress';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        // Two racing decodes would interleave their state writes (and leak
        // the loser's artwork URL).
        if (loadAudioInFlightRef.current) {
            const message = 'A track is already loading';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        loadAudioInFlightRef.current = true;
        setLogs([]);
        setProgress(0, false, 'audio');
        try {
            // Revoke object URLs from the previous track before it is replaced.
            if (originalUrlRef.current) {
                URL.revokeObjectURL(originalUrlRef.current);
            }
            originalUrlRef.current = null;
            setOriginalUrl(null);
            Object.values(stemUrlsRef.current).forEach(url => URL.revokeObjectURL(url));
            stemUrlsRef.current = {};
            setStemUrls({});
            setStemPeaks({});
            if (artworkUrlRef.current) {
                URL.revokeObjectURL(artworkUrlRef.current);
            }
            artworkUrlRef.current = null;
            setArtworkUrl(null);
            // Clear every previous-track reference up front. If the replacement
            // fails, the hook consistently reports no loaded audio rather than
            // exposing old audio under the new operation's error.
            audioBufferRef.current = null;
            setState(prev => ({
                ...prev,
                audioLoaded: false,
                audioBuffer: null,
                audioFile: null,
            }));
            setTrackTitle(null);
            setTrackArtist(null);

            setAudioError(null);
            addLog(`Loading audio: ${file.name}`, 'info');
            const ctx = getAudioContext();

            const { buffer: audioBuffer, artwork, title, artist, usedFallback } = await decodeAudioFile(
                file,
                ctx,
                s => { if (mountedRef.current) setStatus(s); }
            );
            if (!mountedRef.current) {
                if (artwork) URL.revokeObjectURL(artwork);
                return false;
            }

            if (usedFallback === 'ffmpeg') {
                addLog('Audio decoded using fallback decoder (ffmpeg.wasm)', 'info');
            } else {
                addLog('Audio decoded with Mediabunny', 'info');
            }

            // Store artwork if present
            if (artwork) {
                artworkUrlRef.current = artwork;
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

            const sourceUrl = URL.createObjectURL(file);
            originalUrlRef.current = sourceUrl;
            setOriginalUrl(sourceUrl);
            audioBufferRef.current = audioBuffer;
            setState(prev => ({
                ...prev,
                audioLoaded: true,
                audioBuffer,
                audioFile: file,
            }));
            return true;
        } catch (error) {
            if (!mountedRef.current) return false;
            const errorMessage = (error as Error).message;
            addLog(`Failed to load audio: ${errorMessage}`, 'error');
            setAudioError(errorMessage);
            return false;
        } finally {
            loadAudioInFlightRef.current = false;
        }
    }, [addLog, getAudioContext, setProgress, setStatus]);

    const separateAudio = useCallback(async (): Promise<boolean> => {
        const separator = separatorRef.current;
        if (!separator) {
            const message = 'Model not loaded';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        const audioBuffer = audioBufferRef.current;
        if (!audioBuffer) {
            const message = 'Audio not loaded';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        // The library documents concurrent separate() calls on one instance
        // as unsafe; guard like loadModel does.
        if (separateInFlightRef.current) {
            const message = 'Separation already in progress';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        // Separating while a new track decodes would publish the old track's
        // stems under the new track's metadata when the decode resolves.
        if (loadAudioInFlightRef.current) {
            const message = 'Cannot separate while a track is still loading';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        // And mid-model-swap the current separator is being torn down.
        if (modelLoadInFlightRef.current) {
            const message = 'Cannot separate while a model is loading';
            addLog(message, 'error');
            setAudioError(message);
            return false;
        }
        const controller = new AbortController();
        separationAbortRef.current = controller;
        separateInFlightRef.current = true;
        setAudioError(null);
        let localUrls: string[] = [];

        try {
            setState(prev => ({ ...prev, separating: true }));
            // Revoke the previous run's object URLs before dropping them.
            Object.values(stemUrlsRef.current).forEach(url => URL.revokeObjectURL(url));
            stemUrlsRef.current = {};
            setStemUrls({});
            setStatus('Preparing audio...');
            setProgress(0, false, 'audio');

            // Yield once so React paints the "separating" UI before the
            // pipeline starts hammering the main thread.
            await new Promise(resolve => setTimeout(resolve, 0));
            addLog('Starting separation...', 'info');

            let segmentStartedAtMs = performance.now();
            let segmentExpectedMs = DEFAULT_SEGMENT_MS[separator.model];
            const result = await separator.separate(audioBuffer, {
                signal: controller.signal,
                onProgress: ({ stage, segIdx, totalSegs, fraction }) => {
                    if (!mountedRef.current) return;
                    const now = performance.now();
                    if (stage === 'started') {
                        segmentStartedAtMs = now;
                    } else {
                        const elapsed = Math.max(1, now - segmentStartedAtMs);
                        // Adapt to this device without letting one slow or fast
                        // segment make the next estimate wildly optimistic.
                        segmentExpectedMs = Math.max(
                            250,
                            Math.min(60_000, segmentExpectedMs * 0.55 + elapsed * 0.45),
                        );
                    }
                    setStatus(stage === 'started'
                        ? `Separating segment ${segIdx} of ${totalSegs}...`
                        : `Completed segment ${segIdx} of ${totalSegs}.`);
                    setState(prev => ({
                        ...prev,
                        progress: fraction * 100,
                        progressDeterminate: true,
                        progressPhase: 'separate',
                        segmentsDone: Math.round(fraction * totalSegs),
                        segmentsTotal: totalSegs,
                        segmentStartedAtMs,
                        segmentExpectedMs,
                    }));
                },
            });
            if (!mountedRef.current) return false;

            // Build blob URLs for the player UI.
            setStatus('Finalizing...');
            setProgress(0, false, 'finalize');
            // Let React paint the finalizing phase before starting the worker.
            await new Promise(resolve => setTimeout(resolve, 0));

            const urls: Record<string, string> = {};
            const peaks: Record<string, number[]> = {};
            const finalized = await finalizeStems(
                result.stems,
                SAMPLE_RATE,
                controller.signal,
                (done, total, source) => {
                    if (mountedRef.current) {
                        setStatus(`Finalized ${source} (${done} of ${total})...`);
                    }
                },
            );

            for (const stem of finalized) {
                urls[stem.source] = URL.createObjectURL(stem.blob);
                const source = stem.source;
                localUrls.push(urls[source]);
                peaks[source] = stem.peaks;
            }

            // Transfer URL ownership to the cleanup ref synchronously before
            // publishing React state; an unmount cannot fall into an effect gap.
            stemUrlsRef.current = urls;
            setStemUrls(urls);
            setStemPeaks(peaks);
            localUrls = [];

            setStatus('Complete!');
            setProgress(100, true, 'complete');
            addLog(`Finished separation in ${(result.wallMs / 1000).toFixed(2)}s.`, 'success');
            setState(prev => ({ ...prev, separating: false }));
            return true;
        } catch (error) {
            localUrls.forEach(url => URL.revokeObjectURL(url));
            // Any failed worker-backed run permanently invalidates the library
            // Separator. Detach exactly the instance this call used so a future
            // load cannot be clobbered by a late catch/finally from this run.
            if (separatorRef.current === separator) {
                separatorRef.current = null;
                setLoadedModel(null);
            }
            await separator.unload();
            if (controller.signal.aborted || isAbortError(error)) {
                if (mountedRef.current) {
                    setStatus('Separation cancelled');
                    setProgress(0, false, 'idle');
                    setState(prev => ({
                        ...prev,
                        modelLoaded: false,
                        modelLoading: false,
                        separating: false,
                    }));
                }
                return false;
            }
            if (!mountedRef.current) return false;
            const detail = error instanceof Error ? error.message : String(error);
            const message = `Separation failed: ${detail}`;
            addLog(message, 'error');
            setAudioError(message);
            setStatus('Error during separation');
            setProgress(0, false, 'idle');
            setState(prev => ({
                ...prev,
                modelLoaded: false,
                modelLoading: false,
                separating: false,
            }));
            return false;
        } finally {
            if (separationAbortRef.current === controller) {
                separationAbortRef.current = null;
                separateInFlightRef.current = false;
            }
        }
    }, [addLog, setStatus, setProgress]);

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
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            // Abort first so in-flight load/separation promises reject promptly;
            // then unload any fully constructed Separator still owned here.
            modelLoadAbortRef.current?.abort();
            separationAbortRef.current?.abort();
            if (separatorRef.current) {
                void separatorRef.current.unload();
                separatorRef.current = null;
            }
            if (audioContextRef.current) {
                void audioContextRef.current.close();
                audioContextRef.current = null;
            }
            if (originalUrlRef.current) URL.revokeObjectURL(originalUrlRef.current);
            Object.values(stemUrlsRef.current).forEach(url => URL.revokeObjectURL(url));
            if (artworkUrlRef.current) URL.revokeObjectURL(artworkUrlRef.current);
        };
    }, []);

    return {
        ...state,
        logs,
        originalUrl,
        stemUrls,
        stemPeaks,
        artworkUrl,
        trackTitle,
        trackArtist,
        audioError,
        loadedModel,
        loadModel,
        loadAudio,
        clearAudioError,
        clearLogs,
        separateAudio,
    };
}
