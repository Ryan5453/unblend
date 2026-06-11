/** Hidden ``/benchmark`` page: runs MUSDB18-HQ through the same pipeline as the public app and reports SDR. */

import { useCallback, useMemo, useRef, useState } from 'react';

import {
    MUSDB_STEMS,
    pickMusdbDirectory,
    readTracksFromDirectory,
    readTracksFromFileList,
    supportsDirectoryPicker,
    type MusdbStem,
    type MusdbTrack,
} from '../../utils/musdb-loader';
import { decodeAudioFile } from '../../utils/audio-decoder';
import {
    Separator,
    SAMPLE_RATE,
    type ModelType,
    type ModelPrecision,
} from 'demucs-next';
import { computeSDRAsync, meanFiniteSDR, type StemSDR } from '../../utils/sdr';
import { ORT_WASM_PATHS } from '../../onnx-config';

type Phase = 'idle' | 'loading_model' | 'running' | 'done' | 'error';

interface TrackResult {
    name: string;
    durationSec: number;
    wallSec: number;
    inferenceSec: number;
    realtime: number;
    meanSDR: number;
    stemSDR: StemSDR;
    error?: string;
}

interface AggregateStats {
    okCount: number;
    errorCount: number;
    meanWall: number;
    medianWall: number;
    meanSDR: number;
    medianSDR: number;
    perStemMean: StemSDR;
}

/**
 * Native ``decodeAudioData`` for plain WAV — much faster than the public
 * app's MediaBunny pipeline. Falls back if the native decoder rejects.
 */
async function decodeWavFast(file: File, audioContext: AudioContext): Promise<AudioBuffer> {
    const arrayBuffer = await file.arrayBuffer();
    try {
        return await audioContext.decodeAudioData(arrayBuffer.slice(0));
    } catch {
        return (await decodeAudioFile(file, audioContext)).buffer;
    }
}

function audioBufferToInterleaved(buffer: AudioBuffer): Float32Array {
    const numChannels = 2;
    const numSamples = buffer.length;
    const interleaved = new Float32Array(numSamples * numChannels);
    const left = buffer.getChannelData(0);
    const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
    for (let i = 0; i < numSamples; i++) {
        interleaved[i * 2] = left[i];
        interleaved[i * 2 + 1] = right[i];
    }
    return interleaved;
}

function aggregate(results: TrackResult[]): AggregateStats {
    const ok = results.filter(r => !r.error);
    const wallTimes = ok.map(r => r.wallSec).sort((a, b) => a - b);
    const meanSDRs = ok.map(r => r.meanSDR).filter(v => Number.isFinite(v)).sort((a, b) => a - b);

    function median(arr: number[]): number {
        if (arr.length === 0) return Number.NaN;
        const mid = Math.floor(arr.length / 2);
        return arr.length % 2 === 0 ? (arr[mid - 1] + arr[mid]) / 2 : arr[mid];
    }

    const perStemMean: StemSDR = {};
    for (const stem of MUSDB_STEMS) {
        const values = ok.map(r => r.stemSDR[stem]).filter(v => Number.isFinite(v));
        perStemMean[stem] = values.length === 0 ? Number.NaN : values.reduce((a, b) => a + b, 0) / values.length;
    }

    return {
        okCount: ok.length,
        errorCount: results.length - ok.length,
        meanWall: wallTimes.length === 0 ? Number.NaN : wallTimes.reduce((a, b) => a + b, 0) / wallTimes.length,
        medianWall: median(wallTimes),
        meanSDR: meanSDRs.length === 0 ? Number.NaN : meanSDRs.reduce((a, b) => a + b, 0) / meanSDRs.length,
        medianSDR: median(meanSDRs),
        perStemMean,
    };
}

function fmt(value: number, digits = 2): string {
    if (!Number.isFinite(value)) return '—';
    return value.toFixed(digits);
}

export function Benchmark() {
    const [phase, setPhase] = useState<Phase>('idle');
    const [model, setModel] = useState<ModelType>('htdemucs');
    const [backend, setBackend] = useState<'webgpu' | 'wasm'>('webgpu');
    const [precision, setPrecision] = useState<ModelPrecision>('fp16');
    const [tracks, setTracks] = useState<MusdbTrack[]>([]);
    const [results, setResults] = useState<TrackResult[]>([]);
    const [currentIndex, setCurrentIndex] = useState<number>(-1);
    const [error, setError] = useState<string | null>(null);
    const [logs, setLogs] = useState<string[]>([]);

    const cancelRef = useRef(false);
    const audioContextRef = useRef<AudioContext | null>(null);

    const log = useCallback((msg: string) => {
        const stamp = new Date().toLocaleTimeString();
        setLogs(prev => [...prev.slice(-99), `[${stamp}] ${msg}`]);
    }, []);

    const getAudioContext = useCallback(() => {
        if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
        }
        return audioContextRef.current;
    }, []);

    const handleDirectoryPick = useCallback(async () => {
        try {
            const handle = await pickMusdbDirectory();
            log(`Scanning ${handle.name}…`);
            const found = await readTracksFromDirectory(handle);
            log(`Found ${found.length} tracks`);
            setTracks(found);
            setResults([]);
            setError(null);
        } catch (err) {
            const e = err as Error;
            if (e.name !== 'AbortError') {
                setError(`Directory picker failed: ${e.message}`);
            }
        }
    }, [log]);

    const handleDrop = useCallback(async (ev: React.DragEvent<HTMLDivElement>) => {
        ev.preventDefault();
        const dt = ev.dataTransfer;
        if (!dt) return;

        // Try the modern FileSystemAccess interface first.
        if (dt.items && dt.items.length > 0) {
            const handles: File[] = [];
            for (let i = 0; i < dt.items.length; i++) {
                const item = dt.items[i];
                const entry = (item as DataTransferItem & {
                    webkitGetAsEntry?: () => FileSystemEntry | null;
                }).webkitGetAsEntry?.();
                if (entry?.isDirectory) {
                    await collectFiles(entry as FileSystemDirectoryEntry, '', handles);
                } else if (entry?.isFile) {
                    const f = await new Promise<File>((res, rej) => {
                        (entry as FileSystemFileEntry).file(res, rej);
                    });
                    handles.push(f);
                }
            }
            const found = readTracksFromFileList(handles);
            log(`Drop yielded ${found.length} tracks`);
            setTracks(found);
            setResults([]);
            setError(null);
        }
    }, [log]);

    const run = useCallback(async () => {
        if (tracks.length === 0) {
            setError('No tracks loaded');
            return;
        }

        cancelRef.current = false;
        setError(null);
        setResults([]);
        setLogs([]);
        setPhase('loading_model');
        log(`Loading model ${model} (${backend}, ${precision})…`);

        let separator: Separator;
        try {
            separator = await Separator.load(model, {
                backend,
                precision,
                wasmPaths: ORT_WASM_PATHS,
            });
        } catch (err) {
            setError(`Model failed to load: ${(err as Error).message}`);
            setPhase('error');
            return;
        }
        if (backend === 'webgpu' && separator.backend === 'wasm') {
            log('WebGPU unavailable — fell back to wasm');
        }
        log(`Loaded ${separator.backend} (${separator.sources.join(', ')})`);

        setPhase('running');
        const ctx = getAudioContext();
        const localResults: TrackResult[] = [];

        for (let i = 0; i < tracks.length; i++) {
            if (cancelRef.current) {
                log('Cancelled');
                break;
            }

            const track = tracks[i];
            setCurrentIndex(i);
            log(`[${i + 1}/${tracks.length}] ${track.name}`);

            try {
                const mixBuffer = await decodeWavFast(track.mixture, ctx);
                const durationSec = mixBuffer.length / mixBuffer.sampleRate;
                const sep = await separator.separate(mixBuffer);

                const stemSDR: StemSDR = {};
                for (const stem of MUSDB_STEMS) {
                    const file = track.stems[stem as MusdbStem];
                    const estimate = sep.stems[stem];
                    if (!file || !estimate) {
                        stemSDR[stem] = Number.NaN;
                        continue;
                    }
                    const refBuffer = await decodeWavFast(file, ctx);
                    const refInterleaved = audioBufferToInterleaved(refBuffer);
                    stemSDR[stem] = await computeSDRAsync(estimate, refInterleaved);
                    delete sep.stems[stem];
                }

                const meanSDR = meanFiniteSDR(stemSDR);

                const result: TrackResult = {
                    name: track.name,
                    durationSec,
                    wallSec: sep.wallMs / 1000,
                    inferenceSec: sep.inferenceMs / 1000,
                    realtime: durationSec / (sep.wallMs / 1000),
                    meanSDR,
                    stemSDR,
                };

                localResults.push(result);
                setResults([...localResults]);
                log(
                    `  ${fmt(sep.wallMs / 1000, 2)}s wall, ${fmt(durationSec / (sep.wallMs / 1000), 1)}× realtime, mean SDR ${fmt(meanSDR, 3)}`
                );
            } catch (err) {
                const e = err as Error;
                console.error(`[bench] track ${i + 1} failed:`, e);
                const result: TrackResult = {
                    name: track.name,
                    durationSec: 0,
                    wallSec: 0,
                    inferenceSec: 0,
                    realtime: 0,
                    meanSDR: Number.NaN,
                    stemSDR: {},
                    error: e.message,
                };
                localResults.push(result);
                setResults([...localResults]);
                log(`  failed: ${e.message}`);
                // ONNX/WebGPU session is often wedged after a failure; bail.
                log('Aborting batch after failure (WebGPU/ORT may be wedged).');
                break;
            }
        }

        await separator.unload();
        setCurrentIndex(-1);
        setPhase('done');
        log('Benchmark complete');
    }, [tracks, model, backend, precision, getAudioContext, log]);

    const cancel = useCallback(() => {
        cancelRef.current = true;
    }, []);

    const stats = useMemo(() => aggregate(results), [results]);

    const canRun = tracks.length > 0 && phase !== 'running' && phase !== 'loading_model';

    return (
        <div className="w-full max-w-6xl mx-auto px-6 py-8 flex-1">
            <div className="content-card">
                <h1 className="content-title">Benchmark</h1>

                <div className="content-body">
                    <p>
                        Point this at a copy of the{' '}
                        <a href="https://zenodo.org/records/3338373" target="_blank" rel="noreferrer">
                            MUSDB18-HQ
                        </a>{' '}
                        test split (each subfolder containing <code>mixture.wav</code> plus four reference stems) and it
                        will run every track through the same in-browser separation pipeline the public app uses,
                        recording per-track wall time, ONNX inference time, realtime factor, and SDR per stem.
                    </p>

                    <div className="my-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                        <label className="flex flex-col gap-2">
                            <span className="bench-field-label">Model</span>
                            <select
                                className="bench-select"
                                value={model}
                                onChange={e => setModel(e.target.value as ModelType)}
                                disabled={phase === 'running' || phase === 'loading_model'}
                            >
                                <option value="htdemucs">htdemucs (4 stems)</option>
                                <option value="htdemucs_6s">htdemucs_6s (6 stems, experimental)</option>
                            </select>
                        </label>

                        <label className="flex flex-col gap-2">
                            <span className="bench-field-label">Backend</span>
                            <select
                                className="bench-select"
                                value={backend}
                                onChange={e => setBackend(e.target.value as 'webgpu' | 'wasm')}
                                disabled={phase === 'running' || phase === 'loading_model'}
                            >
                                <option value="webgpu">WebGPU</option>
                                <option value="wasm">WebAssembly (CPU)</option>
                            </select>
                        </label>

                        <label className="flex flex-col gap-2">
                            <span className="bench-field-label">Precision</span>
                            <select
                                className="bench-select"
                                value={precision}
                                onChange={e => setPrecision(e.target.value as ModelPrecision)}
                                disabled={phase === 'running' || phase === 'loading_model'}
                            >
                                <option value="fp16">fp16 (half)</option>
                                <option value="fp32">fp32 (full)</option>
                            </select>
                        </label>
                    </div>

                    <div
                        className={`bench-dropzone my-4${tracks.length > 0 ? ' is-loaded' : ''}`}
                        onDragOver={e => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {tracks.length > 0 ? (
                            <p className="mb-3">
                                <span className="bench-dropzone-check">✓</span>{' '}
                                <strong>{tracks.length}</strong> track{tracks.length === 1 ? '' : 's'} loaded — drop a different folder or pick again to replace.
                            </p>
                        ) : (
                            <p className="mb-3">
                                Drop a folder containing per-track subdirectories, or pick one below.
                            </p>
                        )}
                        <div className="flex justify-center gap-3 flex-wrap">
                            {supportsDirectoryPicker() && (
                                <button
                                    type="button"
                                    onClick={handleDirectoryPick}
                                    className="bench-btn bench-btn-primary"
                                    disabled={phase === 'running' || phase === 'loading_model'}
                                >
                                    {tracks.length > 0 ? 'Pick a different directory…' : 'Pick directory…'}
                                </button>
                            )}
                        </div>
                    </div>

                    <div className="flex gap-3 my-4 items-center">
                        <button
                            type="button"
                            onClick={run}
                            disabled={!canRun}
                            className="bench-btn bench-btn-primary"
                        >
                            {phase === 'running' ? 'Running…' : phase === 'loading_model' ? 'Loading model…' : 'Run benchmark'}
                        </button>
                        {phase === 'running' && (
                            <button
                                type="button"
                                onClick={cancel}
                                className="bench-btn bench-btn-danger"
                            >
                                Cancel
                            </button>
                        )}
                    </div>

                    {error && (
                        <div className="bench-error my-4">
                            {error}
                        </div>
                    )}

                    {(phase === 'running' || phase === 'loading_model') && (
                        <div className="bench-banner my-4">
                            {currentIndex >= 0
                                ? `Track ${currentIndex + 1} of ${tracks.length}: ${tracks[currentIndex]?.name}`
                                : 'Preparing…'}
                        </div>
                    )}

                    {results.length > 0 && (
                        <>
                            <h2>Aggregate</h2>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 my-3">
                                <Stat label="OK / Total" value={`${stats.okCount} / ${results.length}`} />
                                <Stat label="Mean s/track" value={fmt(stats.meanWall, 2)} />
                                <Stat label="Median s/track" value={fmt(stats.medianWall, 2)} />
                                <Stat label="Mean SDR" value={fmt(stats.meanSDR, 3)} />
                                {MUSDB_STEMS.map(stem => (
                                    <Stat key={stem} label={`${stem} SDR`} value={fmt(stats.perStemMean[stem], 3)} />
                                ))}
                            </div>

                            <h2>Per-track results</h2>
                            <div className="overflow-x-auto my-3">
                                <table className="bench-table">
                                    <thead>
                                        <tr>
                                            <th>Track</th>
                                            <th>Duration</th>
                                            <th>Wall</th>
                                            <th>Realtime</th>
                                            <th>Mean SDR</th>
                                            {MUSDB_STEMS.map(stem => (
                                                <th key={stem}>{stem}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {results.map((r, i) => (
                                            <tr key={`${r.name}-${i}`} className={r.error ? 'has-error' : ''}>
                                                <td>{r.name}</td>
                                                <td>{fmt(r.durationSec, 1)}s</td>
                                                <td>{fmt(r.wallSec, 2)}s</td>
                                                <td>{fmt(r.realtime, 1)}×</td>
                                                <td style={{ fontWeight: 600 }}>{fmt(r.meanSDR, 3)}</td>
                                                {MUSDB_STEMS.map(stem => (
                                                    <td key={stem}>{fmt(r.stemSDR[stem], 3)}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </>
                    )}

                    {logs.length > 0 && (
                        <details className="my-4">
                            <summary style={{ cursor: 'pointer', color: '#888', fontSize: '0.875rem' }}>Logs</summary>
                            <pre className="bench-logs mt-2">{logs.join('\n')}</pre>
                        </details>
                    )}
                </div>
            </div>
        </div>
    );
}

function Stat({ label, value }: { label: string; value: string }) {
    return (
        <div className="bench-stat">
            <div className="bench-stat-label">{label}</div>
            <div className="bench-stat-value">{value}</div>
        </div>
    );
}

/** Recursively walk a dropped directory entry, collecting all files. */
async function collectFiles(
    dir: FileSystemDirectoryEntry,
    prefix: string,
    out: File[]
): Promise<void> {
    const reader = dir.createReader();
    const entries: FileSystemEntry[] = [];

    // ``readEntries`` returns at most 100 entries per call.
    for (;;) {
        const batch: FileSystemEntry[] = await new Promise((res, rej) => {
            reader.readEntries(res as (e: FileSystemEntry[]) => void, rej);
        });
        if (batch.length === 0) break;
        entries.push(...batch);
    }

    for (const entry of entries) {
        const childPath = prefix ? `${prefix}/${entry.name}` : entry.name;
        if (entry.isDirectory) {
            await collectFiles(entry as FileSystemDirectoryEntry, childPath, out);
        } else if (entry.isFile) {
            const f: File = await new Promise((res, rej) => {
                (entry as FileSystemFileEntry).file(res, rej);
            });
            // Synthesize a relative path so ``readTracksFromFileList`` can
            // group by track.
            Object.defineProperty(f, 'webkitRelativePath', { value: childPath, writable: false });
            out.push(f);
        }
    }
}
