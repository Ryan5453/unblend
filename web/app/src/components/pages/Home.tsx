import { useState, useRef, useEffect, useMemo, useCallback, type DragEvent } from 'react';
import { useDemucs } from '../../hooks/useDemucs';
import { useHomeReset } from '../home-reset';
import { WaveCanvas, RulerCanvas } from '../ui/WaveLane';
import { Braid } from '../ui/Braid';
import { peaksFromBuffer } from '../../utils/peaks';
import { makeZip } from '../../utils/zip';

const INK = '25,25,22';
const RED = '207,59,23';
const ORIGINAL = '__original';

interface StemMeta {
    name: string;
    sub: string;
}

const STEM_META: Record<string, StemMeta> = {
    vocals: { name: 'VOCALS', sub: 'LEAD + HARMONY' },
    drums: { name: 'DRUMS', sub: 'KIT + PERCUSSION' },
    bass: { name: 'BASS', sub: 'LOW END · 20–250 HZ' },
    other: { name: 'OTHER', sub: 'SYNTH · GTR · FX' },
    guitar: { name: 'GUITAR', sub: '6-STRING' },
    piano: { name: 'PIANO', sub: 'KEYS' },
};

const STEM_ORDER = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];

const fmtTenths = (t: number) =>
    `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(1).padStart(4, '0')}`;

type Phase = 'drop' | 'processing' | 'studio';

interface Lane {
    key: string;
    idx: string;
    name: string;
    sub: string;
    height: number;
    peaks: number[];
    download: boolean;
}

export function Home() {
    const {
        modelLoaded,
        modelLoading,
        separating,
        progress,
        status,
        audioBuffer,
        audioFile,
        originalUrl,
        stemUrls,
        stemPeaks,
        trackTitle,
        trackArtist,
        artworkUrl,
        logs,
        audioError,
        loadModel,
        loadAudio,
        clearAudioError,
        separateAudio,
    } = useDemucs();

    const fileInputRef = useRef<HTMLInputElement>(null);
    const [phase, setPhase] = useState<Phase>('drop');
    const [isDragging, setIsDragging] = useState(false);
    // Drag enter/leave events fire for every child element; track depth so
    // the drag state only clears when the pointer truly leaves the page.
    const dragDepth = useRef(0);
    const [displayPct, setDisplayPct] = useState(0);

    const [volumes, setVolumes] = useState<Record<string, number>>({});
    const [muted, setMuted] = useState<Record<string, boolean>>({ [ORIGINAL]: true });
    const [solo, setSolo] = useState<Record<string, boolean>>({});
    const [master, setMaster] = useState(100);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [exportLabel, setExportLabel] = useState('EXPORT .ZIP ↓');

    const audioRefs = useRef<Record<string, HTMLAudioElement>>({});
    // The displayPct value captured when separation begins, so the separation
    // phase can continue upward from the model-load crawl without jumping.
    const sepBaseRef = useRef<number | null>(null);
    const duration = audioBuffer?.duration ?? 0;
    const progressFrac = duration > 0 ? Math.min(1, currentTime / duration) : 0;

    const originalPeaks = useMemo(
        () => (audioBuffer ? peaksFromBuffer(audioBuffer) : []),
        [audioBuffer]
    );

    const stemKeys = useMemo(() => {
        const keys = Object.keys(stemUrls);
        return keys.sort((a, b) => {
            const ia = STEM_ORDER.indexOf(a);
            const ib = STEM_ORDER.indexOf(b);
            return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
        });
    }, [stemUrls]);

    const lanes: Lane[] = useMemo(() => {
        const list: Lane[] = [
            {
                key: ORIGINAL,
                idx: '01',
                name: 'ORIGINAL',
                sub: 'SOURCE MIX',
                height: 58,
                peaks: originalPeaks,
                download: false,
            },
        ];
        stemKeys.forEach((key, i) => {
            const meta = STEM_META[key] ?? { name: key.toUpperCase(), sub: 'STEM' };
            list.push({
                key,
                idx: String(i + 2).padStart(2, '0'),
                name: meta.name,
                sub: meta.sub,
                height: 78,
                peaks: stemPeaks[key] ?? [],
                download: true,
            });
        });
        return list;
    }, [stemKeys, originalPeaks, stemPeaks]);

    const laneKeys = useMemo(() => lanes.map(l => l.key), [lanes]);
    const clockKey = stemKeys[0];

    const anySolo = lanes.some(l => solo[l.key] && !muted[l.key]);

    const getVolume = useCallback(
        (key: string) => {
            if (muted[key]) return 0;
            if (anySolo && !solo[key]) return 0;
            const base = (volumes[key] ?? 90) / 100;
            return base * (master / 100);
        },
        [muted, solo, volumes, master, anySolo]
    );

    // Apply computed volumes to the live audio elements.
    useEffect(() => {
        laneKeys.forEach(key => {
            const a = audioRefs.current[key];
            if (a) a.volume = getVolume(key);
        });
    }, [laneKeys, getVolume]);

    // Reset transport/mix state when a fresh set of stems arrives. Done during
    // render (React's reset-on-input-change pattern) rather than in an effect.
    // The processing -> studio switch itself is handled by the delayed effect
    // below so the counter can settle on 100% first.
    const [prevStems, setPrevStems] = useState(stemUrls);
    if (prevStems !== stemUrls) {
        setPrevStems(stemUrls);
        if (Object.keys(stemUrls).length > 0) {
            setVolumes({});
            setMuted({ [ORIGINAL]: true });
            setSolo({});
            setMaster(100);
            setCurrentTime(0);
            setIsPlaying(false);
        }
    }

    // Once stems are ready, hold on 100% briefly, then reveal the studio.
    const stemsReady = Object.keys(stemUrls).length > 0;
    useEffect(() => {
        if (phase !== 'processing' || !stemsReady) return;
        const id = setTimeout(() => setPhase('studio'), 550);
        return () => clearTimeout(id);
    }, [phase, stemsReady]);

    // Drive the currentTime clock while playing off a real stem element.
    useEffect(() => {
        if (!isPlaying) return;
        let raf = 0;
        const loop = () => {
            const a = clockKey ? audioRefs.current[clockKey] : undefined;
            if (a) setCurrentTime(a.currentTime);
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [isPlaying, clockKey]);

    // Drive the big processing counter. Model download reports real byte
    // progress, mapped across the entire 0-30 pre-separation window; the
    // compile step after it has no progress signal of its own, but by then
    // the download has already carried the number to (near) 30, so there's
    // only a small gap left to crawl. Separation then continues smoothly
    // from wherever that left off (anchored by sepBaseRef) up to 100 — no
    // backward jumps, always lands on 100.
    useEffect(() => {
        if (phase !== 'processing') return;
        let raf = 0;
        const loop = () => {
            setDisplayPct(prev => {
                let target: number;
                let factor = 0.12;
                if (stemsReady) {
                    target = 100;
                } else if (separating) {
                    if (sepBaseRef.current === null) sepBaseRef.current = prev;
                    const base = sepBaseRef.current;
                    target = base + (progress / 100) * (100 - base);
                } else if (modelLoading && progress > 0) {
                    target = (progress / 100) * 30;
                    factor = 0.25;
                } else {
                    // Audio decode, download-start latency, or an unknown
                    // download size (no Content-Length): indeterminate,
                    // decelerating crawl.
                    target = 30;
                    factor = 0.03;
                }
                let next = prev + (target - prev) * factor;
                if (next < prev) next = prev; // never count down
                return Math.abs(target - next) < 0.3 ? target : next;
            });
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [phase, modelLoading, separating, progress, stemsReady]);

    // ---- transport ----------------------------------------------------
    const playAll = useCallback(async () => {
        const t = currentTime;
        await Promise.all(
            laneKeys.map(async key => {
                const a = audioRefs.current[key];
                if (!a) return;
                a.currentTime = t;
                a.volume = getVolume(key);
                try {
                    await a.play();
                } catch {
                    /* ignore autoplay rejections */
                }
            })
        );
        setIsPlaying(true);
    }, [currentTime, laneKeys, getVolume]);

    const pauseAll = useCallback(() => {
        laneKeys.forEach(key => audioRefs.current[key]?.pause());
        setIsPlaying(false);
    }, [laneKeys]);

    const resetTransport = useCallback(() => {
        laneKeys.forEach(key => {
            const a = audioRefs.current[key];
            if (a) a.currentTime = 0;
        });
        setCurrentTime(0);
    }, [laneKeys]);

    const seek = useCallback(
        (fraction: number) => {
            const t = Math.max(0, Math.min(duration, fraction * duration));
            laneKeys.forEach(key => {
                const a = audioRefs.current[key];
                if (a) a.currentTime = t;
            });
            setCurrentTime(t);
        },
        [duration, laneKeys]
    );

    const toggleMute = useCallback((key: string) => {
        setMuted(prev => ({ ...prev, [key]: !prev[key] }));
        setSolo(prev => (prev[key] ? { ...prev, [key]: false } : prev));
    }, []);

    const toggleSolo = useCallback((key: string) => {
        setSolo(prev => ({ ...prev, [key]: !prev[key] }));
    }, []);

    // ---- pipeline -----------------------------------------------------
    const runFile = useCallback(
        async (file: File) => {
            clearAudioError();
            setPhase('processing');
            setDisplayPct(0);
            sepBaseRef.current = null;
            const ok = await loadAudio(file);
            if (!ok) {
                setPhase('drop');
                return;
            }
            if (!modelLoaded) {
                const loaded = await loadModel('htdemucs');
                if (!loaded) {
                    setPhase('drop');
                    return;
                }
            }
            const separated = await separateAudio();
            if (!separated) setPhase('drop');
        },
        [clearAudioError, loadAudio, loadModel, modelLoaded, separateAudio]
    );

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        e.target.value = '';
        if (file) void runFile(file);
    };

    const handleDrop = (e: DragEvent<HTMLElement>) => {
        e.preventDefault();
        dragDepth.current = 0;
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) void runFile(file);
    };

    const handleNewFile = () => {
        pauseAll();
        setCurrentTime(0);
        setIsPlaying(false);
        setPhase('drop');
    };

    // Let the header wordmark / "Studio" link trigger this same soft reset.
    // It reuses handleNewFile, so the loaded model stays warm — no reload.
    const reset = useHomeReset();
    const newFileRef = useRef(handleNewFile);
    useEffect(() => {
        newFileRef.current = handleNewFile;
    });
    useEffect(() => {
        reset?.register(() => newFileRef.current());
        return () => reset?.register(null);
    }, [reset]);

    // ---- keyboard -----------------------------------------------------
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (phase !== 'studio') return;
            const target = e.target as HTMLElement;
            if (/INPUT|TEXTAREA/.test(target.tagName)) return;
            if (e.code === 'Space') {
                e.preventDefault();
                if (isPlaying) pauseAll();
                else void playAll();
                return;
            }
            // Ignore browser / OS chords (⌘ / Ctrl / Alt + key) so we never
            // fight shortcuts like tab or profile switching.
            if (e.metaKey || e.ctrlKey || e.altKey) return;
            // Match the physical digit key via e.code, not e.key: with Shift
            // held most layouts report "!" "@" … for the number row, so the old
            // parseInt(e.key) never saw a digit and the mute chord never fired.
            const digit = e.code.match(/^Digit([1-9])$/);
            if (digit) {
                const n = parseInt(digit[1], 10);
                if (n <= lanes.length) {
                    e.preventDefault();
                    const key = lanes[n - 1].key;
                    if (e.shiftKey) toggleMute(key);
                    else toggleSolo(key);
                }
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [phase, isPlaying, lanes, pauseAll, playAll, toggleMute, toggleSolo]);

    // ---- downloads / export ------------------------------------------
    const downloadUrl = (url: string, filename: string) => {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        a.remove();
    };

    const handleExport = async () => {
        const targets = stemKeys.filter(k => stemUrls[k]);
        if (targets.length === 0) return;
        setExportLabel(`PACKING ${targets.length} STEMS…`);
        try {
            const entries = await Promise.all(
                targets.map(async key => {
                    const res = await fetch(stemUrls[key]);
                    const buf = new Uint8Array(await res.arrayBuffer());
                    return { name: `${key}.wav`, data: buf };
                })
            );
            const zip = makeZip(entries);
            const url = URL.createObjectURL(zip);
            downloadUrl(url, 'stems.zip');
            setTimeout(() => URL.revokeObjectURL(url), 2000);
            setExportLabel('STEMS.ZIP ↓');
        } catch {
            setExportLabel('EXPORT FAILED');
        }
        setTimeout(() => setExportLabel('EXPORT .ZIP ↓'), 2600);
    };

    const laneColors = (key: string): [string, string] => {
        if (muted[key]) return [`rgba(${INK},.12)`, `rgba(${INK},.12)`];
        if (anySolo) {
            return solo[key]
                ? [`rgba(${RED},1)`, `rgba(${RED},.28)`]
                : [`rgba(${INK},.1)`, `rgba(${INK},.1)`];
        }
        return [`rgba(${INK},.92)`, `rgba(${INK},.26)`];
    };

    const dbLabel = (key: string) => {
        const g = (volumes[key] ?? 90) / 100;
        if (g <= 0) return '-∞ dB';
        return `${(20 * Math.log10(g)).toFixed(1)} dB`;
    };

    const trackName = trackTitle || audioFile?.name?.replace(/\.[^/.]+$/, '') || 'untitled';
    const modelName = 'HTDEMUCS';

    return (
        <>
            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
            />

            {/* Hidden audio elements — one per lane */}
            {originalUrl && (
                <audio
                    ref={el => {
                        if (el) audioRefs.current[ORIGINAL] = el;
                        else delete audioRefs.current[ORIGINAL];
                    }}
                    src={originalUrl}
                    onEnded={() => setIsPlaying(false)}
                />
            )}
            {stemKeys.map(key => (
                <audio
                    key={key}
                    ref={el => {
                        if (el) audioRefs.current[key] = el;
                        else delete audioRefs.current[key];
                    }}
                    src={stemUrls[key]}
                    onEnded={() => setIsPlaying(false)}
                />
            ))}

            {/* ---------- DROP ---------- */}
            {phase === 'drop' && (
                <section
                    className="view-drop animate-fade-in"
                    onDragOver={e => e.preventDefault()}
                    onDragEnter={e => {
                        e.preventDefault();
                        dragDepth.current += 1;
                        setIsDragging(true);
                    }}
                    onDragLeave={e => {
                        e.preventDefault();
                        dragDepth.current -= 1;
                        if (dragDepth.current <= 0) {
                            dragDepth.current = 0;
                            setIsDragging(false);
                        }
                    }}
                    onDrop={handleDrop}
                >
                    <div
                        className={`drop-stage${isDragging ? ' drag' : ''}`}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <Braid active={isDragging} />
                        <div className="drop-caption">
                            <div className="t">Drop a song anywhere</div>
                            <div className="or">
                                or{' '}
                                <button
                                    className="u"
                                    onClick={e => {
                                        e.stopPropagation();
                                        fileInputRef.current?.click();
                                    }}
                                >
                                    browse files →
                                </button>
                            </div>
                        </div>
                    </div>
                </section>
            )}

            {/* ---------- PROCESSING ---------- */}
            {phase === 'processing' && (
                <section className="view-proc animate-fade-in">
                    <div className="proc-stack">
                        <div className="proc-braid">
                            <Braid progress={displayPct} audioBuffer={audioBuffer} />
                        </div>
                        <div className="proc">
                            <div className="pct">{Math.round(displayPct)}</div>
                            <div className="proc-log">
                                {logs.slice(-13).map((line, i) => (
                                    <div key={i}>&gt; {line.toUpperCase()}</div>
                                ))}
                                <div className="cur">&gt; {(status || 'WORKING').toUpperCase()} </div>
                            </div>
                        </div>
                    </div>
                </section>
            )}

            {/* ---------- STUDIO ---------- */}
            {phase === 'studio' && (
                <section className="view-studio animate-fade-in">
                    <div className="specrow">
                        {artworkUrl && <img className="art" src={artworkUrl} alt="" />}
                        <div className="tmeta">
                            <span className="fn">{trackName}</span>
                            {trackArtist && <span className="tartist">{trackArtist}</span>}
                        </div>
                        <span className="dots" />
                        <span className="spec st">{modelName}</span>
                        <span className="spec st">{stemKeys.length} STEMS</span>
                        <button className="spec" onClick={handleNewFile}>
                            NEW FILE ×
                        </button>
                    </div>

                    <div className="board">
                        <div className="brow rulerb">
                            <div className="lab" />
                            <div className="wav">
                                <RulerCanvas progress={progressFrac} duration={duration} onSeek={seek} />
                            </div>
                        </div>

                        {lanes.map(lane => {
                            const isSolo = !!solo[lane.key] && !muted[lane.key];
                            const isDim = !!muted[lane.key] || (anySolo && !solo[lane.key]);
                            const [cp, cf] = laneColors(lane.key);
                            return (
                                <div
                                    key={lane.key}
                                    className={`brow${isSolo ? ' solo-on' : ''}${isDim ? ' dim' : ''}`}
                                >
                                    <div className="lab">
                                        <div className="ltop">
                                            <span className="idx">{lane.idx}</span>
                                            <span className="lname">{lane.name}</span>
                                        </div>
                                        <div className="lsub">{lane.sub}</div>
                                        <div className="lctl">
                                            <button
                                                className={`msq${muted[lane.key] ? ' on-m' : ''}`}
                                                onClick={() => toggleMute(lane.key)}
                                                title="Mute"
                                            >
                                                M
                                            </button>
                                            <button
                                                className={`msq${isSolo ? ' on-s' : ''}`}
                                                onClick={() => toggleSolo(lane.key)}
                                                title="Solo"
                                            >
                                                S
                                            </button>
                                            <input
                                                type="range"
                                                min={0}
                                                max={100}
                                                value={volumes[lane.key] ?? 90}
                                                onChange={e =>
                                                    setVolumes(prev => ({
                                                        ...prev,
                                                        [lane.key]: Number(e.target.value),
                                                    }))
                                                }
                                            />
                                            <span className="db">{dbLabel(lane.key)}</span>
                                            {lane.download && (
                                                <button
                                                    className="dl"
                                                    onClick={() =>
                                                        downloadUrl(stemUrls[lane.key], `${lane.key}.wav`)
                                                    }
                                                    title="Download stem"
                                                >
                                                    ↓
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                    <div className="wav">
                                        <WaveCanvas
                                            peaks={lane.peaks}
                                            height={lane.height}
                                            progress={progressFrac}
                                            duration={duration}
                                            gain={(volumes[lane.key] ?? 90) / 100}
                                            colorPlayed={cp}
                                            colorFuture={cf}
                                            onSeek={seek}
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    <div className="hint">
                        SPACE — PLAY/PAUSE · 1–{lanes.length} — SOLO · SHIFT+1–{lanes.length} — MUTE
                    </div>
                </section>
            )}

            {/* ---------- TRANSPORT ---------- */}
            {phase === 'studio' && (
                <footer className="transport-bar">
                    <button
                        className="play"
                        onClick={() => (isPlaying ? pauseAll() : void playAll())}
                        title="Play / Pause"
                    >
                        {isPlaying ? '❚❚' : '▶'}
                    </button>
                    <div className="fcell">
                        <button className="tsq" onClick={resetTransport} title="Restart">
                            ↺
                        </button>
                    </div>
                    <div className="fcell">
                        <span className="flab">T</span>
                        <span className="tc">
                            {fmtTenths(currentTime)} <em>/ {fmtTenths(duration)}</em>
                        </span>
                    </div>
                    <div className="fcell grow">
                        <span className="flab">VOL</span>
                        <input
                            type="range"
                            min={0}
                            max={100}
                            value={master}
                            onChange={e => setMaster(Number(e.target.value))}
                        />
                    </div>
                    <button className="export" onClick={handleExport}>
                        {exportLabel}
                    </button>
                </footer>
            )}

            {/* ---------- ERROR MODAL ---------- */}
            {audioError && (
                <div className="modal-backdrop" onClick={clearAudioError}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <div className="modal-title">Operation failed</div>
                        <div className="modal-msg">{audioError}</div>
                        <button className="modal-dismiss" onClick={clearAudioError}>
                            Dismiss
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}
