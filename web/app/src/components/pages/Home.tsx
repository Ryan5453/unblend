import {
    useState,
    useRef,
    useEffect,
    useMemo,
    useCallback,
    useSyncExternalStore,
    type DragEvent,
    type KeyboardEvent as ReactKeyboardEvent,
} from 'react';
import { useUnblend } from '../../hooks/useUnblend';
import { useLaneMixer } from '../../hooks/useLaneMixer';
import { useHomeReset } from '../home-reset';
import { WaveCanvas, RulerCanvas } from '../ui/WaveLane';
import { Braid } from '../ui/Braid';
import { peaksFromBuffer } from '../../utils/peaks';
import { makeZip } from '../../utils/zip';
import { createPlayhead, type Playhead } from '../../utils/playhead';
import { MODEL_ARTIFACTS, MODEL_CONFIGS, type ModelType } from 'unblend';
import type { ProgressPhase } from '../../types';

const INK = '25,25,22';
const RED = '189,54,19';
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

interface ModelChoice {
    short: string;
    description: string;
    stems: number;
}

// useUnblend's loadModel defaults to fp16, so that is what the app downloads.
// MiB, matching the download progress readout.
const downloadSize = (model: ModelType) =>
    `${Math.round(MODEL_ARTIFACTS[model].fp16.sizeBytes / (1024 * 1024))} MiB`;

const MODEL_CHOICES: Record<ModelType, ModelChoice> = {
    htdemucs: {
        short: 'HTDEMUCS',
        description: 'FAST, BALANCED FOUR-STEM SPLIT',
        stems: 4,
    },
    htdemucs_6s: {
        short: 'HTDEMUCS 6S',
        description: 'GUITAR + PIANO, EXPERIMENTAL',
        stems: 6,
    },
    bs_roformer_sw: {
        short: 'BS-ROFORMER SW',
        description: 'HIGH-DETAIL MULTI-STEM SEPARATION',
        stems: 6,
    },
    melband_roformer_kim: {
        short: 'MEL-BAND ROFORMER KIM',
        description: 'VOCALS + FULL INSTRUMENTAL',
        stems: 2,
    },
    scnet_small: {
        short: 'SCNET MASKED SMALL',
        description: 'MASKED SMALL MODEL',
        stems: 4,
    },
    scnet_xl_wide_v5: {
        short: 'SCNET XL IHF',
        description: 'WIDE HIGH-FREQUENCY XL MODEL',
        stems: 4,
    },
};

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

interface VisualProgressInput {
    phase: ProgressPhase;
    measured: number;
    determinate: boolean;
    segmentsDone: number;
    segmentsTotal: number;
    segmentStartedAtMs: number;
    segmentExpectedMs: number;
}

/**
 * Smooth presentation without hiding what is measured. Downloads ease toward
 * (and never exceed) received bytes. Separation estimates only the fraction
 * of the currently running opaque ONNX call; the exact segment counter stays
 * visible in the status log.
 */
function useVisualProgress(input: VisualProgressInput): number | null {
    const inputRef = useRef(input);
    // Written after commit rather than during render: the only reader is the
    // interval callback below, which just needs the latest value by the time it
    // fires, and assigning during render trips react-hooks/refs.
    useEffect(() => {
        inputRef.current = input;
    });
    const [visual, setVisual] = useState<{ phase: ProgressPhase; value: number | null }>({
        phase: 'idle',
        value: null,
    });

    // Only animate while loading or separating (plus the final ease to 100
    // on completion). Other phases need a single tick to reset the value, so
    // the interval isn't left running while the page sits idle.
    const animating =
        input.phase === 'download'
        || input.phase === 'cache'
        || input.phase === 'separate'
        || (input.phase === 'complete'
            && !(visual.phase === 'complete' && visual.value === 100));

    useEffect(() => {
        const tick = () => {
            setVisual(previous => {
                const {
                    phase,
                    measured,
                    determinate,
                    segmentsDone,
                    segmentsTotal,
                    segmentStartedAtMs,
                    segmentExpectedMs,
                } = inputRef.current;
                const samePhase = previous.phase === phase;
                // Staying below one point per update prevents visible jumps such
                // as 10 -> 13 when an exact segment-complete event arrives.
                const advanceToward = (current: number, target: number) =>
                    target <= current
                        ? current
                        : Math.min(target, current + 0.9);

                if (phase === 'complete') {
                    const current = previous.value ?? 100;
                    const value = advanceToward(current, 100);
                    return samePhase && value === current ? previous : { phase, value };
                }

                if ((phase === 'download' || phase === 'cache') && determinate) {
                    const target = Math.max(0, Math.min(100, measured));
                    const current = samePhase && previous.value !== null ? previous.value : 0;
                    const value = advanceToward(current, target);
                    return Math.abs(value - current) < 0.005 && samePhase
                        ? previous
                        : { phase, value };
                }

                if (phase === 'separate' && segmentsTotal > 0) {
                    const elapsed = Math.max(0, performance.now() - segmentStartedAtMs);
                    const expected = Math.max(250, segmentExpectedMs || 1_000);
                    const withinSegment = segmentsDone >= segmentsTotal
                        ? 0
                        : Math.min(0.96, 0.96 * (1 - Math.exp(-2.5 * elapsed / expected)));
                    const estimate = Math.min(
                        99.4,
                        ((segmentsDone + withinSegment) / segmentsTotal) * 100,
                    );
                    const current = samePhase && previous.value !== null
                        ? previous.value
                        : measured;
                    const value = advanceToward(current, Math.max(estimate, measured));
                    return Math.abs(value - current) < 0.005 && samePhase
                        ? previous
                        : { phase, value };
                }

                return previous.phase === phase && previous.value === null
                    ? previous
                    : { phase, value: null };
            });
        };

        tick();
        if (!animating) return;
        const id = window.setInterval(tick, 30);
        return () => window.clearInterval(id);
    }, [animating, input.phase]);

    return visual.phase === input.phase ? visual.value : null;
}

/**
 * Make a track name safe to use as a download filename on every major OS:
 * drop reserved, control and bidi characters, collapse whitespace, cut it to
 * fit (see truncateName), and strip leading dots (hidden files) and the
 * trailing dots/spaces Windows rejects.
 */
function sanitizeFileName(name: string): string {
    return cleanFileName(name) || 'untitled';
}

/** sanitizeFileName's work without its fallback: '' when nothing usable is left. */
function cleanFileName(name: string): string {
    return truncateName(
        name
            // Whitespace controls (tab, newline) separate words; keep a space.
            .replace(/[\t\n\v\f\r\u0085]/g, ' ')
            // eslint-disable-next-line no-control-regex
            .replace(/[<>:"/\\|?*\u0000-\u001f\u007f-\u009f]/g, '')
            // Bidi controls and invisible spaces: they only make the name
            // display misleadingly (ZWJ/ZWNJ stay; emoji sequences need them).
            .replace(/\p{Bidi_Control}|[\u200b\u2060\ufeff]/gu, '')
            // Collapsed after the removals, which can leave spaces side by side.
            .replace(/\s+/g, ' ')
            // Leading dots/spaces before the cut, so they can't use up its budget.
            .replace(/^[. ]+/, ''),
    ).replace(/[. ]+$/g, '');
}

/**
 * Cut a name to at most 120 code points and 200 UTF-8 bytes, at whole
 * visible characters (an emoji sequence or an accented letter is kept or
 * dropped entirely). The byte budget leaves room for " - {stem}.wav" within
 * Linux's 255-byte file-name limit, which a CJK title would otherwise exceed.
 */
function truncateName(name: string): string {
    const encoder = new TextEncoder();
    const pieces =
        typeof Intl !== 'undefined' && 'Segmenter' in Intl
            ? Array.from(new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(name), s => s.segment)
            : Array.from(name);
    let out = '';
    let bytes = 0;
    let chars = 0;
    for (const piece of pieces) {
        const size = encoder.encode(piece).length;
        const count = Array.from(piece).length;
        if (chars + count > 120 || bytes + size > 200) break;
        out += piece;
        bytes += size;
        chars += count;
    }
    // A first cluster too long on its own (stacked combining marks) would
    // leave nothing; cut it at code points rather than lose the whole name.
    return out || (pieces.length > 0 ? truncateCodePoints(name, encoder) : '');
}

/** The code-point fallback for truncateName's limits. */
function truncateCodePoints(name: string, encoder: TextEncoder): string {
    let out = '';
    let bytes = 0;
    let chars = 0;
    for (const char of name) {
        const size = encoder.encode(char).length;
        if (chars + 1 > 120 || bytes + size > 200) break;
        out += char;
        bytes += size;
        chars += 1;
    }
    return out;
}

/** Transport time readout; subscribes to the playhead on its own. */
function TimeReadout({ playhead, duration }: { playhead: Playhead; duration: number }) {
    const t = useSyncExternalStore(playhead.subscribe, playhead.get);
    return (
        <span className="tc">
            {fmtTenths(t)} <em>/ {fmtTenths(duration)}</em>
        </span>
    );
}

export function Home() {
    const {
        modelLoaded,
        progressDeterminate,
        progressPhase,
        progress,
        segmentsDone,
        segmentsTotal,
        segmentStartedAtMs,
        segmentExpectedMs,
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
        loadedModel,
        loadModel,
        loadAudio,
        clearAudioError,
        separateAudio,
        clearStems,
        switchOriginalToPcm,
        cancel,
    } = useUnblend();

    const fileInputRef = useRef<HTMLInputElement>(null);
    const modalDismissRef = useRef<HTMLButtonElement>(null);
    const modalRef = useRef<HTMLDivElement>(null);
    // Bumped to cancel a runFile in flight: its later steps see a stale id
    // and stop instead of loading a model or separating.
    const runIdRef = useRef(0);
    const [phase, setPhase] = useState<Phase>('drop');
    const [isDragging, setIsDragging] = useState(false);
    // Drag enter/leave events fire for every child element; track depth so
    // the drag state only clears when the pointer truly leaves the page.
    const dragDepth = useRef(0);
    const [selectedModel, setSelectedModel] = useState<ModelType>('htdemucs');

    const [volumes, setVolumes] = useState<Record<string, number>>({});
    const [muted, setMuted] = useState<Record<string, boolean>>({ [ORIGINAL]: true });
    const [solo, setSolo] = useState<Record<string, boolean>>({});
    const [master, setMaster] = useState(100);
    const [isPlaying, setIsPlaying] = useState(false);
    // Playback position lives outside React state so playback doesn't
    // re-render this whole page every animation frame.
    const [playhead] = useState(createPlayhead);
    const [exportLabel, setExportLabel] = useState('EXPORT .ZIP ↓');

    const audioRefs = useRef<Record<string, HTMLAudioElement>>({});
    const mixer = useLaneMixer();
    const displayPct = useVisualProgress({
        phase: progressPhase,
        measured: progress,
        determinate: progressDeterminate,
        segmentsDone,
        segmentsTotal,
        segmentStartedAtMs,
        segmentExpectedMs,
    });
    const displayPctInteger = displayPct === null ? null : Math.floor(displayPct);
    let progressCaption = 'WORKING';
    if (progressPhase === 'download') progressCaption = 'DOWNLOADED';
    else if (progressPhase === 'cache') progressCaption = 'LOADED';
    else if (progressPhase === 'separate') progressCaption = 'EST. SEPARATED';
    else if (progressPhase === 'finalize') progressCaption = 'FINALIZING';
    else if (progressPhase === 'complete') progressCaption = 'COMPLETE';
    const duration = audioBuffer?.duration ?? 0;

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

    // Mute/solo silence a lane via `.muted`; the fader level is separate so
    // it can go through a GainNode where `.volume` is read-only (iOS).
    const applyMix = useCallback(
        (key: string, a: HTMLAudioElement) => {
            const silenced = !!muted[key] || (anySolo && !solo[key]);
            const level = ((volumes[key] ?? 90) / 100) * (master / 100);
            mixer.apply(a, level, silenced);
        },
        [muted, solo, volumes, master, anySolo, mixer]
    );

    // Apply the mix to the live audio elements.
    useEffect(() => {
        laneKeys.forEach(key => {
            const a = audioRefs.current[key];
            if (a) applyMix(key, a);
        });
    }, [laneKeys, applyMix, originalUrl]);

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
            setIsPlaying(false);
        }
    }
    useEffect(() => {
        if (Object.keys(stemUrls).length > 0) playhead.set(0);
    }, [stemUrls, playhead]);

    // Keep the fixed processing layout visible until the smoothed counter has
    // actually reached 100, then hold briefly before revealing the studio.
    const stemsReady = Object.keys(stemUrls).length > 0;
    useEffect(() => {
        if (
            phase !== 'processing'
            || !stemsReady
            || progressPhase !== 'complete'
            || displayPctInteger !== 100
        ) return;
        const id = setTimeout(() => setPhase('studio'), 350);
        return () => clearTimeout(id);
    }, [displayPctInteger, phase, progressPhase, stemsReady]);

    // Drive the currentTime clock while playing off a real stem element.
    useEffect(() => {
        if (!isPlaying) return;
        let raf = 0;
        const loop = () => {
            const a = clockKey ? audioRefs.current[clockKey] : undefined;
            if (a) playhead.set(a.currentTime);
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [isPlaying, clockKey, playhead]);

    // ---- transport ----------------------------------------------------
    const playAll = useCallback(async () => {
        let t = playhead.get();
        const lanesToPlay = laneKeys
            .map(key => [key, audioRefs.current[key]] as const)
            .filter((entry): entry is readonly [string, HTMLAudioElement] => !!entry[1]);
        // At the end (handleEnded parks every lane there), restart them all
        // from 0 explicitly, rather than relying on each browser restarting
        // an ended element on play() (lanes differ slightly in length).
        // Against the decoded track's length (the transport's), not each
        // element's own duration, which a damaged file can overstate.
        if (lanesToPlay.length > 0 && duration > 0 && t >= duration - 0.01) {
            t = 0;
            playhead.set(0);
        }
        // Synchronously, inside the user gesture: start/resume the shared
        // AudioContext and route any new elements (iOS only).
        mixer.prepare(lanesToPlay.map(([, a]) => a));
        const started = await Promise.all(
            lanesToPlay.map(async ([key, a]) => {
                a.currentTime = t;
                applyMix(key, a);
                try {
                    await a.play();
                    return true;
                } catch {
                    // Autoplay rejection or an unplayable source.
                    return false;
                }
            })
        );
        if (started.some(Boolean)) setIsPlaying(true);
    }, [laneKeys, applyMix, mixer, playhead, duration]);

    // Stems can differ in length by a few samples (and the original is a
    // separately decoded file), so one element ending must not stop the
    // transport while the rest are still playing. Stop only once every lane
    // has finished (or was never playing), then line them all up at the end.
    const handleEnded = useCallback(() => {
        const elements = laneKeys
            .map(key => audioRefs.current[key])
            .filter((a): a is HTMLAudioElement => !!a);
        if (!elements.every(a => a.ended || a.paused)) return;
        const end = Math.max(0, ...elements.map(a => a.currentTime));
        elements.forEach(a => {
            a.pause();
            a.currentTime = end;
        });
        playhead.set(end);
        setIsPlaying(false);
    }, [laneKeys, playhead]);

    const pauseAll = useCallback(() => {
        laneKeys.forEach(key => audioRefs.current[key]?.pause());
        setIsPlaying(false);
    }, [laneKeys]);

    const resetTransport = useCallback(() => {
        laneKeys.forEach(key => {
            const a = audioRefs.current[key];
            if (a) a.currentTime = 0;
        });
        playhead.set(0);
    }, [laneKeys, playhead]);

    const seek = useCallback(
        (fraction: number) => {
            const t = Math.max(0, Math.min(duration, fraction * duration));
            laneKeys.forEach(key => {
                const a = audioRefs.current[key];
                if (a) a.currentTime = t;
            });
            playhead.set(t);
        },
        [duration, laneKeys, playhead]
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
            const runId = ++runIdRef.current;
            const cancelled = () => runIdRef.current !== runId;
            clearAudioError();
            setPhase('processing');
            const ok = await loadAudio(file);
            if (cancelled()) return;
            if (!ok) {
                setPhase('drop');
                return;
            }
            if (!modelLoaded || loadedModel !== selectedModel) {
                const loaded = await loadModel(selectedModel);
                if (cancelled()) return;
                if (!loaded) {
                    setPhase('drop');
                    return;
                }
            }
            const separated = await separateAudio();
            if (!separated && !cancelled()) setPhase('drop');
        },
        [
            clearAudioError,
            loadAudio,
            loadModel,
            modelLoaded,
            loadedModel,
            selectedModel,
            separateAudio,
        ]
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

    const cancelRun = () => {
        runIdRef.current += 1;
        cancel();
        // A run that finished just before the click (the "COMPLETE" moment
        // before the studio opens) already has stems: drop them too, or they
        // stay in memory, unreachable, with the leave-page warning still on.
        clearStems();
        setPhase('drop');
    };

    const handleNewFile = () => {
        if (phase === 'processing') {
            if (window.confirm('Cancel processing this file?')) cancelRun();
            return;
        }
        // The studio has no way back once the drop screen replaces it, so
        // treat leaving it like leaving the page.
        if (
            phase === 'studio'
            && stemsReady
            && !window.confirm('Start over with a new file? The separated stems will be discarded.')
        ) {
            return;
        }
        pauseAll();
        playhead.set(0);
        setIsPlaying(false);
        // The user chose to discard the stems; drop them so later leave/reset
        // prompts don't ask again about work that no longer exists.
        clearStems();
        setPhase('drop');
    };

    // Work that leaving the page (or unmounting Home) would throw away.
    const hasWork = phase === 'processing' || stemsReady;
    const confirmLeave = () => {
        if (phase === 'processing') {
            return window.confirm('Leaving this page cancels processing. Leave anyway?');
        }
        if (stemsReady) {
            return window.confirm('Leaving this page discards the separated stems. Leave anyway?');
        }
        return true;
    };

    // Let the header wordmark / "Studio" link trigger this same soft reset.
    // It reuses handleNewFile, so the loaded model stays warm — no reload.
    const reset = useHomeReset();
    const handlersRef = useRef({ reset: handleNewFile, confirmLeave });
    useEffect(() => {
        handlersRef.current = { reset: handleNewFile, confirmLeave };
    });
    useEffect(() => {
        reset?.register({
            reset: () => handlersRef.current.reset(),
            confirmLeave: () => handlersRef.current.confirmLeave(),
        });
        return () => reset?.register(null);
    }, [reset]);

    useEffect(() => {
        if (!hasWork) return;
        const onBeforeUnload = (e: BeforeUnloadEvent) => {
            e.preventDefault();
            e.returnValue = '';
        };
        window.addEventListener('beforeunload', onBeforeUnload);
        return () => window.removeEventListener('beforeunload', onBeforeUnload);
    }, [hasWork]);

    // A file dropped outside the drop zone would otherwise make the browser
    // navigate to it, destroying the session.
    useEffect(() => {
        const preventFileDrop = (e: globalThis.DragEvent) => {
            if (e.dataTransfer?.types.includes('Files')) e.preventDefault();
        };
        window.addEventListener('dragover', preventFileDrop);
        window.addEventListener('drop', preventFileDrop);
        return () => {
            window.removeEventListener('dragover', preventFileDrop);
            window.removeEventListener('drop', preventFileDrop);
        };
    }, []);

    useEffect(() => {
        if (!audioError) return;
        const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        modalDismissRef.current?.focus();
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                clearAudioError();
                return;
            }
            if (e.key !== 'Tab') return;
            // Focus trap: Tab / Shift+Tab cycle within the dialog.
            const focusable = Array.from(
                modalRef.current?.querySelectorAll<HTMLElement>(
                    'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                ) ?? []
            );
            if (focusable.length === 0) return;
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            const active = document.activeElement;
            const inside = active instanceof Node && !!modalRef.current?.contains(active);
            if (e.shiftKey && (!inside || active === first)) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && (!inside || active === last)) {
                e.preventDefault();
                first.focus();
            }
        };
        window.addEventListener('keydown', onKey);
        return () => {
            window.removeEventListener('keydown', onKey);
            // Return focus to whatever opened the dialog, if still present.
            if (opener?.isConnected) opener.focus();
        };
    }, [audioError, clearAudioError]);

    // ---- keyboard -----------------------------------------------------
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (phase !== 'studio') return;
            const target = e.target instanceof HTMLElement ? e.target : null;
            // Leave typing and dialogs alone entirely.
            if (
                target
                && (target.isContentEditable
                    || target.closest('input, select, textarea, [contenteditable], [role="dialog"], dialog'))
            ) return;
            if (e.code === 'Space') {
                // A focused button activates on Space natively; toggling
                // playback too would double-fire (or hijack e.g. Dismiss).
                if (target?.closest('button')) return;
                e.preventDefault();
                if (isPlaying) pauseAll();
                else void playAll();
                return;
            }
            // Ignore browser / OS chords (⌘ / Ctrl / Alt + key) so we never
            // fight shortcuts like tab or profile switching.
            if (e.metaKey || e.ctrlKey || e.altKey) return;
            // Match the physical digit key via e.code, not e.key: with Shift
            // held most layouts report "!" "@" … for the number row.
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

    // A tag with nothing usable in a file name (blank, or only characters the
    // sanitizer removes) falls back to the file's own name.
    const usableTitle = trackTitle && cleanFileName(trackTitle) ? trackTitle.trim() : '';
    const trackName =
        usableTitle || audioFile?.name?.replace(/\.[^/.]+$/, '') || 'untitled';
    const exportingRef = useRef(false);
    const exportResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    // Truncate the track name alone, so a long title can't cut off the stem
    // suffix and give every stem the same name.
    const stemFileName = (key: string) =>
        `${sanitizeFileName(trackName)} - ${sanitizeFileName(key)}.wav`;

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
        // One pack at a time: a double-click would download two ZIPs.
        if (targets.length === 0 || exportingRef.current) return;
        exportingRef.current = true;
        if (exportResetRef.current !== null) clearTimeout(exportResetRef.current);
        setExportLabel(`PACKING ${targets.length} STEMS…`);
        try {
            const entries = await Promise.all(
                targets.map(async key => {
                    const res = await fetch(stemUrls[key]);
                    const buf = new Uint8Array(await res.arrayBuffer());
                    return { name: stemFileName(key), data: buf };
                })
            );
            const zip = makeZip(entries);
            const url = URL.createObjectURL(zip);
            downloadUrl(url, `${sanitizeFileName(trackName)} (stems).zip`);
            setTimeout(() => URL.revokeObjectURL(url), 2000);
            setExportLabel('STEMS.ZIP ↓');
        } catch (err) {
            console.error('Stem export failed:', err);
            setExportLabel('EXPORT FAILED');
        } finally {
            exportingRef.current = false;
        }
        // Only the latest export's reset runs, so a new pack keeps its label.
        exportResetRef.current = setTimeout(() => {
            exportResetRef.current = null;
            setExportLabel('EXPORT .ZIP ↓');
        }, 2600);
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

    const modelName = MODEL_CHOICES[selectedModel].short;

    // WAI-ARIA radio group keys: arrows move the selection (wrapping) and
    // focus with it; Home/End jump to the first/last model.
    const onModelKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
        const models = Object.keys(MODEL_CHOICES) as ModelType[];
        const current = models.indexOf(selectedModel);
        let next: number;
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight') next = (current + 1) % models.length;
        else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') next = (current - 1 + models.length) % models.length;
        else if (e.key === 'Home') next = 0;
        else if (e.key === 'End') next = models.length - 1;
        else return;
        e.preventDefault();
        setSelectedModel(models[next]);
        e.currentTarget.querySelectorAll<HTMLElement>('[role="radio"]')[next]?.focus();
    };
    // Screen-reader announcement of the status line, minus the per-chunk
    // download byte counts, which would otherwise be announced continuously.
    const statusAnnouncement = status.replace(/\.\.\. [\d.]+(?: \/ [\d.]+)? MiB$/, '...');

    return (
        <>
            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,video/*,.flac,.opus,.wma,.ape,.aif,.aiff,.mka"
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
                    onEnded={handleEnded}
                    // The browser can't play the source file itself (e.g. a
                    // codec only ffmpeg.wasm decoded); use the decoded PCM.
                    onError={() => void switchOriginalToPcm()}
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
                    onEnded={handleEnded}
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
                            <div
                                className="model-selector"
                                role="radiogroup"
                                aria-label="Separation model"
                                onClick={e => e.stopPropagation()}
                                onKeyDown={onModelKeyDown}
                            >
                                <div className="model-selector-head">
                                    <span>Separation model</span>
                                    <span className="model-selector-dots" />
                                </div>
                                <div className="model-board">
                                    {(Object.entries(MODEL_CHOICES) as [ModelType, ModelChoice][])
                                        .map(([value, choice], index) => {
                                            const selected = value === selectedModel;
                                            return (
                                                <button
                                                    key={value}
                                                    type="button"
                                                    role="radio"
                                                    aria-checked={selected}
                                                    // Roving tabindex: Tab enters the group at
                                                    // the checked option; arrows move within it.
                                                    tabIndex={selected ? 0 : -1}
                                                    className={`model-option${selected ? ' selected' : ''}`}
                                                    onClick={() => setSelectedModel(value)}
                                                >
                                                    <span className="model-option-lab">
                                                        <span className="model-option-index">
                                                            {String(index + 1).padStart(2, '0')}
                                                        </span>
                                                        <span className="model-option-copy">
                                                            <span className="model-option-name">{choice.short}</span>
                                                            <span className="model-option-sub">{choice.description}</span>
                                                        </span>
                                                    </span>
                                                    <span className="model-option-meta">
                                                        <span>{choice.stems} STEMS</span>
                                                        <span>{downloadSize(value)}</span>
                                                        <span title={`Model weights license: ${MODEL_CONFIGS[value].license}`}>
                                                            {MODEL_CONFIGS[value].license.toUpperCase()}
                                                        </span>
                                                    </span>
                                                </button>
                                            );
                                        })}
                                </div>
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
                            <Braid
                                active={false}
                                progress={progressPhase === 'separate'
                                    ? (displayPct ?? 0)
                                    : progressPhase === 'finalize' || progressPhase === 'complete'
                                        ? 100
                                        : undefined}
                                audioBuffer={audioBuffer}
                            />
                        </div>
                        <div className="proc">
                            <div className="pct-block">
                                <div className="pct-caption">{progressCaption}</div>
                                <div
                                    className={`pct${displayPctInteger === null ? ' is-indeterminate' : ''}`}
                                    role="progressbar"
                                    aria-label={progressCaption.toLowerCase()}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                    aria-valuenow={displayPctInteger ?? undefined}
                                    aria-valuetext={displayPctInteger === null
                                        ? 'Progress unavailable for this phase'
                                        : progressPhase === 'separate'
                                            ? `${displayPctInteger} percent estimated; ${segmentsDone} of ${segmentsTotal} segments complete`
                                            : `${displayPctInteger} percent`}
                                >
                                    <span className="pct-value">
                                        {displayPctInteger === null ? '—' : displayPctInteger}
                                    </span>
                                    <span className="pct-unit" aria-hidden="true">%</span>
                                </div>
                            </div>
                            <div className="proc-log">
                                {logs.slice(-13).map((line, i) => (
                                    <div key={i}>&gt; {line.toUpperCase()}</div>
                                ))}
                                <div className="cur">&gt; {(status || 'WORKING').toUpperCase()} </div>
                            </div>
                            <div className="sr-only" role="status">{statusAnnouncement}</div>
                        </div>
                        <div className="proc-actions">
                            <button className="spec" onClick={cancelRun}>
                                CANCEL ×
                            </button>
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
                                <RulerCanvas
                                    playhead={playhead}
                                    duration={duration}
                                    onSeek={seek}
                                />
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
                                                aria-label={`Mute ${lane.name.toLowerCase()}`}
                                                aria-pressed={!!muted[lane.key]}
                                            >
                                                M
                                            </button>
                                            <button
                                                className={`msq${isSolo ? ' on-s' : ''}`}
                                                onClick={() => toggleSolo(lane.key)}
                                                title="Solo"
                                                aria-label={`Solo ${lane.name.toLowerCase()}`}
                                                aria-pressed={isSolo}
                                            >
                                                S
                                            </button>
                                            <input
                                                type="range"
                                                min={0}
                                                max={100}
                                                value={volumes[lane.key] ?? 90}
                                                aria-label={`${lane.name.toLowerCase()} volume`}
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
                                                        downloadUrl(stemUrls[lane.key], stemFileName(lane.key))
                                                    }
                                                    title="Download stem"
                                                    aria-label={`Download ${lane.name.toLowerCase()} stem`}
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
                                            playhead={playhead}
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
                        aria-label={isPlaying ? 'Pause' : 'Play'}
                    >
                        {isPlaying ? '❚❚' : '▶'}
                    </button>
                    <div className="fcell">
                        <button className="tsq" onClick={resetTransport} title="Restart" aria-label="Restart">
                            ↺
                        </button>
                    </div>
                    <div className="fcell">
                        <span className="flab">T</span>
                        <TimeReadout playhead={playhead} duration={duration} />
                    </div>
                    <div className="fcell grow">
                        <span className="flab">VOL</span>
                        <input
                            type="range"
                            min={0}
                            max={100}
                            value={master}
                            aria-label="Master volume"
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
                    <div
                        ref={modalRef}
                        className="modal-content"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="error-modal-title"
                        aria-describedby="error-modal-msg"
                        onClick={e => e.stopPropagation()}
                    >
                        <div className="modal-title" id="error-modal-title">Operation failed</div>
                        <div className="modal-msg" id="error-modal-msg">{audioError}</div>
                        <button
                            ref={modalDismissRef}
                            className="modal-dismiss"
                            onClick={clearAudioError}
                        >
                            Dismiss
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}
