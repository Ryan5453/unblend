import { useEffect, useRef, useSyncExternalStore, type KeyboardEvent, type RefObject } from 'react';
import type { Playhead } from '../../utils/playhead';

const INK = '25,25,22';
const RED = '189,54,19';

function gridLines(ctx: CanvasRenderingContext2D, w: number, h: number, duration: number) {
    const dur = Math.max(1, Math.round(duration));
    // Cap the number of drawn gridlines so very long tracks stay cheap.
    const stepSec = dur > 400 ? Math.ceil(dur / 400) : 1;
    for (let s = 0; s <= dur; s += stepSec) {
        const x = (s / dur) * w;
        ctx.fillStyle = s % 5 === 0 ? `rgba(${INK},.09)` : `rgba(${INK},.045)`;
        ctx.fillRect(x, 0, 1, h);
    }
    ctx.fillStyle = `rgba(${INK},.12)`;
    ctx.fillRect(0, h / 2 - 0.5, w, 1);
}

/** Playback fraction 0..1, re-rendering only the subscribing component. */
function usePlayheadFraction(playhead: Playhead, duration: number): number {
    const t = useSyncExternalStore(playhead.subscribe, playhead.get);
    return duration > 0 ? Math.min(1, Math.max(0, t / duration)) : 0;
}

/**
 * Observe the canvas once for pointer seeking and resizes. The resize handler
 * calls through a ref so it always redraws with the latest props rather than
 * the first render's closure (which drew stale or empty data).
 */
function useCanvasBindings(
    ref: RefObject<HTMLCanvasElement | null>,
    draw: () => void,
    onSeek: (fraction: number) => void,
) {
    const drawRef = useRef(draw);
    const seekRef = useRef(onSeek);
    useEffect(() => {
        drawRef.current = draw;
        seekRef.current = onSeek;
    });
    useEffect(() => {
        const c = ref.current;
        if (!c) return;
        const detach = attachSeek(c, f => seekRef.current(f));
        const ro = new ResizeObserver(() => drawRef.current());
        ro.observe(c);
        return () => {
            detach();
            ro.disconnect();
        };
    }, [ref]);
}

function attachSeek(canvas: HTMLCanvasElement, onSeek: (fraction: number) => void) {
    const seek = (clientX: number) => {
        const r = canvas.getBoundingClientRect();
        onSeek(Math.min(1, Math.max(0, (clientX - r.left) / r.width)));
    };
    // Pointer capture keeps the drag's move/up events on the canvas even
    // when the pointer leaves it (or the window), and releases on its own
    // when the pointer is lifted or the gesture is cancelled.
    let dragging: number | null = null;
    const down = (e: PointerEvent) => {
        if (e.button !== 0) return;
        e.preventDefault();
        dragging = e.pointerId;
        canvas.setPointerCapture(e.pointerId);
        seek(e.clientX);
    };
    const move = (e: PointerEvent) => {
        if (e.pointerId === dragging) seek(e.clientX);
    };
    const end = (e: PointerEvent) => {
        if (e.pointerId === dragging) dragging = null;
    };
    canvas.addEventListener('pointerdown', down);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('lostpointercapture', end);
    return () => {
        canvas.removeEventListener('pointerdown', down);
        canvas.removeEventListener('pointermove', move);
        canvas.removeEventListener('lostpointercapture', end);
    };
}

interface WaveCanvasProps {
    peaks: number[];
    height: number;
    playhead: Playhead;
    duration: number;
    gain: number;
    colorPlayed: string;
    colorFuture: string;
    onSeek: (fraction: number) => void;
}

export function WaveCanvas({
    peaks,
    height,
    playhead,
    duration,
    gain,
    colorPlayed,
    colorFuture,
    onSeek,
}: WaveCanvasProps) {
    const ref = useRef<HTMLCanvasElement>(null);
    const progress = usePlayheadFraction(playhead, duration);

    const draw = () => {
        const c = ref.current;
        if (!c) return;
        const ctx = c.getContext('2d');
        if (!ctx) return;
        const w = c.clientWidth;
        const h = c.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        if (c.width !== Math.round(w * dpr)) {
            c.width = Math.round(w * dpr);
            c.height = Math.round(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        gridLines(ctx, w, h, duration);
        const n = peaks.length;
        const step = w / n;
        for (let i = 0; i < n; i++) {
            const bh = Math.max(1.5, peaks[i] * gain * h * 0.92);
            ctx.fillStyle = i / n <= progress ? colorPlayed : colorFuture;
            ctx.fillRect(i * step, (h - bh) / 2, Math.max(1, step * 0.5), bh);
        }
        ctx.fillStyle = `rgb(${RED})`;
        ctx.fillRect(progress * w, 0, 1, h);
    };

    // Redraw on any visual input change.
    useEffect(draw);
    useCanvasBindings(ref, draw, onSeek);

    return <canvas ref={ref} style={{ height: `${height}px` }} />;
}

interface RulerCanvasProps {
    playhead: Playhead;
    duration: number;
    onSeek: (fraction: number) => void;
}

/** Seconds moved per arrow-key press on the focused ruler. */
const KEY_SEEK_SECONDS = 5;

function rulerStep(dur: number): number {
    const minStep = dur > 400 ? Math.ceil(dur / 400) : 1;
    for (const step of [1, 5, 30, 60]) {
        if (step >= minStep) return step;
    }
    return Math.ceil(minStep / 60) * 60;
}

function clock(seconds: number): string {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

export function RulerCanvas({ playhead, duration, onSeek }: RulerCanvasProps) {
    const ref = useRef<HTMLCanvasElement>(null);
    const currentTime = useSyncExternalStore(playhead.subscribe, playhead.get);
    const progress = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0;

    const draw = () => {
        const c = ref.current;
        if (!c) return;
        const ctx = c.getContext('2d');
        if (!ctx) return;
        const w = c.clientWidth;
        const h = c.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        if (c.width !== Math.round(w * dpr)) {
            c.width = Math.round(w * dpr);
            c.height = Math.round(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = `rgba(${INK},.12)`;
        ctx.fillRect(0, h - 1, w, 1);
        const dur = Math.max(1, Math.round(duration));
        // Same cap as gridLines. Steps stay multiples of 1, 5, 30 or 60 s
        // so the 5 s / 30 s emphasis and the labels below still land.
        const stepSec = rulerStep(dur);
        for (let s = 0; s <= dur; s += stepSec) {
            const x = (s / dur) * w;
            let th = 4;
            let a = 0.18;
            if (s % 30 === 0) {
                th = 12;
                a = 0.7;
            } else if (s % 5 === 0) {
                th = 7;
                a = 0.4;
            }
            ctx.fillStyle = `rgba(${INK},${a})`;
            ctx.fillRect(x, h - 1 - th, 1, th);
            if (s % 30 === 0 && s < dur) {
                ctx.fillStyle = 'rgba(102,102,96,1)';
                ctx.font = '9px "IBM Plex Mono", monospace';
                ctx.fillText(`${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`, x + 4, 11);
            }
        }
        const px = progress * w;
        ctx.fillStyle = `rgb(${RED})`;
        ctx.beginPath();
        ctx.moveTo(px - 4, 0);
        ctx.lineTo(px + 4, 0);
        ctx.lineTo(px, 6);
        ctx.fill();
        ctx.fillRect(px - 0.5, 0, 1, h);
    };

    useEffect(draw);
    useCanvasBindings(ref, draw, onSeek);

    // Keyboard seeking: the ruler is the timeline's single focusable slider.
    const onKeyDown = (e: KeyboardEvent<HTMLCanvasElement>) => {
        if (!(duration > 0)) return;
        let target: number;
        if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') target = currentTime - KEY_SEEK_SECONDS;
        else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') target = currentTime + KEY_SEEK_SECONDS;
        else if (e.key === 'Home') target = 0;
        else if (e.key === 'End') target = duration;
        else return;
        e.preventDefault();
        onSeek(Math.min(1, Math.max(0, target / duration)));
    };

    return (
        <canvas
            ref={ref}
            style={{ height: '32px' }}
            tabIndex={0}
            role="slider"
            aria-label="Playback position"
            aria-valuemin={0}
            aria-valuemax={Math.round(duration)}
            aria-valuenow={Math.round(currentTime)}
            aria-valuetext={`${clock(currentTime)} of ${clock(duration)}`}
            aria-keyshortcuts="ArrowLeft ArrowRight Home End"
            onKeyDown={onKeyDown}
        />
    );
}
