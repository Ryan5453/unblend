import { useEffect, useRef } from 'react';

const INK = '25,25,22';
const RED = '207,59,23';

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

function attachSeek(canvas: HTMLCanvasElement, onSeek: (fraction: number) => void) {
    const seek = (clientX: number) => {
        const r = canvas.getBoundingClientRect();
        onSeek(Math.min(1, Math.max(0, (clientX - r.left) / r.width)));
    };
    const down = (e: PointerEvent) => {
        e.preventDefault();
        seek(e.clientX);
        const move = (ev: PointerEvent) => seek(ev.clientX);
        const up = () => {
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
        };
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', up);
    };
    canvas.addEventListener('pointerdown', down);
    return () => canvas.removeEventListener('pointerdown', down);
}

interface WaveCanvasProps {
    peaks: number[];
    height: number;
    progress: number;
    duration: number;
    gain: number;
    colorPlayed: string;
    colorFuture: string;
    onSeek: (fraction: number) => void;
}

export function WaveCanvas({
    peaks,
    height,
    progress,
    duration,
    gain,
    colorPlayed,
    colorFuture,
    onSeek,
}: WaveCanvasProps) {
    const ref = useRef<HTMLCanvasElement>(null);

    // Keep the latest onSeek without re-binding the pointer listener.
    const seekRef = useRef(onSeek);
    useEffect(() => {
        seekRef.current = onSeek;
    });

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

    // Bind pointer seeking + redraw on resize once.
    useEffect(() => {
        const c = ref.current;
        if (!c) return;
        const detach = attachSeek(c, (f) => seekRef.current(f));
        const ro = new ResizeObserver(() => draw());
        ro.observe(c);
        return () => {
            detach();
            ro.disconnect();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return <canvas ref={ref} style={{ height: `${height}px` }} />;
}

interface RulerCanvasProps {
    progress: number;
    duration: number;
    onSeek: (fraction: number) => void;
}

export function RulerCanvas({ progress, duration, onSeek }: RulerCanvasProps) {
    const ref = useRef<HTMLCanvasElement>(null);
    const seekRef = useRef(onSeek);
    useEffect(() => {
        seekRef.current = onSeek;
    });

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
        for (let s = 0; s <= dur; s++) {
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
                ctx.fillStyle = 'rgba(118,118,110,1)';
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

    useEffect(() => {
        const c = ref.current;
        if (!c) return;
        const detach = attachSeek(c, (f) => seekRef.current(f));
        const ro = new ResizeObserver(() => draw());
        ro.observe(c);
        return () => {
            detach();
            ro.disconnect();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return <canvas ref={ref} style={{ height: '32px' }} />;
}
