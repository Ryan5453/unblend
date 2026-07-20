import { useEffect, useMemo, useRef } from 'react';

interface BraidProps {
    /** Hover/drag excitement (drop phase). */
    active?: boolean;
    /** 0–100 unbraids the strands into their lanes (processing phase). */
    progress?: number;
    /** When present, the braid's amplitude follows the track's envelope. */
    audioBuffer?: AudioBuffer | null;
}

interface StemLine {
    label: string;
    amp: number;
    lane: number;
    wave: (x: number, t: number) => number;
}

// Same stem characters as SplitDiagram: each strand waves like its
// instrument, so the braid is a true picture of the blended mix.
const STEMS: StemLine[] = [
    {
        label: 'VOCALS',
        amp: 9,
        lane: -1,
        wave: (x, t) => 0.55 * Math.sin(x * 0.018 + t * 2.0) + 0.45 * Math.sin(x * 0.041 - t * 1.4),
    },
    {
        label: 'DRUMS',
        amp: 13,
        lane: -1 / 3,
        wave: (x, t) => {
            const u = x - t * 150;
            const s = u / 120 - Math.floor(u / 120);
            const beat = Math.floor(u / 120);
            return (beat % 2 === 0 ? 1 : -0.75) * Math.exp(-s * 9);
        },
    },
    {
        label: 'BASS',
        amp: 14,
        lane: 1 / 3,
        wave: (x, t) => Math.sin(x * 0.0085 - t * 1.2),
    },
    {
        label: 'OTHER',
        amp: 7,
        lane: 1,
        wave: (x, t) => 0.5 * Math.sin(x * 0.07 + t * 3.1) + 0.5 * Math.sin(x * 0.033 - t * 2.3),
    },
];

const INK = [25, 25, 22];
const RED = [207, 59, 23];

const N = STEMS.length;
// Each strand pulls out of the braid over a 40-point window of progress,
// staggered top lane first.
const PEEL_WINDOW = 40;
const PEEL_STAGGER = (100 - PEEL_WINDOW) / (N - 1);

const ENVELOPE_BUCKETS = 160;

const smooth = (u: number) => u * u * (3 - 2 * u);
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/**
 * The mix as a braided line: four strands (one per stem, each with its own
 * waveform character) wound around a common carrier so they read as a single
 * rope of signal. `active` excites it; `progress` unbraids it strand by
 * strand into the SplitDiagram's lanes; with an AudioBuffer the amplitude
 * follows the actual track.
 */
export function Braid({ active = false, progress, audioBuffer = null }: BraidProps) {
    const wrapRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const tagRefs = useRef<(HTMLSpanElement | null)[]>([]);

    const envelope = useMemo(() => {
        if (!audioBuffer) return null;
        const data = audioBuffer.getChannelData(0);
        const env = new Float32Array(ENVELOPE_BUCKETS);
        const per = Math.max(1, Math.floor(data.length / ENVELOPE_BUCKETS));
        const stride = Math.max(1, Math.floor(per / 64));
        for (let b = 0; b < ENVELOPE_BUCKETS; b++) {
            let peak = 0;
            const startIdx = b * per;
            const end = Math.min(startIdx + per, data.length);
            for (let j = startIdx; j < end; j += stride) {
                const v = Math.abs(data[j]);
                if (v > peak) peak = v;
            }
            env[b] = peak;
        }
        const smoothed = new Float32Array(ENVELOPE_BUCKETS);
        for (let b = 0; b < ENVELOPE_BUCKETS; b++) {
            const prev = env[Math.max(0, b - 1)];
            const next = env[Math.min(ENVELOPE_BUCKETS - 1, b + 1)];
            smoothed[b] = (prev + env[b] * 2 + next) / 4;
        }
        let max = 0;
        for (const v of smoothed) max = Math.max(max, v);
        if (max > 0) {
            for (let b = 0; b < ENVELOPE_BUCKETS; b++) smoothed[b] /= max;
        }
        return smoothed;
    }, [audioBuffer]);

    const live = useRef({ active, progress, envelope });
    useEffect(() => {
        live.current = { active, progress, envelope };
    });
    const drawOnce = useRef<(() => void) | null>(null);

    useEffect(() => {
        const wrap = wrapRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!wrap || !canvas || !ctx) return;

        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        const resize = () => {
            const dpr = window.devicePixelRatio || 1;
            canvas.width = Math.round(wrap.clientWidth * dpr);
            canvas.height = Math.round(wrap.clientHeight * dpr);
        };
        const ro = new ResizeObserver(() => {
            resize();
            if (reduced) drawOnce.current?.();
        });
        ro.observe(wrap);
        resize();

        let energy = 0;
        const peel = new Float32Array(N);
        let lastT = 0;

        const drawFrame = (t: number, snap: boolean) => {
            const { active: hot, progress: prog, envelope: env } = live.current;
            const dt = snap ? 0 : Math.min(t - lastT, 0.05);
            lastT = t;

            if (snap) energy = hot ? 1 : 0;
            else energy += ((hot ? 1 : 0) - energy) * 0.07;

            const p = prog ?? 0;
            for (let i = 0; i < N; i++) {
                const target = prog === undefined
                    ? 0
                    : smooth(clamp01((p - i * PEEL_STAGGER) / PEEL_WINDOW));
                peel[i] = snap ? target : peel[i] + (target - peel[i]) * Math.min(1, dt * 4);
            }

            const dpr = window.devicePixelRatio || 1;
            const w = canvas.width / dpr;
            const h = canvas.height / dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);

            const cy = h / 2;
            const x0 = Math.max(16, w * 0.02);
            const x1 = w - 92;
            const spread = Math.min(h * 0.36, 132);
            const mul = h / 340;
            const carrierAmp = 36 * mul * (1 + energy * 0.6);
            const wSpeed = prog !== undefined ? 2.6 : 1.1 + energy * 0.9;
            const k = (Math.PI * 2 * 3.2) / Math.max(1, w);
            const step = 2;
            ctx.lineWidth = 1;

            const envAt = (x: number) => {
                if (env) {
                    const pos = ((x - x0) / Math.max(1, x1 - x0)) * (ENVELOPE_BUCKETS - 1);
                    const b0 = Math.min(ENVELOPE_BUCKETS - 1, Math.max(0, Math.floor(pos)));
                    const b1 = Math.min(ENVELOPE_BUCKETS - 1, b0 + 1);
                    const frac = clamp01(pos - b0);
                    return 0.3 + 0.7 * (env[b0] + (env[b1] - env[b0]) * frac);
                }
                return (
                    0.62 +
                    0.24 * Math.sin(x * 0.006 + t * 0.55) +
                    0.14 * Math.sin(t * 0.23 + x * 0.013)
                );
            };

            let maxSwing = 0;
            for (let i = 0; i < N; i++) {
                const stem = STEMS[i];
                const s = peel[i];
                const laneY = cy + stem.lane * spread * s;
                const phi = (i * Math.PI) / 2;
                const charAmp = stem.amp * mul * (0.3 + 0.7 * s) * (1 + energy * 0.5);

                // Mid-peel strands flash red, settling back to ink in their
                // lane; hover excitement warms the whole braid.
                const redAmt = Math.max(4 * s * (1 - s), energy * 0.85);
                const cr = Math.round(lerp(INK[0], RED[0], redAmt));
                const cg = Math.round(lerp(INK[1], RED[1], redAmt));
                const cb = Math.round(lerp(INK[2], RED[2], redAmt));

                ctx.beginPath();
                for (let x = x0; x <= x1; x += step) {
                    const e = envAt(x);
                    const carrier = carrierAmp * e * (1 - s) * Math.sin(k * x - wSpeed * t + phi);
                    const char = stem.wave(x, t) * charAmp * e;
                    const y = laneY + carrier + char;
                    if (x === x0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                    if (x < x0 + 4) maxSwing = Math.max(maxSwing, Math.abs(y - cy));
                }
                ctx.strokeStyle = `rgba(${cr},${cg},${cb},${0.55 + 0.3 * s})`;
                ctx.stroke();

                // End tick appears with the strand's lane.
                if (s > 0.02) {
                    const stemPeak = stem.amp * mul * (1 + energy * 0.5) + 3;
                    const tickY = cy + stem.lane * spread * s;
                    ctx.beginPath();
                    ctx.moveTo(x1 + 4, tickY - stemPeak);
                    ctx.lineTo(x1 + 4, tickY + stemPeak);
                    ctx.strokeStyle = `rgba(${INK[0]},${INK[1]},${INK[2]},${0.5 * s})`;
                    ctx.stroke();
                }

                const tag = tagRefs.current[i];
                if (tag) {
                    tag.style.opacity = prog === undefined ? '0' : String(s);
                    tag.style.top = `calc(50% + ${stem.lane * s} * min(36%, 132px))`;
                }
            }

            // Start tick bracketing the braid's swing at the left terminal.
            ctx.beginPath();
            ctx.moveTo(x0, cy - maxSwing - 3);
            ctx.lineTo(x0, cy + maxSwing + 3);
            ctx.strokeStyle = `rgba(${INK[0]},${INK[1]},${INK[2]},.5)`;
            ctx.stroke();
        };

        let raf = 0;
        if (reduced) {
            drawOnce.current = () => drawFrame(21.7, true);
            drawFrame(21.7, true);
        } else {
            const start = performance.now();
            const loop = (now: number) => {
                drawFrame((now - start) / 1000, false);
                raf = requestAnimationFrame(loop);
            };
            raf = requestAnimationFrame(loop);
        }

        return () => {
            cancelAnimationFrame(raf);
            ro.disconnect();
            drawOnce.current = null;
        };
    }, []);

    useEffect(() => {
        drawOnce.current?.();
    }, [active, progress, envelope]);

    return (
        <div ref={wrapRef} className="braid-wrap">
            <canvas ref={canvasRef} />
            {STEMS.map((s, i) => (
                <span
                    key={s.label}
                    ref={el => {
                        tagRefs.current[i] = el;
                    }}
                    className="lane-tag"
                    style={{ opacity: 0, top: '50%' }}
                >
                    {s.label}
                </span>
            ))}
        </div>
    );
}
