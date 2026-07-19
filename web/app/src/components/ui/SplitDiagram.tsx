import { useEffect, useRef } from 'react';

interface StemLine {
    label: string;
    amp: number;
    lane: number;
    wave: (x: number, t: number) => number;
}

// Each stem has its own waveform character; the mix line is literally their
// sum, so the diagram is a true picture of what separation does.
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

const AMP_SUM = STEMS.reduce((sum, s) => sum + s.amp, 0);

const smooth = (u: number) => u * u * (3 - 2 * u);
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

export function SplitDiagram({ active }: { active: boolean }) {
    const wrapRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const activeRef = useRef(active);

    useEffect(() => {
        activeRef.current = active;
    }, [active]);

    useEffect(() => {
        const wrap = wrapRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!wrap || !canvas || !ctx) return;

        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let raf = 0;
        let energy = 0;

        const resize = () => {
            const dpr = window.devicePixelRatio || 1;
            canvas.width = Math.round(wrap.clientWidth * dpr);
            canvas.height = Math.round(wrap.clientHeight * dpr);
        };
        const ro = new ResizeObserver(resize);
        ro.observe(wrap);
        resize();

        const stroke = (alpha: number) => {
            const r = Math.round(lerp(INK[0], RED[0], energy));
            const g = Math.round(lerp(INK[1], RED[1], energy));
            const b = Math.round(lerp(INK[2], RED[2], energy));
            return `rgba(${r},${g},${b},${alpha})`;
        };

        const frame = (now: number) => {
            const t = reduced ? 21.7 : now / 1000;
            energy += ((activeRef.current ? 1 : 0) - energy) * 0.07;

            // Size and transform come from the canvas itself each frame, so a
            // resize (or HMR remount) can never leave a stale-scale drawing.
            const dpr = window.devicePixelRatio || 1;
            const w = canvas.width / dpr;
            const h = canvas.height / dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);
            const cy = h / 2;
            const x0 = Math.max(16, w * 0.02);
            const x1 = w - 92;
            const jx = x0 + (x1 - x0) * 0.42;
            const fanEnd = jx + (x1 - jx) * 0.55;
            const spread = Math.min(h * 0.36, 132);
            const ampMul = 1 + energy * 0.9;
            const step = 2;
            ctx.lineWidth = 1;

            // Mix line: sum of the stems, pinched as it enters the junction.
            ctx.beginPath();
            for (let x = x0; x <= jx; x += step) {
                let d = 0;
                for (const s of STEMS) d += s.wave(x, t) * s.amp;
                d *= 0.62 * ampMul;
                if (x > jx - 70) d *= 1 - 0.7 * smooth((x - (jx - 70)) / 70);
                if (x === x0) ctx.moveTo(x, cy + d);
                else ctx.lineTo(x, cy + d);
            }
            ctx.strokeStyle = stroke(0.9);
            ctx.stroke();

            // Start tick on the mix line — tall enough to bracket the line's
            // maximum swing so the signal never escapes its terminal.
            const mixPeak = AMP_SUM * 0.62 * ampMul + 3;
            ctx.beginPath();
            ctx.moveTo(x0, cy - mixPeak);
            ctx.lineTo(x0, cy + mixPeak);
            ctx.strokeStyle = stroke(0.5);
            ctx.stroke();

            // Stem lines fanning out to their lanes.
            for (const s of STEMS) {
                const laneY = cy + s.lane * spread;
                ctx.beginPath();
                for (let x = jx; x <= x1; x += step) {
                    const u = x >= fanEnd ? 1 : smooth((x - jx) / (fanEnd - jx));
                    const base = lerp(cy, laneY, u);
                    const amp = s.amp * (0.25 + 0.75 * u) * ampMul;
                    const y = base + s.wave(x, t) * amp;
                    if (x === jx) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.strokeStyle = stroke(0.55 + 0.25 * energy);
                ctx.stroke();

                const stemPeak = s.amp * ampMul + 3;
                ctx.beginPath();
                ctx.moveTo(x1 + 4, laneY - stemPeak);
                ctx.lineTo(x1 + 4, laneY + stemPeak);
                ctx.strokeStyle = stroke(0.5);
                ctx.stroke();
            }

            // The un/blend slash at the junction.
            ctx.beginPath();
            ctx.moveTo(jx + 5, cy - 9);
            ctx.lineTo(jx - 5, cy + 9);
            ctx.strokeStyle = `rgba(207,59,23,${0.85 + 0.15 * energy})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            raf = requestAnimationFrame(frame);
        };
        raf = requestAnimationFrame(frame);

        return () => {
            cancelAnimationFrame(raf);
            ro.disconnect();
        };
    }, []);

    return (
        <div ref={wrapRef} className="split-wrap">
            <canvas ref={canvasRef} />
            {STEMS.map(s => (
                <span
                    key={s.label}
                    className="lane-tag"
                    style={{ top: `calc(50% + ${s.lane} * min(36%, 132px))` }}
                >
                    {s.label}
                </span>
            ))}
        </div>
    );
}
