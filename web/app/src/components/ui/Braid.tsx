import { useEffect, useMemo, useRef } from 'react';
import { STEMS, INK, RED, smooth, lerp, clamp01, type StemLine } from './stems';

interface BraidProps {
    /** Hover/drag excitement (drop phase). */
    active?: boolean;
    /** 0–100 unbraids the strands into their lanes (processing phase). */
    progress?: number;
    /** When present, the braid's amplitude follows the track's envelope. */
    audioBuffer?: AudioBuffer | null;
}

const N = STEMS.length;

const ENVELOPE_BUCKETS = 160;

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
        // Separation "wavefront": 0 = fully braided, 1 = fully split into lanes.
        // Eased toward progress so the seam glides rather than jumping.
        let wf = 0;
        let lastT = 0;
        // Carrier phase is integrated (not speed × t) so speed changes read
        // as smooth acceleration instead of scrubbing the whole braid.
        let phase = 21.7;

        const drawFrame = (t: number, snap: boolean) => {
            const { active: hot, progress: prog, envelope: env } = live.current;
            const dt = snap ? 0 : Math.min(t - lastT, 0.05);
            lastT = t;

            if (snap) energy = hot ? 1 : 0;
            else energy += ((hot ? 1 : 0) - energy) * 0.07;

            const p = prog ?? 0;
            const wfTarget = prog === undefined ? 0 : clamp01(p / 100);
            wf = snap ? wfTarget : wf + (wfTarget - wf) * Math.min(1, dt * 4);

            const dpr = window.devicePixelRatio || 1;
            const w = canvas.width / dpr;
            const h = canvas.height / dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);

            const cy = h / 2;
            const x0 = Math.max(16, w * 0.02);
            // Reserve room on the right for the lane labels + end-ticks only when
            // the strands actually peel out to their lanes (processing). On the
            // drop screen there are no visible labels, so mirror the left margin
            // and keep the braid centered under the caption / on the page.
            const x1 = w - (prog === undefined ? x0 : 92);
            const spread = Math.min(h * 0.36, 132);
            const mul = h / 340;
            const carrierAmp = 36 * mul * (1 + energy * 0.6);
            phase += (prog !== undefined ? 2.6 : 1.1 + energy * 0.9) * dt;
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

            // ---- Separation wavefront -------------------------------------
            // A red "seam" travels left→right as progress climbs. Left of it the
            // braid has split into its lane stems; right of it it is still wound.
            // TW is the diagonal width over which each strand peels apart.
            const TW = Math.max(70, (x1 - x0) * 0.16);
            // Run the seam a full TW past x1 so the whole width settles at 100%.
            const xf = x0 + (x1 - x0 + TW) * wf;
            const sepAt = (x: number) =>
                prog === undefined ? 0 : smooth(clamp01((xf - x) / TW));

            const strandY = (stem: StemLine, phi: number, x: number, sx: number) => {
                const e = envAt(x);
                const laneY = cy + stem.lane * spread * sx;
                const carrier = carrierAmp * e * (1 - sx) * Math.sin(k * x - phase + phi);
                const charAmp = stem.amp * mul * (0.3 + 0.7 * sx) * (1 + energy * 0.5);
                return laneY + carrier + stem.wave(x, t) * charAmp * e;
            };

            let maxSwing = 0;
            // Hover still warms the whole braid on the drop screen.
            const warm = energy * 0.85;
            const inkR = Math.round(lerp(INK[0], RED[0], warm));
            const inkG = Math.round(lerp(INK[1], RED[1], warm));
            const inkB = Math.round(lerp(INK[2], RED[2], warm));

            for (let i = 0; i < N; i++) {
                const stem = STEMS[i];
                const phi = (i * Math.PI) / 2;

                // Base strand.
                ctx.beginPath();
                for (let x = x0; x <= x1; x += step) {
                    const y = strandY(stem, phi, x, sepAt(x));
                    if (x === x0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                    if (x < x0 + 4) maxSwing = Math.max(maxSwing, Math.abs(y - cy));
                }
                ctx.strokeStyle = `rgba(${inkR},${inkG},${inkB},${0.55 + 0.3 * wf})`;
                ctx.lineWidth = 1;
                ctx.stroke();

                // Red glow riding the peel seam (only while actively splitting).
                if (prog !== undefined && wf > 0.001 && wf < 0.999) {
                    const gx0 = Math.max(x0, xf - TW);
                    const gx1 = Math.min(x1, xf);
                    if (gx1 - gx0 > 1) {
                        const grad = ctx.createLinearGradient(xf - TW, 0, xf, 0);
                        grad.addColorStop(0, 'rgba(207,59,23,0)');
                        grad.addColorStop(0.5, 'rgba(207,59,23,.95)');
                        grad.addColorStop(1, 'rgba(207,59,23,0)');
                        ctx.beginPath();
                        for (let x = gx0; x <= gx1; x += step) {
                            const y = strandY(stem, phi, x, sepAt(x));
                            if (x === gx0) ctx.moveTo(x, y);
                            else ctx.lineTo(x, y);
                        }
                        ctx.strokeStyle = grad;
                        ctx.lineWidth = 1.6;
                        ctx.stroke();
                    }
                }

                // End tick + lane label track the strand's right-edge lane.
                const sRight = sepAt(x1);
                if (sRight > 0.02) {
                    const stemPeak = stem.amp * mul * (1 + energy * 0.5) + 3;
                    const tickY = cy + stem.lane * spread * sRight;
                    ctx.beginPath();
                    ctx.moveTo(x1 + 4, tickY - stemPeak);
                    ctx.lineTo(x1 + 4, tickY + stemPeak);
                    ctx.strokeStyle = `rgba(${INK[0]},${INK[1]},${INK[2]},${0.5 * sRight})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
                const tag = tagRefs.current[i];
                if (tag) {
                    tag.style.opacity = String(sRight);
                    tag.style.top = `calc(50% + ${stem.lane * sRight} * min(36%, 132px))`;
                }
            }

            // Vertical seam "read head" at the wavefront.
            if (prog !== undefined && xf > x0 + 2 && xf < x1 - 2) {
                ctx.beginPath();
                ctx.moveTo(xf, cy - spread - 6);
                ctx.lineTo(xf, cy + spread + 6);
                ctx.strokeStyle = 'rgba(207,59,23,.35)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Start tick bracketing the braid's swing at the left terminal.
            ctx.beginPath();
            ctx.moveTo(x0, cy - maxSwing - 3);
            ctx.lineTo(x0, cy + maxSwing + 3);
            ctx.strokeStyle = `rgba(${INK[0]},${INK[1]},${INK[2]},.5)`;
            ctx.lineWidth = 1;
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
