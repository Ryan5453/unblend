// Shared "stem voices" for the mix visualisations (Braid + SplitDiagram).
//
// Both components draw the same four instrument waveforms, so they live here in
// one place. Previously each component kept its own copy, which let the drum
// envelope fix land in one and silently drift in the other — keep them shared.

export interface StemLine {
    label: string;
    amp: number;
    lane: number;
    wave: (x: number, t: number) => number;
}

// Percussive envelope for the drum stem. Each beat is a smooth "thump":
// sin(pi * s^0.6)^2 rises from and returns to zero with *zero slope* at both
// ends, so consecutive beats meet with no value or slope discontinuity. That
// removes the sharp corner that used to read as a "bump" in the drawn line,
// while the s^0.6 warp skews the peak early (~s=0.31) to keep an attack-like,
// percussive feel. The expression already peaks at 1, so the stem keeps its
// visual amplitude.
const drumEnv = (s: number) => {
    const thump = Math.sin(Math.PI * Math.pow(s, 0.6));
    return thump * thump;
};

// Each stem has its own waveform character; the mix line is literally their
// sum, so the diagram is a true picture of what separation does.
export const STEMS: StemLine[] = [
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
            return (beat % 2 === 0 ? 1 : -0.75) * drumEnv(s);
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

export const AMP_SUM = STEMS.reduce((sum, s) => sum + s.amp, 0);

export const INK = [25, 25, 22];
export const RED = [207, 59, 23];

export const smooth = (u: number) => u * u * (3 - 2 * u);
export const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
