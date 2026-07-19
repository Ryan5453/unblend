// Number of amplitude bins we render per waveform lane.
export const WAVE_BINS = 400;

// Compute normalized (0..1) peak amplitudes from an interleaved Float32 buffer.
export function peaksFromInterleaved(
    data: Float32Array,
    channels: number,
    bins: number = WAVE_BINS,
): number[] {
    const frames = Math.floor(data.length / channels);
    const per = Math.max(1, Math.floor(frames / bins));
    const out = new Array<number>(bins).fill(0);
    let gmax = 1e-6;
    for (let b = 0; b < bins; b++) {
        const start = b * per;
        const end = Math.min(start + per, frames);
        let m = 0;
        for (let i = start; i < end; i++) {
            for (let c = 0; c < channels; c++) {
                const v = Math.abs(data[i * channels + c]);
                if (v > m) m = v;
            }
        }
        out[b] = m;
        if (m > gmax) gmax = m;
    }
    for (let b = 0; b < bins; b++) out[b] = Math.min(1, out[b] / gmax);
    return out;
}

// Compute normalized (0..1) peak amplitudes from a decoded AudioBuffer.
export function peaksFromBuffer(
    buffer: AudioBuffer,
    bins: number = WAVE_BINS,
): number[] {
    const channels: Float32Array[] = [];
    for (let c = 0; c < buffer.numberOfChannels; c++) {
        channels.push(buffer.getChannelData(c));
    }
    const frames = buffer.length;
    const per = Math.max(1, Math.floor(frames / bins));
    const out = new Array<number>(bins).fill(0);
    let gmax = 1e-6;
    for (let b = 0; b < bins; b++) {
        const start = b * per;
        const end = Math.min(start + per, frames);
        let m = 0;
        for (let i = start; i < end; i++) {
            for (const ch of channels) {
                const v = Math.abs(ch[i]);
                if (v > m) m = v;
            }
        }
        out[b] = m;
        if (m > gmax) gmax = m;
    }
    for (let b = 0; b < bins; b++) out[b] = Math.min(1, out[b] / gmax);
    return out;
}
