import { useMemo, useRef, useState } from 'react';

interface WaveformProps {
    audioBuffer: AudioBuffer | null;
    duration: number;
    currentTime: number;
    onSeek: (time: number) => void;
    bars?: number;
}

export function Waveform({
    audioBuffer,
    duration,
    currentTime,
    onSeek,
    bars = 200,
}: WaveformProps) {
    const peaks = useMemo(
        () => (audioBuffer ? computePeaks(audioBuffer, bars) : new Array(bars).fill(0)),
        [audioBuffer, bars]
    );

    const trackRef = useRef<HTMLDivElement>(null);
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const progress = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0;

    const indexFromClientX = (clientX: number): number | null => {
        const el = trackRef.current;
        if (!el) return null;
        const rect = el.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        return Math.min(bars - 1, Math.floor(fraction * bars));
    };

    const seekFromClientX = (clientX: number) => {
        const el = trackRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        onSeek(fraction * duration);
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
        seekFromClientX(e.clientX);
        const onMove = (m: MouseEvent) => {
            seekFromClientX(m.clientX);
            setHoverIndex(indexFromClientX(m.clientX));
        };
        const onUp = () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    };

    return (
        <div
            ref={trackRef}
            className="waveform"
            onMouseDown={handleMouseDown}
            onMouseMove={(e) => setHoverIndex(indexFromClientX(e.clientX))}
            onMouseLeave={() => setHoverIndex(null)}
        >
            {peaks.map((peak, i) => {
                const past = (i + 0.5) / bars <= progress;
                const hovered = i === hoverIndex;
                return (
                    <div
                        key={i}
                        className={`waveform-bar${past ? ' past' : ''}${hovered ? ' hover' : ''}`}
                        style={{ height: `${Math.max(4, peak * 100)}%` }}
                    />
                );
            })}
        </div>
    );
}

function computePeaks(buffer: AudioBuffer, bars: number): number[] {
    const totalSamples = buffer.length;
    const samplesPerBar = Math.max(1, Math.floor(totalSamples / bars));
    const channelData = buffer.getChannelData(0);
    const peaks = new Array(bars);
    for (let b = 0; b < bars; b++) {
        const start = b * samplesPerBar;
        const end = Math.min(start + samplesPerBar, totalSamples);
        let max = 0;
        for (let i = start; i < end; i++) {
            const v = Math.abs(channelData[i]);
            if (v > max) max = v;
        }
        peaks[b] = max;
    }
    return peaks;
}
