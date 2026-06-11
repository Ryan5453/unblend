import React, { useEffect, useRef, useState } from 'react';

interface GoldVinylProps {
    className?: string;
    progress?: number;
    spinning?: boolean;
    artworkUrl?: string | null;
}

export const Vinyl: React.FC<GoldVinylProps> = ({
    className = '',
    progress,
    spinning = false,
    artworkUrl
}) => {
    const circumference = 2 * Math.PI * 130;
    
    return (
        <div className={`gold-vinyl-container ${className}`}>
            {/* Progress ring */}
            {progress !== undefined && (
                <div className="vinyl-progress-ring">
                    <svg viewBox="0 0 280 280">
                        <circle
                            cx="140"
                            cy="140"
                            r="130"
                            className="vinyl-progress-track"
                        />
                        <circle
                            cx="140"
                            cy="140"
                            r="130"
                            className="vinyl-progress-fill"
                            strokeDasharray={circumference}
                            strokeDashoffset={circumference - (circumference * progress / 100)}
                        />
                    </svg>
                </div>
            )}

            {/* The gold record */}
            <div className={`gold-vinyl ${spinning ? 'spinning' : ''}`}>
                {/* Groove rings */}
                <div className="gold-vinyl-grooves" />
                
                {/* Center label */}
                <div className="gold-vinyl-label">
                    {artworkUrl ? (
                        <img src={artworkUrl} alt="Album art" />
                    ) : (
                        <div className="gold-vinyl-label-text">demucs</div>
                    )}
                    <div className="gold-vinyl-spindle" />
                </div>
            </div>
        </div>
    );
};

interface ChannelStripProps {
    name: string;
    color: string;
    colorRgb: string;
    volume: number;
    isPlaying: boolean;
    isMergeMode: boolean;
    isSelected: boolean;
    onVolumeChange: (value: number) => void;
    onTogglePlay: () => void;
    onDownload: () => void;
    onToggleSelect: () => void;
}

export const ChannelStrip: React.FC<ChannelStripProps> = ({
    name,
    color,
    colorRgb,
    volume,
    isPlaying,
    isMergeMode,
    isSelected,
    onVolumeChange,
    onTogglePlay,
    onDownload,
    onToggleSelect
}) => {
    const [vuLevel, setVuLevel] = useState(0);
    const animationRef = useRef<number | null>(null);

    useEffect(() => {
        if (isPlaying) {
            const animate = () => {
                const base = volume * 0.8;
                const variation = Math.random() * 20;
                setVuLevel(Math.min(100, base + variation));
                animationRef.current = requestAnimationFrame(animate);
            };
            animationRef.current = requestAnimationFrame(animate);
        } else if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
        }
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isPlaying, volume]);

    const segments = 10;
    // Derive lit segments from playback state: when paused the VU meter reads 0
    // without needing to write state from the effect (the rAF loop only updates
    // vuLevel while playing).
    const litSegments = isPlaying ? Math.floor((vuLevel / 100) * segments) : 0;

    const handleFaderMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
        const track = e.currentTarget;
        const updateVolume = (clientY: number) => {
            const rect = track.getBoundingClientRect();
            const y = clientY - rect.top;
            const percent = 100 - Math.max(0, Math.min(100, (y / rect.height) * 100));
            onVolumeChange(percent);
        };

        updateVolume(e.clientY);

        const handleMouseMove = (e: MouseEvent) => updateVolume(e.clientY);
        const handleMouseUp = () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
    };

    return (
        <div 
            className={`channel-strip ${isPlaying ? 'active' : ''}`}
            style={{ '--stem-color': color, '--stem-color-rgb': colorRgb } as React.CSSProperties}
        >
            <div className="channel-label">{name}</div>

            <div className="vu-meter">
                <div className="vu-bar">
                    {Array.from({ length: segments }).map((_, i) => (
                        <div
                            key={i}
                            className={`vu-segment ${i < litSegments ? 'lit' : ''} ${i >= segments - 2 && i < litSegments ? 'peak' : ''}`}
                        />
                    ))}
                </div>
                <div className="vu-bar">
                    {Array.from({ length: segments }).map((_, i) => (
                        <div
                            key={i}
                            className={`vu-segment ${i < litSegments ? 'lit' : ''} ${i >= segments - 2 && i < litSegments ? 'peak' : ''}`}
                        />
                    ))}
                </div>
            </div>

            <div className="fader-track" onMouseDown={handleFaderMouseDown}>
                <div className="fader-fill" style={{ height: `${volume}%` }} />
                <div className="fader-knob" style={{ bottom: `calc(${volume}% - 4px)` }} />
            </div>

            <div className="flex gap-1">
                {isMergeMode ? (
                    <button
                        className={`merge-checkbox ${isSelected ? 'checked' : ''}`}
                        onClick={onToggleSelect}
                    >
                        {isSelected && (
                            <svg className="w-3 h-3 text-black" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                        )}
                    </button>
                ) : (
                    <>
                        <button
                            className={`channel-btn ${isPlaying ? 'active' : ''}`}
                            onClick={onTogglePlay}
                            style={isPlaying ? { background: color, borderColor: color } : undefined}
                        >
                            {isPlaying ? (
                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                                </svg>
                            ) : (
                                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M8 5v14l11-7z" />
                                </svg>
                            )}
                        </button>
                        <button className="channel-btn" onClick={onDownload}>
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
                            </svg>
                        </button>
                    </>
                )}
            </div>
        </div>
    );
};
