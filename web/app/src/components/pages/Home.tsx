import { useState, useRef, useEffect, type DragEvent } from 'react';
import { useDemucs } from '../../hooks/useDemucs';
import { Vinyl, ChannelStrip } from '../ui/Vinyl';
import { Waveform } from '../ui/Waveform';

interface StemStyle {
    name: string;
    color: string;
    colorRgb: string;
}

const STEM_STYLES: Record<string, StemStyle> = {
    drums: { name: 'Drums', color: '#c9a227', colorRgb: '201, 162, 39' },
    bass: { name: 'Bass', color: '#22c55e', colorRgb: '34, 197, 94' },
    guitar: { name: 'Guitar', color: '#ef4444', colorRgb: '239, 68, 68' },
    piano: { name: 'Piano', color: '#a855f7', colorRgb: '168, 85, 247' },
    other: { name: 'Other', color: '#6b7280', colorRgb: '107, 114, 128' },
    vocals: { name: 'Vocals', color: '#ec4899', colorRgb: '236, 72, 153' },
};

const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${String(secs).padStart(2, '0')}`;
};

export function Home() {
    const {
        modelLoaded,
        modelLoading,
        audioLoaded,
        audioBuffer,
        audioFile,
        separating,
        progress,
        status,
        stemUrls,
        artworkUrl,
        trackTitle,
        trackArtist,
        audioError,
        loadModel,
        loadAudio,
        clearAudioError,
        separateAudio,
    } = useDemucs();

    const fileInputRef = useRef<HTMLInputElement>(null);
    const dropZoneRef = useRef<HTMLDivElement>(null);
    const [volumes, setVolumes] = useState<Record<string, number>>({});
    const [mutedStems, setMutedStems] = useState<Record<string, boolean>>({});
    const [isTransportPlaying, setIsTransportPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [isDragging, setIsDragging] = useState(false);
    const audioRefs = useRef<Record<string, HTMLAudioElement>>({});

    // Merge mode
    const [mergeMode, setMergeMode] = useState(false);
    const [selectedStems, setSelectedStems] = useState<Set<string>>(new Set());
    const [mergedStemUrl, setMergedStemUrl] = useState<string | null>(null);
    const [mergedStemMembers, setMergedStemMembers] = useState<string[]>([]);
    const [isMerging, setIsMerging] = useState(false);

    const duration = audioBuffer?.duration ?? 0;
    const stems = Object.keys(stemUrls);
    const hasStemsReady = stems.length > 0;
    const mergedMemberSet = new Set(mergedStemMembers);
    const visibleStemKeys = stems.filter(stem => !mergedMemberSet.has(stem));

    const getPlaybackVolume = (stemName: string) => {
        if (mutedStems[stemName]) return 0;
        return (volumes[stemName] ?? 80) / 100;
    };

    const getTransportStemKeys = () => {
        const stemKeys = [...visibleStemKeys];
        if (mergedStemUrl) {
            stemKeys.push('merged');
        }
        return stemKeys;
    };

    // Sync volumes
    useEffect(() => {
        Object.keys(audioRefs.current).forEach(key => {
            const audio = audioRefs.current[key];
            if (audio) {
                audio.volume = getPlaybackVolume(key);
            }
        });
    }, [mutedStems, volumes]);

    // Time sync
    useEffect(() => {
        if (!isTransportPlaying) return;

        const interval = setInterval(() => {
            const referenceStem = getTransportStemKeys()[0];
            if (!referenceStem) return;

            const audio = audioRefs.current[referenceStem];
            if (audio) {
                setCurrentTime(audio.currentTime);
            }
        }, 100);
        return () => clearInterval(interval);
    }, [isTransportPlaying, mergedStemUrl, visibleStemKeys]);

    useEffect(() => {
        setMutedStems({});
        setIsTransportPlaying(false);
        setCurrentTime(0);
        setMergedStemMembers([]);
    }, [stemUrls]);

    useEffect(() => {
        const mergedAudio = audioRefs.current.merged;
        if (mergedAudio) {
            mergedAudio.pause();
            mergedAudio.currentTime = 0;
        }
    }, [mergedStemUrl]);

    useEffect(() => {
        mergedStemMembers.forEach(stemName => {
            const audio = audioRefs.current[stemName];
            if (audio) {
                audio.pause();
            }
        });
    }, [mergedStemMembers]);

    useEffect(() => {
        const mergedAudio = audioRefs.current.merged;
        if (!mergedAudio || !mergedStemUrl) return;

        mergedAudio.currentTime = currentTime;
        mergedAudio.volume = getPlaybackVolume('merged');

        if (isTransportPlaying) {
            void mergedAudio.play();
        }
    }, [mergedStemUrl]);

    const handleSeparate = async () => {
        if (!audioLoaded) return;

        if (!modelLoaded) {
            const success = await loadModel('htdemucs');
            if (success) {
                separateAudio();
            }
        } else {
            separateAudio();
        }
    };

    const handleFileClick = () => fileInputRef.current?.click();

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) loadAudio(file);
    };

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) loadAudio(file);
    };

    const togglePlay = (stemName: string) => {
        const audio = audioRefs.current[stemName];
        if (!audio) return;

        if (!isTransportPlaying) {
            void playAll();
        }

        setMutedStems(prev => {
            const nextMuted = !prev[stemName];
            audio.volume = nextMuted ? 0 : getPlaybackVolume(stemName);
            return { ...prev, [stemName]: nextMuted };
        });
    };

    const playAll = async () => {
        const transportStemKeys = getTransportStemKeys();
        await Promise.all(transportStemKeys.map(async stem => {
            const audio = audioRefs.current[stem];
            if (audio) {
                audio.currentTime = currentTime;
                audio.volume = getPlaybackVolume(stem);
                await audio.play();
            }
        }));
        setIsTransportPlaying(true);
    };

    const pauseAll = () => {
        getTransportStemKeys().forEach(stem => {
            const audio = audioRefs.current[stem];
            if (audio) audio.pause();
        });
        setIsTransportPlaying(false);
    };

    const stopAll = () => {
        getTransportStemKeys().forEach(stem => {
            const audio = audioRefs.current[stem];
            if (audio) {
                audio.pause();
                audio.currentTime = 0;
            }
        });
        setIsTransportPlaying(false);
        setCurrentTime(0);
    };

    const seekTo = (time: number) => {
        const clamped = Math.max(0, Math.min(duration, time));
        setCurrentTime(clamped);
        getTransportStemKeys().forEach(stem => {
            const audio = audioRefs.current[stem];
            if (audio) audio.currentTime = clamped;
        });
    };

    const handleDownload = (stemName: string) => {
        const url = stemUrls[stemName];
        if (!url) return;
        const a = document.createElement('a');
        a.href = url;
        a.download = `${stemName}.wav`;
        a.click();
    };

    const handleDownloadAll = () => {
        const downloadKeys = getTransportStemKeys();
        downloadKeys.forEach((source, index) => {
            const url = source === 'merged' ? mergedStemUrl : stemUrls[source];
            if (url) {
                setTimeout(() => {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = source === 'merged' ? 'merged.wav' : `${source}.wav`;
                    a.click();
                }, index * 200);
            }
        });
    };

    const toggleStemSelection = (stemKey: string) => {
        setSelectedStems(prev => {
            const next = new Set(prev);
            if (next.has(stemKey)) next.delete(stemKey);
            else next.add(stemKey);
            return next;
        });
    };

    const handleMergeStems = async () => {
        if (selectedStems.size === 0) return;
        setIsMerging(true);
        const audioContext = new AudioContext({ sampleRate: 44100 });
        const mergeMembers = Array.from(selectedStems);

        try {
            const decodedBuffers: AudioBuffer[] = [];
            
            for (const stemKey of mergeMembers) {
                const url = stemUrls[stemKey];
                if (url) {
                    const response = await fetch(url);
                    const arrayBuffer = await response.arrayBuffer();
                    const buffer = await audioContext.decodeAudioData(arrayBuffer);
                    decodedBuffers.push(buffer);
                }
            }

            if (decodedBuffers.length === 0) {
                setIsMerging(false);
                return;
            }

            const numSamples = decodedBuffers[0].length;
            const mergedData = new Float32Array(numSamples * 2);
            
            for (const buffer of decodedBuffers) {
                const left = buffer.getChannelData(0);
                const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
                for (let i = 0; i < numSamples; i++) {
                    mergedData[i * 2] += left[i];
                    mergedData[i * 2 + 1] += right[i];
                }
            }

            const { createWavBlob } = await import('../../utils/wav-utils');
            const blob = createWavBlob(mergedData, 2, 44100);
            const url = URL.createObjectURL(blob);

            const mergedAudio = audioRefs.current.merged;
            if (mergedAudio) {
                mergedAudio.pause();
                mergedAudio.currentTime = 0;
            }
            if (mergedStemUrl) URL.revokeObjectURL(mergedStemUrl);
            setMergedStemUrl(url);
            setMergedStemMembers(mergeMembers);
            setMergeMode(false);
            setSelectedStems(new Set());
        } catch (error) {
            console.error('Error merging:', error);
        } finally {
            void audioContext.close();
            setIsMerging(false);
        }
    };

    // Get display name for plaque - prefer metadata title over filename
    const getTrackName = () => {
        if (!audioFile) return null;
        // Prefer metadata title if available
        if (trackTitle) return trackTitle;
        // Fall back to filename without extension
        const name = audioFile.name;
        return name.replace(/\.[^/.]+$/, '');
    };

    // Get artist name for plaque subtitle
    const getArtistName = () => {
        return trackArtist;
    };

    return (
        <>
            <main className="flex-1 relative">
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                />

                {/* Hidden audio elements */}
                {stems.map(stemKey => (
                    <audio
                        key={stemKey}
                        ref={el => { if (el) audioRefs.current[stemKey] = el; }}
                        src={stemUrls[stemKey]}
                        onEnded={() => setIsTransportPlaying(false)}
                    />
                ))}
                {mergedStemUrl && (
                    <audio
                        ref={el => { if (el) audioRefs.current['merged'] = el; }}
                        src={mergedStemUrl}
                        onEnded={() => setIsTransportPlaying(false)}
                    />
                )}

                {/* Award Frame - Only show when no stems ready */}
                {!hasStemsReady && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div 
                            ref={dropZoneRef}
                            className={`award-frame pointer-events-auto ${isDragging ? 'ring-2 ring-[#c9a227]' : ''}`}
                            onClick={!audioFile ? handleFileClick : undefined}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            style={{ cursor: !audioFile ? 'pointer' : 'default' }}
                        >
                            <div className="award-frame-inner">
                                {/* Gold Vinyl */}
                                <Vinyl
                                    spinning={separating || isTransportPlaying}
                                    artworkUrl={artworkUrl}
                                    progress={separating ? progress : undefined}
                                />

                                {/* Plaque */}
                                <div className="plaque">
                                    {audioFile ? (
                                        <>
                                            <div className="plaque-title">{getTrackName()}</div>
                                            <div className="plaque-subtitle">
                                                {separating 
                                                    ? status 
                                                    : (getArtistName() || formatTime(duration))
                                                }
                                            </div>
                                        </>
                                    ) : (
                                        <div className="plaque-empty">
                                            {isDragging ? 'Drop to load' : 'Drop audio or click to browse'}
                                        </div>
                                    )}
                                </div>

                            </div>
                        </div>
                    </div>
                )}

                {/* Separate button - positioned below center */}
                {audioFile && !hasStemsReady && !separating && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ paddingTop: '620px' }}>
                        <button
                            onClick={handleSeparate}
                            disabled={!audioLoaded || modelLoading}
                            className="btn btn-primary pointer-events-auto animate-fade-in"
                        >
                            {modelLoading ? 'Loading model...' : 'Separate'}
                        </button>
                    </div>
                )}

                {/* Mixer controls - centered on page when stems ready */}
                {hasStemsReady && (
                    <div className="absolute inset-0 flex items-center justify-center animate-fade-in">
                        <div className="flex flex-col items-center gap-6">
                            {/* Track info + Time + Transport on one row */}
                            <div className="flex items-center justify-center gap-6">
                                <div className="track-info">
                                    {artworkUrl ? (
                                        <img src={artworkUrl} alt="" className="track-info-art" />
                                    ) : (
                                        <div className="track-info-art track-info-art-empty" />
                                    )}
                                    <div className="track-info-text">
                                        <div className="track-info-title">{getTrackName()}</div>
                                        {getArtistName() && (
                                            <div className="track-info-artist">{getArtistName()}</div>
                                        )}
                                    </div>
                                </div>

                                <div className="lcd-display text-center">
                                    <div className="lcd-time">
                                        {formatTime(currentTime)}
                                        <span className="lcd-time-total"> / {formatTime(duration)}</span>
                                    </div>
                                </div>

                                <div className="transport">
                                    <button className="transport-btn" onClick={stopAll} title="Stop">
                                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                            <rect x="6" y="6" width="12" height="12" />
                                        </svg>
                                    </button>
                                    <button
                                        className="transport-btn primary"
                                        onClick={isTransportPlaying ? pauseAll : () => { void playAll(); }}
                                    >
                                        {isTransportPlaying ? (
                                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                                            </svg>
                                        ) : (
                                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                                <path d="M8 5v14l11-7z" />
                                            </svg>
                                        )}
                                    </button>
                                </div>
                            </div>

                            <Waveform
                                audioBuffer={audioBuffer}
                                duration={duration}
                                currentTime={currentTime}
                                onSeek={seekTo}
                            />

                            {/* Mixer */}
                            <div className="mixer-console">
                                <div className="mixer-channels justify-center">
                                    {visibleStemKeys.map(stemKey => {
                                        const style = STEM_STYLES[stemKey] || STEM_STYLES.other;
                                        return (
                                            <ChannelStrip
                                                key={stemKey}
                                                name={style.name}
                                                color={style.color}
                                                colorRgb={style.colorRgb}
                                                volume={volumes[stemKey] ?? 80}
                                                isPlaying={isTransportPlaying && !mutedStems[stemKey]}
                                                isMergeMode={mergeMode}
                                                isSelected={selectedStems.has(stemKey)}
                                                onVolumeChange={(v) => setVolumes(prev => ({ ...prev, [stemKey]: v }))}
                                                onTogglePlay={() => togglePlay(stemKey)}
                                                onDownload={() => handleDownload(stemKey)}
                                                onToggleSelect={() => toggleStemSelection(stemKey)}
                                            />
                                        );
                                    })}
                                    {/* Merged stem channel if exists */}
                                    {mergedStemUrl && (
                                        <ChannelStrip
                                            name="Merged"
                                            color="#c9a227"
                                            colorRgb="201, 162, 39"
                                            volume={volumes['merged'] ?? 80}
                                            isPlaying={isTransportPlaying && !mutedStems.merged}
                                            isMergeMode={false}
                                            isSelected={false}
                                            onVolumeChange={(v) => setVolumes(prev => ({ ...prev, merged: v }))}
                                            onTogglePlay={() => togglePlay('merged')}
                                            onDownload={() => {
                                                const a = document.createElement('a');
                                                a.href = mergedStemUrl;
                                                a.download = 'merged.wav';
                                                a.click();
                                            }}
                                            onToggleSelect={() => {}}
                                        />
                                    )}
                                </div>
                            </div>

                            {/* Actions */}
                            <div className="flex flex-wrap justify-center gap-2">
                                {!mergeMode ? (
                                    <>
                                        <button onClick={handleDownloadAll} className="btn btn-primary">
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
                                            </svg>
                                            Download All
                                        </button>
                                        <button onClick={() => setMergeMode(true)} className="btn btn-ghost">
                                            Merge
                                        </button>
                                    </>
                                ) : (
                                    <>
                                        <button 
                                            onClick={handleMergeStems} 
                                            disabled={selectedStems.size === 0 || isMerging}
                                            className="btn btn-primary"
                                        >
                                            {isMerging ? 'Merging...' : `Merge (${selectedStems.size})`}
                                        </button>
                                        <button onClick={() => { setMergeMode(false); setSelectedStems(new Set()); }} className="btn btn-ghost">
                                            Cancel
                                        </button>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </main>

            {/* Error Modal */}
            {audioError && (
                <div className="modal-backdrop" onClick={clearAudioError}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <div className="flex items-start gap-3 mb-4">
                            <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center flex-shrink-0">
                                <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </div>
                            <div>
                                <h3 className="font-semibold text-white text-sm mb-1">Error</h3>
                                <p className="text-xs text-[#666]">{audioError}</p>
                            </div>
                        </div>
                        <div className="flex justify-end">
                            <button onClick={clearAudioError} className="btn btn-ghost text-sm">
                                Dismiss
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
