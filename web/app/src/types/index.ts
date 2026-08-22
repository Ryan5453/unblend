export type ProgressPhase =
    | 'idle'
    | 'audio'
    | 'download'
    | 'initialize'
    | 'separate'
    | 'finalize'
    | 'complete';

export interface DemucsState {
    modelLoaded: boolean;
    modelLoading: boolean;
    audioLoaded: boolean;
    audioBuffer: AudioBuffer | null;
    audioFile: File | null;
    separating: boolean;
    /** Whether `progress` is a measured percentage for the current phase. */
    progressDeterminate: boolean;
    /** Current work phase, used to keep the processing visual semantically honest. */
    progressPhase: ProgressPhase;
    progress: number;
    /** Exact completed-segment count during separation. */
    segmentsDone: number;
    segmentsTotal: number;
    /** Timing for a smooth, explicitly estimated within-segment visual. */
    segmentStartedAtMs: number;
    segmentExpectedMs: number;
    status: string;
}
