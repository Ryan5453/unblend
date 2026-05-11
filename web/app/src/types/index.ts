export interface DemucsState {
    modelLoaded: boolean;
    modelLoading: boolean;
    audioLoaded: boolean;
    audioBuffer: AudioBuffer | null;
    audioFile: File | null;
    separating: boolean;
    progress: number;
    status: string;

    logs: LogEntry[];
}

export interface LogEntry {
    timestamp: Date;
    message: string;
    type: 'info' | 'success' | 'error';
}
