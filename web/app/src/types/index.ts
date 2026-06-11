export interface DemucsState {
    modelLoaded: boolean;
    modelLoading: boolean;
    audioLoaded: boolean;
    audioBuffer: AudioBuffer | null;
    audioFile: File | null;
    separating: boolean;
    progress: number;
    status: string;
}
