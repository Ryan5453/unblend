/**
 * The transport position, held outside React state so the studio page does
 * not re-render every animation frame during playback. Only the components
 * that draw the playhead subscribe (via useSyncExternalStore).
 */
export interface Playhead {
    get: () => number;
    set: (seconds: number) => void;
    subscribe: (listener: () => void) => () => void;
}

export function createPlayhead(): Playhead {
    let value = 0;
    const listeners = new Set<() => void>();
    return {
        get: () => value,
        set: seconds => {
            if (seconds === value) return;
            value = seconds;
            listeners.forEach(listener => listener());
        },
        subscribe: listener => {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
    };
}
