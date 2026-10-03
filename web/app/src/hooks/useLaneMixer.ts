import { useCallback, useEffect, useMemo, useRef } from 'react';

interface LaneNodes {
    source: MediaElementAudioSourceNode;
    gain: GainNode;
}

let volumeWritableCache: boolean | null = null;

/**
 * iOS and iPadOS Safari ignore writes to HTMLMediaElement.volume (it always
 * reads back 1, since the hardware buttons own the level). Probe once.
 */
function volumeWritable(): boolean {
    if (volumeWritableCache === null) {
        try {
            const probe = document.createElement('audio');
            probe.volume = 0.5;
            volumeWritableCache = Math.abs(probe.volume - 0.5) < 1e-3;
        } catch {
            volumeWritableCache = true;
        }
    }
    return volumeWritableCache;
}

/**
 * Per-lane levels for a set of <audio> elements.
 *
 * Mute/solo always use `.muted`, which every browser honours. Levels use
 * `.volume` where it is writable (desktop, unchanged); where it is not
 * (iOS/iPadOS), each element is routed through its own GainNode on a single
 * shared AudioContext, created and resumed from `prepare()` inside the play
 * gesture, and closed on unmount.
 */
export function useLaneMixer() {
    const ctxRef = useRef<AudioContext | null>(null);
    const nodesRef = useRef(new Map<HTMLAudioElement, LaneNodes>());

    /**
     * Call synchronously from the user gesture that starts playback, before
     * `play()`, so the AudioContext may start (or resume after an iOS
     * interruption) and every current element is routed through a gain.
     */
    const prepare = useCallback((elements: HTMLAudioElement[]) => {
        if (volumeWritable() || typeof AudioContext === 'undefined') return;
        let ctx = ctxRef.current;
        if (!ctx || ctx.state === 'closed') {
            try {
                ctx = new AudioContext();
            } catch (error) {
                console.error('[unblend] AudioContext unavailable; faders are inactive:', error);
                return;
            }
            ctxRef.current = ctx;
            nodesRef.current.clear();
        }
        if (ctx.state !== 'running') void ctx.resume().catch(() => {});

        const nodes = nodesRef.current;
        for (const el of elements) {
            if (nodes.has(el)) continue;
            try {
                const source = ctx.createMediaElementSource(el);
                const gain = ctx.createGain();
                source.connect(gain).connect(ctx.destination);
                nodes.set(el, { source, gain });
            } catch (error) {
                // The element keeps playing directly, just without a fader.
                console.error('[unblend] Could not route a lane through Web Audio:', error);
            }
        }
        // Drop nodes for elements React has since unmounted.
        for (const [el, lane] of nodes) {
            if (!el.isConnected && !elements.includes(el)) {
                lane.source.disconnect();
                lane.gain.disconnect();
                nodes.delete(el);
            }
        }
    }, []);

    /** Apply a lane's fader level (0..1) and whether it is silenced. */
    const apply = useCallback((el: HTMLAudioElement, level: number, silenced: boolean) => {
        el.muted = silenced;
        const lane = nodesRef.current.get(el);
        if (lane) {
            lane.gain.gain.value = level;
        } else {
            el.volume = Math.max(0, Math.min(1, level));
        }
    }, []);

    useEffect(() => {
        const nodes = nodesRef.current;
        return () => {
            const ctx = ctxRef.current;
            ctxRef.current = null;
            nodes.clear();
            if (ctx && ctx.state !== 'closed') void ctx.close().catch(() => {});
        };
    }, []);

    return useMemo(() => ({ prepare, apply }), [prepare, apply]);
}
