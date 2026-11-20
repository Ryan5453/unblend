// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { useUnblend } from './useUnblend';
import { decodeAudioFile } from '../utils/audio-decoder';
import { finalizeStems } from '../utils/stem-finalizer';

vi.mock('../utils/audio-decoder', () => ({ decodeAudioFile: vi.fn() }));
vi.mock('unblend', () => ({ SAMPLE_RATE: 44100, Separator: {} }));
vi.mock('../utils/stem-finalizer', () => ({ finalizeStems: vi.fn() }));

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (error: Error) => void;
    const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
}
const decoded = (title = 'old') => ({
    buffer: {} as AudioBuffer, artwork: `blob:${title}`, title, artist: title,
    usedFallback: 'mediabunny' as const,
});
beforeEach(() => {
    vi.stubGlobal('AudioContext', class { close = vi.fn(); });
    URL.createObjectURL = vi.fn(() => 'blob:source');
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.resetAllMocks(); vi.unstubAllGlobals(); });

for (const outcome of ['success', 'failure'] as const) {
    test(`cancel ignores late decode ${outcome}`, async () => {
        const pending = deferred<ReturnType<typeof decoded>>();
        vi.mocked(decodeAudioFile).mockReturnValueOnce(pending.promise);
        const { result } = renderHook(useUnblend);
        let load!: Promise<boolean>;
        await act(async () => { load = result.current.loadAudio(new File([], 'old.wav')); });
        act(() => result.current.cancel());
        const snapshot = result.current;
        await act(async () => {
            if (outcome === 'success') pending.resolve(decoded());
            else pending.reject(new Error('late failure'));
            expect(await load).toBe(false);
        });
        expect(result.current.audioError).toBeNull();
        expect(result.current.audioBuffer).toBeNull();
        expect(result.current.trackTitle).toBeNull();
        expect(result.current.originalUrl).toBeNull();
        expect(result.current.artworkUrl).toBeNull();
        expect(result.current.logs).toEqual(snapshot.logs);
        if (outcome === 'success') expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:old');
    });
}

for (const cancel of [true, false]) {
  for (const outcome of ['success', 'failure'] as const) {
    test(`new load supersedes old decode (cancel=${cancel}, ${outcome})`, async () => {
        const pending = deferred<ReturnType<typeof decoded>>();
        vi.mocked(decodeAudioFile).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(decoded('new'));
        const { result } = renderHook(useUnblend);
        let old!: Promise<boolean>;
        await act(async () => { old = result.current.loadAudio(new File([], 'old.wav')); });
        if (cancel) act(() => result.current.cancel());
        await act(async () => { expect(await result.current.loadAudio(new File([], 'new.wav'))).toBe(true); });
        const snapshot = result.current;
        await act(async () => {
            if (outcome === 'success') pending.resolve(decoded());
            else pending.reject(new Error('old failure'));
            expect(await old).toBe(false);
        });
        expect(result.current).toBe(snapshot);
        expect(result.current.trackTitle).toBe('new');
        expect(result.current.audioFile?.name).toBe('new.wav');
        if (outcome === 'success') expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:old');
    });
  }
}

test('cancel during PCM encoding revokes both local URLs without publishing metadata', async () => {
    const pending = deferred<Awaited<ReturnType<typeof finalizeStems>>>();
    vi.mocked(finalizeStems).mockReturnValueOnce(pending.promise);
    vi.mocked(decodeAudioFile).mockResolvedValueOnce({
        ...decoded(), usedFallback: 'ffmpeg',
        buffer: { numberOfChannels: 1, length: 1, sampleRate: 44100, getChannelData: () => new Float32Array(1) } as unknown as AudioBuffer,
    });
    const { result } = renderHook(useUnblend);
    let load!: Promise<boolean>;
    await act(async () => { load = result.current.loadAudio(new File([], 'old.wma')); });
    act(() => result.current.cancel());
    const snapshot = result.current;
    await act(async () => {
        pending.resolve([{ source: 'original', blob: new Blob(), peaks: [] }]);
        expect(await load).toBe(false);
    });
    expect(result.current).toBe(snapshot);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:old');
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:source');
});

test('active decode errors are still reported', async () => {
    vi.mocked(decodeAudioFile).mockRejectedValueOnce(new Error('bad file'));
    const { result } = renderHook(useUnblend);
    await act(async () => { expect(await result.current.loadAudio(new File([], 'bad.wav'))).toBe(false); });
    expect(result.current.audioError).toBe('bad file');
});
