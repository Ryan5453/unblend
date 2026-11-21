import { afterEach, expect, test, vi } from 'vitest';
import { decodeAudioFile } from './audio-decoder';

const ffmpeg = vi.hoisted(() => ({
    loaded: true,
    load: vi.fn(), writeFile: vi.fn(), exec: vi.fn(), readFile: vi.fn(), deleteFile: vi.fn(),
}));
vi.mock('@ffmpeg/ffmpeg', () => ({ FFmpeg: class { constructor() { return ffmpeg; } } }));
vi.mock('@ffmpeg/util', () => ({ toBlobURL: vi.fn() }));
vi.mock('mediabunny', () => ({
    Input: class { constructor() { throw new Error('unsupported'); } },
    ALL_FORMATS: [], BufferSource: class {}, AudioSampleSink: class {}, IsobmffInputFormat: class {},
}));

afterEach(() => { vi.resetAllMocks(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

for (const fails of [false, true]) {
    test(`ffmpeg jobs wait for preceding cleanup (failure=${fails})`, async () => {
        let release!: () => void;
        const cleanupGate = new Promise<void>(resolve => { release = resolve; });
        let cleaning!: () => void;
        const cleanupStarted = new Promise<void>(resolve => { cleaning = resolve; });
        vi.spyOn(console, 'error').mockImplementation(() => {});
        vi.stubGlobal('AudioContext', class {
            decodeAudioData = vi.fn().mockResolvedValue({});
            close = vi.fn().mockResolvedValue(undefined);
        });
        ffmpeg.readFile.mockImplementation(async (name: string) => {
            if (name === 'output.wav') return new Uint8Array(4);
            throw new Error('no tags');
        });
        if (fails) ffmpeg.writeFile.mockRejectedValueOnce(new Error('write failed'));
        ffmpeg.deleteFile.mockResolvedValue(undefined).mockImplementationOnce(async () => { cleaning(); await cleanupGate; });
        const file = { name: 'track.wma', arrayBuffer: async () => new ArrayBuffer(1) } as File;
        const context = { sampleRate: 44100 } as AudioContext;
        const first = decodeAudioFile(file, context).catch(error => error);
        await cleanupStarted;
        const status = vi.fn();
        const second = decodeAudioFile(file, context, status);
        await vi.waitFor(() => expect(status).toHaveBeenCalledWith('Decoding audio...'));
        expect(ffmpeg.writeFile).toHaveBeenCalledTimes(1);
        release();
        const result = await first;
        await expect(second).resolves.toMatchObject({ usedFallback: 'ffmpeg' });
        expect(result instanceof Error).toBe(fails);
        expect(ffmpeg.writeFile).toHaveBeenCalledTimes(2);
        expect(ffmpeg.deleteFile.mock.invocationCallOrder[1]).toBeLessThan(ffmpeg.writeFile.mock.invocationCallOrder[1]);
    });
}
