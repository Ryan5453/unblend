/**
 * Audio decoder with two-tier fallback for maximum format support.
 * 
 * Fallback chain:
 * 1. Mediabunny (primary, handles most formats via WebCodecs). For MP3, AAC
 *    and Opus (except Opus in MP4/M4A/MOV) the samples come from the browser's
 *    decodeAudioData instead, which trims encoder padding; Mediabunny still
 *    supplies tags and artwork.
 * 2. ffmpeg.wasm (lazy-loaded, handles exotic codecs like ALAC)
 *
 * Both tiers attempt to extract album artwork from the audio file.
 */

import {
    Input,
    ALL_FORMATS,
    BufferSource,
    AudioSampleSink,
    IsobmffInputFormat,
} from 'mediabunny';
// Type-only: the ffmpeg modules are imported dynamically in loadFFmpeg so
// they stay out of the main bundle unless the fallback is actually needed.
import type { FFmpeg } from '@ffmpeg/ffmpeg';

let ffmpegInstance: FFmpeg | null = null;
let ffmpegLoadPromise: Promise<FFmpeg> | null = null;

// CDN base URL for ffmpeg-core (using multi-threaded version from official example)
const FFMPEG_CORE_VERSION = '0.12.10';
const FFMPEG_CDN_BASE = `https://cdn.jsdelivr.net/npm/@ffmpeg/core-mt@${FFMPEG_CORE_VERSION}/dist/esm`;

/**
 * Result from decoding an audio file
 */
export interface DecodeResult {
    buffer: AudioBuffer;
    artwork: string | null; // Blob URL to artwork image, or null if none
    title: string | null; // Song title from metadata, or null if none
    artist: string | null; // Artist from metadata, or null if none
    usedFallback: 'mediabunny' | 'ffmpeg';
}

/**
 * Lazy load ffmpeg.wasm only when needed
 */
async function loadFFmpeg(): Promise<FFmpeg> {
    if (ffmpegInstance?.loaded) {
        return ffmpegInstance;
    }

    if (ffmpegLoadPromise) {
        return ffmpegLoadPromise;
    }

    ffmpegLoadPromise = (async () => {
        try {
            const [{ FFmpeg }, { toBlobURL }] = await Promise.all([
                import('@ffmpeg/ffmpeg'),
                import('@ffmpeg/util'),
            ]);
            const ffmpeg = new FFmpeg();

            // Blob URLs work around Vite/ESM module loading issues with
            // cross-origin CDN scripts.
            const coreURL = await toBlobURL(
                `${FFMPEG_CDN_BASE}/ffmpeg-core.js`,
                'text/javascript'
            );
            const wasmURL = await toBlobURL(
                `${FFMPEG_CDN_BASE}/ffmpeg-core.wasm`,
                'application/wasm'
            );
            // workerURL is required for multi-threaded version
            const workerURL = await toBlobURL(
                `${FFMPEG_CDN_BASE}/ffmpeg-core.worker.js`,
                'text/javascript'
            );

            await ffmpeg.load({ coreURL, wasmURL, workerURL });

            ffmpegInstance = ffmpeg;
            return ffmpeg;
        } catch (error) {
            console.error('ffmpeg.wasm load failed:', error);
            ffmpegLoadPromise = null; // Allow retry
            throw error;
        }
    })();

    return ffmpegLoadPromise;
}

function getExtension(fileName: string): string {
    const match = fileName.match(/\.[^.]+$/);
    return match ? match[0] : '';
}

/**
 * Extract artwork from audio file using ffmpeg
 */
async function extractArtworkWithFFmpeg(
    ffmpeg: FFmpeg,
    inputName: string
): Promise<string | null> {
    try {
        const artworkName = 'artwork.jpg';

        // Try to extract embedded artwork (album art is usually the first video stream)
        await ffmpeg.exec([
            '-i', inputName,
            '-an',           // No audio
            '-vcodec', 'copy', // Copy the video stream (which is the album art)
            '-f', 'image2',
            artworkName
        ]);

        try {
            const artworkData = await ffmpeg.readFile(artworkName) as Uint8Array;
            if (artworkData && artworkData.length > 0) {
                const blob = new Blob([new Uint8Array(artworkData)], { type: 'image/jpeg' });
                await ffmpeg.deleteFile(artworkName);
                return URL.createObjectURL(blob);
            }
        } catch {
            // Most files have no embedded artwork.
        }

        return null;
    } catch {
        // Artwork is optional; never fail the decode over it.
        return null;
    }
}

/**
 * Extract metadata (title, artist) from audio file using ffmpeg
 */
async function extractMetadataWithFFmpeg(
    ffmpeg: FFmpeg,
    inputName: string
): Promise<{ title: string | null; artist: string | null }> {
    try {
        const metadataName = 'metadata.txt';

        await ffmpeg.exec([
            '-i', inputName,
            '-f', 'ffmetadata',
            metadataName
        ]);

        try {
            const metadataBytes = await ffmpeg.readFile(metadataName) as Uint8Array;
            const metadataText = new TextDecoder().decode(metadataBytes);
            await ffmpeg.deleteFile(metadataName);

            const tags = parseFfmetadataGlobals(metadataText);
            return { title: tags.title || null, artist: tags.artist || null };
        } catch {
            // No metadata file
            return { title: null, artist: null };
        }
    } catch {
        // Metadata extraction failed
        return { title: null, artist: null };
    }
}

/**
 * Decode audio using ffmpeg.wasm (last resort for exotic codecs)
 */
async function decodeWithFFmpeg(
    arrayBuffer: ArrayBuffer,
    fileName: string,
    targetSampleRate: number,
    onStatus?: (status: string) => void
): Promise<{ buffer: AudioBuffer; artwork: string | null; title: string | null; artist: string | null }> {
    // The ffmpeg-core.wasm binary (~30MB) is fetched from a CDN here, on top
    // of the audio decode itself — worth its own status line since it's the
    // slowest, least predictable step in this fallback path.
    onStatus?.('Loading fallback decoder (ffmpeg.wasm)...');
    const ffmpeg = await loadFFmpeg();

    const inputName = 'input' + getExtension(fileName);
    const outputName = 'output.wav';

    let artwork: string | null = null;
    try {
        // Write input file to virtual filesystem (inside the cleanup scope so
        // a partial write is removed too).
        await ffmpeg.writeFile(inputName, new Uint8Array(arrayBuffer));

        artwork = await extractArtworkWithFFmpeg(ffmpeg, inputName);

        const { title, artist } = await extractMetadataWithFFmpeg(ffmpeg, inputName);

        // Convert to WAV format (universally decodable). Keep the source's
        // channels rather than downmixing (-ac 2 would fold surround into
        // stereo): like the Mediabunny path, the separator then duplicates
        // mono and uses only the first two channels of anything wider.
        await ffmpeg.exec([
            '-i', inputName,
            '-ar', String(targetSampleRate),
            '-f', 'wav',
            outputName
        ]);

        const outputData = await ffmpeg.readFile(outputName);

        // Decode the WAV with native Web Audio API. Close the context on
        // every path — Chrome caps live AudioContexts (~6), so leaking one
        // per failed decode would eventually break audio in the session.
        const audioContext = new AudioContext({ sampleRate: targetSampleRate });
        try {
            const wavData = outputData as Uint8Array;
            const wavBuffer = wavData.buffer.slice(wavData.byteOffset, wavData.byteOffset + wavData.byteLength) as ArrayBuffer;
            const buffer = await audioContext.decodeAudioData(wavBuffer);
            return { buffer, artwork, title, artist };
        } finally {
            await audioContext.close().catch(() => {});
        }
    } catch (error) {
        // Mirror the mediabunny path: don't leak the artwork blob URL when
        // the decode fails after extraction.
        if (artwork) {
            URL.revokeObjectURL(artwork);
        }
        throw error;
    } finally {
        // The ffmpeg instance is persistent — clear the MEMFS files on every
        // path so failed decodes don't accumulate track-sized files.
        await ffmpeg.deleteFile(inputName).catch(() => {});
        await ffmpeg.deleteFile(outputName).catch(() => {});
    }
}

/**
 * Decode audio using Mediabunny (handles WebCodecs-supported formats)
 */
async function decodeWithMediabunny(
    arrayBuffer: ArrayBuffer,
    targetSampleRate: number,
    audioContext: AudioContext
): Promise<{ buffer: AudioBuffer; artwork: string | null; title: string | null; artist: string | null }> {
    const input = new Input({
        formats: ALL_FORMATS,
        source: new BufferSource(arrayBuffer),
    });
    // Dispose on every path, including decode errors.
    try {
        return await decodeMediabunnyInput(input, targetSampleRate, audioContext, arrayBuffer);
    } finally {
        input.dispose();
    }
}

async function decodeMediabunnyInput(
    input: Input,
    targetSampleRate: number,
    audioContext: AudioContext,
    bytes: ArrayBuffer
): Promise<{ buffer: AudioBuffer; artwork: string | null; title: string | null; artist: string | null }> {
    // Extract artwork and metadata from tags
    let artwork: string | null = null;
    let title: string | null = null;
    let artist: string | null = null;
    try {
        const tags = await input.getMetadataTags();
        if (tags.images && tags.images.length > 0) {
            const image = tags.images[0];
            const blob = new Blob([new Uint8Array(image.data)], { type: image.mimeType || 'image/jpeg' });
            artwork = URL.createObjectURL(blob);
        }
        if (tags.title) {
            title = tags.title;
        }
        if (tags.artist) {
            artist = tags.artist;
        }
    } catch {
        // Metadata tags are optional; decode without them.
    }

    // From here on, any failure must revoke the artwork object URL created
    // above or it would leak (the caller never sees it on the throw path).
    try {
        const audioTrack = await input.getPrimaryAudioTrack();
        if (!audioTrack) {
            throw new Error('No audio track found in file');
        }

        // MP3, AAC and Opus start or end with encoder padding: AAC's priming
        // frames, MP3's LAME/Xing gapless delay, Opus's end trim (its pre-skip
        // Mediabunny already drops). Decoding every frame keeps it: MP3/AAC
        // stems come out ~25 ms late, Opus ones with an extra tail. The
        // browser's own decoder trims it (as its playback of the original
        // does), so use it for the samples, keeping Mediabunny's tags and
        // artwork. Ahead of the WebCodecs check: browsers without AudioDecoder
        // still decode these this way.
        // Not Opus in MP4/M4A: Chrome's decodeAudioData skips its pre-skip
        // twice there (dOps and the edit list), starting the stems ~6.5 ms
        // early, while Mediabunny skips it once.
        const isobmff = (await input.getFormat()) instanceof IsobmffInputFormat;
        const trimmedByBrowser =
            audioTrack.codec === 'mp3' ||
            audioTrack.codec === 'aac' ||
            (audioTrack.codec === 'opus' && !isobmff);
        if (trimmedByBrowser) {
            try {
                // A copy: decodeAudioData detaches its argument.
                const buffer = await audioContext.decodeAudioData(bytes.slice(0));
                return { buffer, artwork, title, artist };
            } catch {
                // Fall through to frame-by-frame decoding.
            }
        }

        const canDecode = await audioTrack.canDecode();
        if (!canDecode) {
            throw new Error(`Cannot decode audio codec: ${audioTrack.codec || 'unknown'}`);
        }

        const sampleRate = audioTrack.sampleRate;
        const numberOfChannels = audioTrack.numberOfChannels;

        // Collect decoded chunks and size the buffer from the frames actually
        // decoded. computeDuration() is only an estimate from container
        // metadata, so sizing from it could pad silence or truncate audio.
        const chunks: Float32Array[][] = Array.from({ length: numberOfChannels }, () => []);
        const sink = new AudioSampleSink(audioTrack);
        let samplesWritten = 0;

        for await (const sample of sink.samples()) {
            try {
                const frames = sample.numberOfFrames;
                for (let ch = 0; ch < numberOfChannels; ch++) {
                    const channelData = new Float32Array(frames);
                    sample.copyTo(channelData, { planeIndex: ch, format: 'f32-planar', frameCount: frames });
                    chunks[ch].push(channelData);
                }
                samplesWritten += frames;
            } finally {
                sample.close();
            }
        }

        if (samplesWritten === 0) {
            throw new Error('Audio track decoded to zero samples');
        }

        // Create output AudioBuffer (reuse the caller's AudioContext to avoid
        // leaking contexts; browsers cap the number of live AudioContexts).
        const buffer = audioContext.createBuffer(numberOfChannels, samplesWritten, sampleRate);
        for (let ch = 0; ch < numberOfChannels; ch++) {
            const outputChannel = buffer.getChannelData(ch);
            let offset = 0;
            for (const chunk of chunks[ch]) {
                outputChannel.set(chunk, offset);
                offset += chunk.length;
            }
            chunks[ch].length = 0;
        }

        if (sampleRate !== targetSampleRate) {
            const offlineCtx = new OfflineAudioContext(
                numberOfChannels,
                Math.ceil(samplesWritten * targetSampleRate / sampleRate),
                targetSampleRate
            );
            const source = offlineCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(offlineCtx.destination);
            source.start();
            const resampledBuffer = await offlineCtx.startRendering();
            return { buffer: resampledBuffer, artwork, title, artist };
        }

        return { buffer, artwork, title, artist };
    } catch (error) {
        if (artwork) URL.revokeObjectURL(artwork);
        throw error;
    }
}

/**
 * Decode audio file with two-tier fallback system
 */
export async function decodeAudioFile(
    file: File,
    audioContext: AudioContext,
    onStatus?: (status: string) => void
): Promise<DecodeResult> {
    const arrayBuffer = await file.arrayBuffer();

    // Tier 1: Try Mediabunny first (handles most formats via WebCodecs)
    try {
        onStatus?.('Decoding audio...');
        const { buffer, artwork, title, artist } = await decodeWithMediabunny(arrayBuffer, audioContext.sampleRate, audioContext);
        return { buffer, artwork, title, artist, usedFallback: 'mediabunny' };
    } catch {
        // Fall through to ffmpeg.wasm.
    }

    // Tier 2: Try ffmpeg.wasm (handles exotic codecs like ALAC, WMA, etc)
    try {
        const { buffer, artwork, title, artist } = await decodeWithFFmpeg(arrayBuffer, file.name, audioContext.sampleRate, onStatus);
        return { buffer, artwork, title, artist, usedFallback: 'ffmpeg' };
    } catch (ffmpegError) {
        console.error('All decode methods failed:', ffmpegError);
        throw new Error(
            `Unable to decode "${file.name}". This audio format is not supported. ` +
            `Error: ${ffmpegError instanceof Error ? ffmpegError.message : String(ffmpegError)}`,
            { cause: ffmpegError },
        );
    }
}

/**
 * Read the global (file-level) tags from ffmpeg's ffmetadata output.
 *
 * Stops at the first ``[CHAPTER]`` or ``[STREAM]`` section, whose own
 * ``title=`` lines would otherwise overwrite the track's. Values have ``=``,
 * ``;``, ``#``, ``\`` and newlines backslash-escaped; a trailing backslash
 * continues the value on the next line.
 */
export function parseFfmetadataGlobals(text: string): Record<string, string> {
    const tags: Record<string, string> = {};
    let pending = '';
    for (const raw of text.split('\n')) {
        const line = pending + raw;
        pending = '';
        // ffmpeg's only section markers; a key may itself start with '['.
        if (/^\[(CHAPTER|STREAM)\]$/.test(line)) break;
        if (line.startsWith(';') || line.startsWith('#') || !line) continue;
        // An odd number of trailing backslashes escapes the newline.
        const trailing = line.length - line.replace(/\\+$/, '').length;
        if (trailing % 2 === 1) {
            pending = line.slice(0, -1) + '\n';
            continue;
        }
        let key = '';
        let value = '';
        let inValue = false;
        for (let i = 0; i < line.length; i++) {
            let ch = line[i];
            if (ch === '\\' && i + 1 < line.length) {
                ch = line[++i];
            } else if (ch === '=' && !inValue) {
                inValue = true;
                continue;
            }
            if (inValue) value += ch;
            else key += ch;
        }
        if (inValue) tags[key.trim().toLowerCase()] = value.trim();
    }
    return tags;
}

