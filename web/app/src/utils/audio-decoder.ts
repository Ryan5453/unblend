/**
 * Audio decoder with two-tier fallback for maximum format support.
 * 
 * Fallback chain:
 * 1. Mediabunny (primary, handles most formats via WebCodecs)
 * 2. ffmpeg.wasm (lazy-loaded, handles exotic codecs like ALAC)
 * 
 * Both tiers attempt to extract album artwork from the audio file.
 */

import {
    Input,
    ALL_FORMATS,
    BufferSource,
    AudioSampleSink,
} from 'mediabunny';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL } from '@ffmpeg/util';

// Set to true to enable verbose decode logging during development.
const DEBUG = false;
const debug = (...args: unknown[]) => { if (DEBUG) console.log(...args); };

// Lazy-loaded ffmpeg instance
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
        debug('ffmpeg.wasm already loaded');
        return ffmpegInstance;
    }

    if (ffmpegLoadPromise) {
        debug('ffmpeg.wasm load already in progress...');
        return ffmpegLoadPromise;
    }

    ffmpegLoadPromise = (async () => {
        try {
            debug('Initializing ffmpeg.wasm...');
            const ffmpeg = new FFmpeg();

            // Add logging for progress
            ffmpeg.on('log', ({ message }) => {
                debug('[ffmpeg]', message);
            });

            ffmpeg.on('progress', ({ progress, time }) => {
                debug(`[ffmpeg] Progress: ${Math.round(progress * 100)}%, Time: ${time}`);
            });

            debug('Downloading ffmpeg-core... This may take a moment.');

            // Use toBlobURL to fetch and convert to blob URLs
            // This works around Vite/ESM module loading issues with CDN resources
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

            debug('Core files downloaded, initializing...');

            await ffmpeg.load({ coreURL, wasmURL, workerURL });

            debug('ffmpeg.wasm loaded successfully!');
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

/**
 * Get file extension from filename
 */
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

        // Check if artwork was extracted
        try {
            const artworkData = await ffmpeg.readFile(artworkName) as Uint8Array;
            if (artworkData && artworkData.length > 0) {
                const blob = new Blob([new Uint8Array(artworkData)], { type: 'image/jpeg' });
                await ffmpeg.deleteFile(artworkName);
                return URL.createObjectURL(blob);
            }
        } catch {
            // No artwork extracted, that's fine
        }

        return null;
    } catch {
        // Artwork extraction failed, not a big deal
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

        // Extract metadata to ffmetadata format
        await ffmpeg.exec([
            '-i', inputName,
            '-f', 'ffmetadata',
            metadataName
        ]);

        // Read metadata file
        try {
            const metadataBytes = await ffmpeg.readFile(metadataName) as Uint8Array;
            const metadataText = new TextDecoder().decode(metadataBytes);
            await ffmpeg.deleteFile(metadataName);

            // Parse ffmetadata format (key=value lines)
            let title: string | null = null;
            let artist: string | null = null;

            for (const line of metadataText.split('\n')) {
                const [key, ...valueParts] = line.split('=');
                const value = valueParts.join('=').trim();
                
                if (key?.toLowerCase() === 'title' && value) {
                    title = value;
                    debug('Extracted title from ffmpeg:', title);
                }
                if (key?.toLowerCase() === 'artist' && value) {
                    artist = value;
                    debug('Extracted artist from ffmpeg:', artist);
                }
            }

            return { title, artist };
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
    targetSampleRate: number
): Promise<{ buffer: AudioBuffer; artwork: string | null; title: string | null; artist: string | null }> {
    debug('Loading ffmpeg.wasm for decoding...');
    const ffmpeg = await loadFFmpeg();

    // Write input file to virtual filesystem
    const inputName = 'input' + getExtension(fileName);
    await ffmpeg.writeFile(inputName, new Uint8Array(arrayBuffer));

    // Extract artwork first (before modifying the file)
    const artwork = await extractArtworkWithFFmpeg(ffmpeg, inputName);

    // Extract metadata (title, artist)
    const { title, artist } = await extractMetadataWithFFmpeg(ffmpeg, inputName);

    // Convert to WAV format (universally decodable)
    const outputName = 'output.wav';
    await ffmpeg.exec([
        '-i', inputName,
        '-ar', String(targetSampleRate),
        '-ac', '2', // stereo
        '-f', 'wav',
        outputName
    ]);

    // Read output file
    const outputData = await ffmpeg.readFile(outputName);

    // Clean up virtual filesystem
    await ffmpeg.deleteFile(inputName);
    await ffmpeg.deleteFile(outputName);

    // Decode the WAV with native Web Audio API
    const audioContext = new AudioContext({ sampleRate: targetSampleRate });
    const wavData = outputData as Uint8Array;
    const wavBuffer = wavData.buffer.slice(wavData.byteOffset, wavData.byteOffset + wavData.byteLength) as ArrayBuffer;
    const buffer = await audioContext.decodeAudioData(wavBuffer);
    await audioContext.close();

    return { buffer, artwork, title, artist };
}

/**
 * Decode audio using Mediabunny (handles WebCodecs-supported formats)
 */
async function decodeWithMediabunny(
    arrayBuffer: ArrayBuffer,
    targetSampleRate: number,
    audioContext: AudioContext
): Promise<{ buffer: AudioBuffer; artwork: string | null; title: string | null; artist: string | null }> {
    // Create input from array buffer
    const input = new Input({
        formats: ALL_FORMATS,
        source: new BufferSource(arrayBuffer),
    });

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
            debug('Extracted artwork from audio file');
        }
        // Extract title and artist
        if (tags.title) {
            title = tags.title;
            debug('Extracted title:', title);
        }
        if (tags.artist) {
            artist = tags.artist;
            debug('Extracted artist:', artist);
        }
    } catch (e) {
        debug('Could not extract metadata tags:', e);
    }

    // Get audio track
    const audioTrack = await input.getPrimaryAudioTrack();
    if (!audioTrack) {
        input.dispose();
        throw new Error('No audio track found in file');
    }

    // Check if we can decode this track
    const canDecode = await audioTrack.canDecode();
    if (!canDecode) {
        input.dispose();
        throw new Error(`Cannot decode audio codec: ${audioTrack.codec || 'unknown'}`);
    }

    // Get audio properties
    const sampleRate = audioTrack.sampleRate;
    const numberOfChannels = audioTrack.numberOfChannels;
    const duration = await audioTrack.computeDuration();

    // Calculate total samples
    const totalSamples = Math.ceil(duration * sampleRate);

    // Create output AudioBuffer (reuse the caller's AudioContext to avoid
    // leaking contexts; browsers cap the number of live AudioContexts).
    const buffer = audioContext.createBuffer(
        numberOfChannels,
        totalSamples,
        sampleRate
    );

    // Use AudioSampleSink to decode all samples
    const sink = new AudioSampleSink(audioTrack);

    let samplesWritten = 0;

    for await (const sample of sink.samples()) {
        // Copy each channel
        for (let ch = 0; ch < numberOfChannels; ch++) {
            const channelBytesNeeded = sample.allocationSize({ planeIndex: ch, format: 'f32-planar' });
            const channelData = new Float32Array(channelBytesNeeded / 4);
            sample.copyTo(channelData, { planeIndex: ch, format: 'f32-planar' });

            // Copy to output buffer
            const outputChannel = buffer.getChannelData(ch);
            const framesToCopy = Math.min(channelData.length, totalSamples - samplesWritten);
            for (let i = 0; i < framesToCopy; i++) {
                outputChannel[samplesWritten + i] = channelData[i];
            }
        }

        samplesWritten += sample.numberOfFrames;
        sample.close();
    }

    input.dispose();

    // Resample if needed
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
}

/**
 * Decode audio file with two-tier fallback system
 */
export async function decodeAudioFile(
    file: File,
    audioContext: AudioContext
): Promise<DecodeResult> {
    const arrayBuffer = await file.arrayBuffer();

    // Tier 1: Try Mediabunny first (handles most formats via WebCodecs)
    try {
        debug('Attempting to decode with Mediabunny...');
        const { buffer, artwork, title, artist } = await decodeWithMediabunny(arrayBuffer, audioContext.sampleRate, audioContext);
        debug('Successfully decoded with Mediabunny');
        return { buffer, artwork, title, artist, usedFallback: 'mediabunny' };
    } catch (mediabunnyError) {
        debug('Mediabunny decode failed, trying ffmpeg.wasm:', mediabunnyError);
    }

    // Tier 2: Try ffmpeg.wasm (handles exotic codecs like ALAC, WMA, etc)
    try {
        debug('Attempting to decode with ffmpeg.wasm...');
        const { buffer, artwork, title, artist } = await decodeWithFFmpeg(arrayBuffer, file.name, audioContext.sampleRate);
        debug('Successfully decoded with ffmpeg.wasm');
        return { buffer, artwork, title, artist, usedFallback: 'ffmpeg' };
    } catch (ffmpegError) {
        console.error('All decode methods failed:', ffmpegError);
        throw new Error(
            `Unable to decode "${file.name}". This audio format is not supported. ` +
            `Error: ${ffmpegError instanceof Error ? ffmpegError.message : String(ffmpegError)}`
        );
    }
}
