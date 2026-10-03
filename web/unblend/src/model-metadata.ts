import { SAMPLE_RATE, type ModelConfig } from './constants.js';

/**
 * Read an ONNX model's ``metadata_props`` (ModelProto field 14) straight from
 * its protobuf bytes. onnxruntime-web exposes no model metadata, so this walks
 * only the top-level fields: the multi-hundred-megabyte ``graph`` field is
 * skipped by its length prefix, never decoded. Bytes that stop parsing
 * cleanly yield whatever entries were read so far; ORT reports the real
 * protobuf error when it builds the session.
 */
export function readOnnxMetadata(bytes: Uint8Array): Record<string, string> {
    const metadata: Record<string, string> = {};
    const decoder = new TextDecoder();
    let pos = 0;

    // Varints are accumulated with multiplication, not bit shifts, so lengths
    // past 2^31 (large weights) stay exact.
    const readVarint = (end: number): number | null => {
        let value = 0;
        let scale = 1;
        while (pos < end) {
            const byte = bytes[pos++];
            value += (byte & 0x7f) * scale;
            if (byte < 0x80) return value;
            scale *= 128;
            if (scale > 2 ** 63) return null;
        }
        return null;
    };

    /** Skip one field's payload; return false if the bytes are malformed. */
    const skip = (wireType: number, end: number): boolean => {
        if (wireType === 0) return readVarint(end) !== null;
        if (wireType === 1) pos += 8;
        else if (wireType === 5) pos += 4;
        else if (wireType === 2) {
            const length = readVarint(end);
            if (length === null) return false;
            pos += length;
        } else {
            return false;
        }
        return pos <= end;
    };

    while (pos < bytes.length) {
        const tag = readVarint(bytes.length);
        if (tag === null) break;
        const field = Math.floor(tag / 8);
        const wireType = tag % 8;
        if (field !== 14 || wireType !== 2) {
            if (!skip(wireType, bytes.length)) break;
            continue;
        }
        // StringStringEntryProto: key = 1, value = 2.
        const length = readVarint(bytes.length);
        if (length === null || pos + length > bytes.length) break;
        const end = pos + length;
        let key: string | undefined;
        let value = '';
        while (pos < end) {
            const entryTag = readVarint(end);
            if (entryTag === null) break;
            const entryField = Math.floor(entryTag / 8);
            if (entryTag % 8 === 2 && (entryField === 1 || entryField === 2)) {
                const size = readVarint(end);
                if (size === null || pos + size > end) break;
                const text = decoder.decode(bytes.subarray(pos, pos + size));
                pos += size;
                if (entryField === 1) key = text;
                else value = text;
            } else if (!skip(entryTag % 8, end)) {
                break;
            }
        }
        pos = end;
        if (key !== undefined) metadata[key] = value;
    }
    return metadata;
}

/**
 * Throw if an export's embedded metadata is missing a geometry key or
 * contradicts the geometry ``config`` hard-codes for ``model``. Running a
 * graph with another STFT or segment length does not fail: it silently
 * produces wrong stems.
 */
export function checkModelMetadata(
    metadata: Record<string, string>,
    model: string,
    config: ModelConfig,
): void {
    const problems: string[] = [];
    const expectInt = (key: string, expected: number) => {
        const raw = metadata[key];
        if (raw === undefined) problems.push(`${key} is missing`);
        else if (Number(raw) !== expected) problems.push(`${key} is ${raw}, expected ${expected}`);
    };

    expectInt('sample_rate', SAMPLE_RATE);
    expectInt('audio_channels', 2);
    expectInt('stft_n_fft', config.nfft);
    expectInt('stft_hop_length', config.hopLength);
    expectInt('stft_win_length', config.nfft);
    // SCNet's graph length includes its internal padding.
    expectInt('segment_samples', config.modelInputSamples ?? config.segmentSamples);
    if (config.modelInputSamples !== undefined) {
        expectInt('logical_segment_samples', config.segmentSamples);
    }
    if (config.family === 'roformer') expectInt('num_stems', config.modelSources.length);

    const window = metadata.stft_window;
    const expectedWindow = config.window === 'rectangular' ? 'none' : 'hann';
    if (window !== expectedWindow) {
        problems.push(`stft_window is ${window ?? 'missing'}, expected ${expectedWindow}`);
    }

    const normalized = metadata.stft_normalized;
    // HTDemucs always uses a normalized STFT; the other families opt in.
    const expectedNormalized = config.family === 'htdemucs' || config.stftNormalized === true;
    if (normalized !== String(expectedNormalized)) {
        problems.push(`stft_normalized is ${normalized ?? 'missing'}, expected ${expectedNormalized}`);
    }

    let listed: string | undefined;
    try {
        const parsed: unknown = JSON.parse(metadata.sources ?? '');
        if (Array.isArray(parsed)) listed = parsed.map(String).join(',');
    } catch {
        // Reported below.
    }
    // Single-mask exports list the complement stem too.
    if (listed !== config.modelSources.join(',') && listed !== config.sources.join(',')) {
        problems.push(`sources are ${metadata.sources ?? 'missing'}, expected [${config.sources.join(', ')}]`);
    }

    if (problems.length > 0) {
        throw new Error(
            `This ONNX file's metadata does not match the '${model}' configuration: `
            + `${problems.join('; ')}. The browser pipeline runs every '${model}' file with `
            + `its built-in STFT and chunk geometry, so a custom modelUrl must be an export `
            + `of that model.`
        );
    }
}
