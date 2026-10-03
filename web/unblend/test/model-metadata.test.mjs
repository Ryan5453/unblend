import assert from 'node:assert/strict';
import test from 'node:test';

import { MODEL_CONFIGS } from '../dist/constants.js';
import { checkModelMetadata, readOnnxMetadata } from '../dist/model-metadata.js';

function varint(n) {
    const out = [];
    while (n >= 0x80) {
        out.push((n % 128) | 0x80);
        n = Math.floor(n / 128);
    }
    out.push(n);
    return out;
}

function lengthDelimited(field, payload) {
    return [...varint(field * 8 + 2), ...varint(payload.length), ...payload];
}

function entry(key, value) {
    const enc = new TextEncoder();
    return lengthDelimited(14, [
        ...lengthDelimited(1, [...enc.encode(key)]),
        ...lengthDelimited(2, [...enc.encode(value)]),
    ]);
}

/** A ModelProto with ir_version, a stand-in graph, then metadata_props. */
function model(props) {
    const bytes = [
        ...varint(1 * 8 + 0), ...varint(10),
        ...lengthDelimited(7, new Array(300).fill(7)),
        ...Object.entries(props).flatMap(([k, v]) => entry(k, v)),
    ];
    return new Uint8Array(bytes);
}

// Metadata of the published exports, as `unblend export-onnx` writes it.
const roformer = {
    sources: '["bass", "drums", "other", "vocals", "guitar", "piano"]',
    sample_rate: '44100',
    audio_channels: '2',
    weight_precision: 'fp32',
    compute_precision: 'fp32',
    model_family: 'roformer',
    architecture: 'bs_roformer',
    segment_samples: '588800',
    stft_n_fft: '2048',
    stft_hop_length: '512',
    stft_win_length: '2048',
    stft_normalized: 'false',
    stft_window: 'hann',
    batch_mode: 'static',
    external_normalization: 'false',
    license: 'unlicensed',
    num_stems: '6',
    output_complement: 'false',
};
const htdemucs = {
    sources: '["drums", "bass", "other", "vocals"]',
    sample_rate: '44100',
    audio_channels: '2',
    model_family: 'demucs',
    segment_samples: '343980',
    stft_n_fft: '4096',
    stft_hop_length: '1024',
    stft_win_length: '4096',
    stft_normalized: 'true',
    stft_window: 'hann',
    stft_pad_samples: '1536',
    stft_frame_trim: '2',
};
const melband = {
    ...roformer,
    sources: '["vocals", "other"]',
    architecture: 'mel_band_roformer',
    segment_samples: '352800',
    stft_hop_length: '441',
    num_stems: '1',
    output_complement: 'true',
};

test('reads metadata_props past a skipped graph field', () => {
    assert.deepEqual(readOnnxMetadata(model(roformer)), roformer);
});

test('truncated bytes yield the entries read so far', () => {
    const bytes = model({ a: '1', b: '2' });
    assert.deepEqual(readOnnxMetadata(bytes.subarray(0, bytes.length - 2)), { a: '1' });
});

test('published metadata passes', () => {
    checkModelMetadata(roformer, 'bs_roformer_sw', MODEL_CONFIGS.bs_roformer_sw);
    checkModelMetadata(htdemucs, 'htdemucs', MODEL_CONFIGS.htdemucs);
    checkModelMetadata(melband, 'melband_roformer_kim', MODEL_CONFIGS.melband_roformer_kim);
});

test('a contradicting or incomplete export is rejected with every problem named', () => {
    assert.throws(
        () => checkModelMetadata(
            { ...roformer, stft_hop_length: '441', stft_window: 'none' },
            'bs_roformer_sw',
            MODEL_CONFIGS.bs_roformer_sw,
        ),
        /stft_hop_length is 441, expected 512; stft_window is none, expected hann/,
    );
    assert.throws(
        () => checkModelMetadata({ ...htdemucs, sources: '["vocals"]' }, 'htdemucs',
            MODEL_CONFIGS.htdemucs),
        /sources are \["vocals"\]/,
    );
    // Metadata from before the current keys, e.g. an old export's.
    assert.throws(
        () => checkModelMetadata({ sources: htdemucs.sources }, 'htdemucs', MODEL_CONFIGS.htdemucs),
        /sample_rate is missing.*stft_window is missing/,
    );
});
