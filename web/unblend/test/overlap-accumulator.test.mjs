import assert from 'node:assert/strict';
import test from 'node:test';

import { StreamingOverlapAccumulator } from '../dist/overlap-accumulator.js';

function makeRound(sourceCount, channels, starts, lengths, round) {
    return starts.map((segStart, segment) => ({
        segStart,
        segLength: lengths[segment],
        chunks: Array.from({ length: sourceCount }, (_, source) =>
            Float32Array.from(
                { length: lengths[segment] * channels },
                (_, index) => Math.fround(
                    ((round + 1) * 101 + (source + 1) * 17 + segStart * 3 + index) /
                    997
                )
            )
        ),
    }));
}

function runReference({ sourceCount, channels, viewLength, trimStart, weight, rounds }) {
    const outputs = Array.from(
        { length: sourceCount },
        () => new Float32Array((viewLength - trimStart) * channels)
    );

    for (const segments of rounds) {
        const accumulators = Array.from(
            { length: sourceCount },
            () => new Float32Array(viewLength * channels)
        );
        const weights = new Float32Array(viewLength);

        for (const { chunks, segStart, segLength } of segments) {
            for (let source = 0; source < sourceCount; source++) {
                for (let i = 0; i < segLength; i++) {
                    for (let channel = 0; channel < channels; channel++) {
                        accumulators[source][(segStart + i) * channels + channel] +=
                            chunks[source][i * channels + channel];
                    }
                }
            }
            for (let i = 0; i < segLength; i++) {
                weights[segStart + i] += weight[i];
            }
        }

        for (let source = 0; source < sourceCount; source++) {
            for (let sample = trimStart; sample < viewLength; sample++) {
                for (let channel = 0; channel < channels; channel++) {
                    outputs[source][(sample - trimStart) * channels + channel] +=
                        accumulators[source][sample * channels + channel] /
                        weights[sample];
                }
            }
        }
    }
    return outputs;
}

for (const sourceCount of [1, 4, 6]) {
    test(`streaming overlap-add exactly matches full buffers for ${sourceCount} sources`, () => {
        const channels = 2;
        const segmentSamples = 8;
        const step = 6;
        const viewLength = 25;
        const trimStart = 3;
        const starts = [0, 6, 12, 18, 24];
        const lengths = starts.map(start => Math.min(segmentSamples, viewLength - start));
        const weight = Float32Array.from(
            [0.125, 0.5, 0.875, 1, 0.875, 0.5, 0.25, 0.125]
        );
        const rounds = [0, 1].map(round =>
            makeRound(sourceCount, channels, starts, lengths, round)
        );
        const names = Array.from({ length: sourceCount }, (_, i) => `source-${i}`);
        const outputs = Object.fromEntries(
            names.map(name => [
                name,
                new Float32Array((viewLength - trimStart) * channels),
            ])
        );
        const accumulator = new StreamingOverlapAccumulator(
            outputs,
            names,
            segmentSamples,
            channels,
            step,
        );

        for (const segments of rounds) {
            accumulator.startRound(trimStart, viewLength);
            for (const segment of segments) {
                accumulator.add(
                    segment.chunks,
                    segment.segStart,
                    segment.segLength,
                    weight,
                );
            }
            accumulator.finishRound();
        }

        const expected = runReference({
            sourceCount,
            channels,
            viewLength,
            trimStart,
            weight,
            rounds,
        });
        for (let source = 0; source < sourceCount; source++) {
            assert.deepEqual(outputs[names[source]], expected[source]);
        }
    });
}

test('single-segment ring shorter than the step exactly matches full buffers', () => {
    const sourceCount = 2;
    const channels = 2;
    const step = 6;
    const viewLength = 3;
    const trimStart = 1;
    const weight = Float32Array.from([0.25, 1, 0.25]);
    const rounds = [makeRound(sourceCount, channels, [0], [viewLength], 0)];
    const names = ['left', 'right'];
    const outputs = Object.fromEntries(
        names.map(name => [name, new Float32Array((viewLength - trimStart) * channels)])
    );
    const accumulator = new StreamingOverlapAccumulator(
        outputs,
        names,
        viewLength,
        channels,
        step,
    );

    accumulator.startRound(trimStart, viewLength);
    accumulator.add(rounds[0][0].chunks, 0, viewLength, weight);
    accumulator.finishRound();

    const expected = runReference({
        sourceCount,
        channels,
        viewLength,
        trimStart,
        weight,
        rounds,
    });
    for (let source = 0; source < sourceCount; source++) {
        assert.deepEqual(outputs[names[source]], expected[source]);
    }
});

test('short multi-segment ring exactly matches full buffers', () => {
    const sourceCount = 2;
    const channels = 2;
    const segmentSamples = 8;
    const step = 6;
    const viewLength = 7;
    const trimStart = 2;
    const starts = [0, 6];
    const lengths = [7, 1];
    const weight = Float32Array.from(
        [0.125, 0.5, 0.875, 1, 0.875, 0.5, 0.25, 0.125]
    );
    const rounds = [makeRound(sourceCount, channels, starts, lengths, 0)];
    const names = ['left', 'right'];
    const outputs = Object.fromEntries(
        names.map(name => [name, new Float32Array((viewLength - trimStart) * channels)])
    );
    const accumulator = new StreamingOverlapAccumulator(
        outputs,
        names,
        viewLength,
        channels,
        step,
    );

    accumulator.startRound(trimStart, viewLength);
    for (const segment of rounds[0]) {
        accumulator.add(
            segment.chunks,
            segment.segStart,
            segment.segLength,
            weight,
        );
    }
    accumulator.finishRound();

    const expected = runReference({
        sourceCount,
        channels,
        viewLength,
        trimStart,
        weight,
        rounds,
    });
    for (let source = 0; source < sourceCount; source++) {
        assert.deepEqual(outputs[names[source]], expected[source]);
    }
});
