#!/usr/bin/env node

/** Stream every published ONNX artifact and verify its checked-in contract. */

import { createHash } from 'node:crypto';
import { MODEL_ARTIFACTS } from '../dist/model-artifacts.js';

for (const [model, variants] of Object.entries(MODEL_ARTIFACTS)) {
    for (const [precision, artifact] of Object.entries(variants)) {
        if (artifact.url.includes('/resolve/main/')) {
            throw new Error(`${model}/${precision} is not pinned to an immutable revision`);
        }
        const response = await fetch(artifact.url, { redirect: 'follow' });
        if (!response.ok || !response.body) {
            throw new Error(
                `${model}/${precision} returned HTTP ${response.status} for ${artifact.url}`,
            );
        }

        const hash = createHash('sha256');
        let size = 0;
        for await (const chunk of response.body) {
            hash.update(chunk);
            size += chunk.byteLength;
        }
        const digest = hash.digest('hex');
        if (size !== artifact.sizeBytes) {
            throw new Error(
                `${model}/${precision} size mismatch: expected ${artifact.sizeBytes}, got ${size}`,
            );
        }
        if (digest !== artifact.sha256) {
            throw new Error(
                `${model}/${precision} SHA-256 mismatch: expected ${artifact.sha256}, got ${digest}`,
            );
        }
        console.log(`${model}/${precision}: ${size} bytes ${digest}`);
    }
}
