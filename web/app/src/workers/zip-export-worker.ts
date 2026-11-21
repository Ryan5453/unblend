/// <reference lib="webworker" />

import { makeZip } from '../utils/zip';
import type { ZipExportResponse, ZipSource } from '../utils/zip-export';

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

workerScope.onmessage = async ({ data: sources }: MessageEvent<ZipSource[]>) => {
    let response: ZipExportResponse;
    try {
        const entries = await Promise.all(sources.map(async ({ name, blob }) => ({
            name,
            data: new Uint8Array(await blob.arrayBuffer()),
        })));
        response = { blob: makeZip(entries) };
    } catch (error) {
        response = { error: error instanceof Error ? error.message : String(error) };
    }
    workerScope.postMessage(response);
};
