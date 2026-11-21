export interface ZipSource {
    name: string;
    blob: Blob;
}

export type ZipExportResponse = { blob: Blob } | { error: string };

export async function buildStemZip(sources: ZipSource[]): Promise<Blob> {
    const worker = new Worker(
        new URL('../workers/zip-export-worker.ts', import.meta.url),
        { type: 'module' },
    );
    try {
        return await new Promise<Blob>((resolve, reject) => {
            worker.onmessage = ({ data }: MessageEvent<ZipExportResponse>) => {
                if ('error' in data) reject(new Error(data.error));
                else resolve(data.blob);
            };
            worker.onerror = event => reject(new Error(event.message || 'ZIP export worker failed'));
            worker.onmessageerror = () => reject(new Error('Could not read ZIP export worker response'));
            worker.postMessage(sources);
        });
    } finally {
        worker.terminate();
    }
}
