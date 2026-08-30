/**
 * Fetch an ONNX artifact with incremental byte progress.
 *
 * When Content-Length is available, chunks are written directly into one
 * preallocated buffer. Once that buffer is full, the artifact is complete:
 * do not wait for a subsequent stream read just to observe EOF. Some CDNs
 * keep the response stream open after delivering the declared byte count,
 * which otherwise leaves the UI parked at 100% download forever.
 */
export async function fetchModelBytes(
    url: string,
    onProgress: (loaded: number, total: number) => void
): Promise<Uint8Array> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }
    const totalHeader = response.headers.get('Content-Length');
    const contentEncoding = response.headers.get('Content-Encoding');
    const parsedTotal = totalHeader ? Number(totalHeader) : 0;
    // Content-Length describes encoded transfer bytes. A compressed response
    // is decoded by fetch before its chunks reach us, so that length is not a
    // safe allocation size; use the unknown-length fallback in that case.
    const total = (!contentEncoding || contentEncoding === 'identity')
        && Number.isSafeInteger(parsedTotal) && parsedTotal > 0
        ? parsedTotal
        : 0;

    if (!response.body) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        onProgress(bytes.byteLength, total || bytes.byteLength);
        return bytes;
    }

    const reader = response.body.getReader();
    const bytes = total > 0 ? new Uint8Array(total) : null;
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    let lastReport = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (bytes) {
            if (loaded + value.byteLength > bytes.byteLength) {
                await reader.cancel();
                throw new Error(
                    `Model response exceeded Content-Length ${bytes.byteLength}`
                );
            }
            bytes.set(value, loaded);
        } else {
            chunks.push(value);
        }
        loaded += value.byteLength;
        const now = performance.now();
        if (now - lastReport >= 100) {
            onProgress(loaded, total);
            lastReport = now;
        }

        if (bytes && loaded === bytes.byteLength) {
            // The declared response is complete. Do not block on another
            // read merely to observe EOF: large CDN responses can leave that
            // read pending even though every promised byte has arrived.
            void reader.cancel().catch(() => {});
            break;
        }
    }
    onProgress(loaded, total || loaded);

    if (bytes) {
        if (loaded !== bytes.byteLength) {
            throw new Error(
                `Model response ended at ${loaded} bytes; expected ${bytes.byteLength}`
            );
        }
        return bytes;
    }

    const combined = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.byteLength;
    }
    return combined;
}
