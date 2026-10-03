/** Cache Storage bucket holding complete model artifacts. */
export const MODEL_CACHE_NAME = 'unblend-models';

export interface ModelCacheEntry {
    /**
     * Cache key: the immutable-revision artifact URL. Hugging Face answers it
     * with a no-store redirect to a signed CDN URL that changes per request,
     * so the HTTP cache never hits and the key must be the original URL.
     */
    key: string;
    /** Only a body of exactly this many bytes is stored or served. */
    expectedBytes: number;
}

/** Where model bytes are being read from: the network or Cache Storage. */
export type ModelByteSource = 'network' | 'cache';

/**
 * Return the model bytes for ``url``, serving and filling the Cache Storage
 * entry described by ``cacheEntry`` when one is given. Caching is best
 * effort: without ``caches`` (insecure context, some private modes) or when
 * storage fails (e.g. QuotaExceededError), the bytes are simply not cached.
 *
 * A download whose length differs from ``expectedBytes`` (defaulting to the
 * cache entry's) is rejected rather than handed to ORT, where a truncated or
 * substituted file would otherwise surface as an opaque protobuf error.
 *
 * ``onProgress`` receives the byte source as its third argument, so callers
 * can tell a cache read from a download.
 */
export async function loadModelBytes(
    url: string,
    onProgress: (loaded: number, total: number, source: ModelByteSource) => void,
    cacheEntry?: ModelCacheEntry,
    expectedBytes: number | undefined = cacheEntry?.expectedBytes,
): Promise<Uint8Array<ArrayBuffer>> {
    const cache = cacheEntry ? await openModelCache() : null;
    if (cache && cacheEntry) {
        const cached = await readCached(
            cache, cacheEntry, (loaded, total) => onProgress(loaded, total, 'cache'),
        );
        if (cached) return cached;
    }
    const bytes = await fetchModelBytes(
        url, (loaded, total) => onProgress(loaded, total, 'network'),
    );
    if (expectedBytes !== undefined && bytes.byteLength !== expectedBytes) {
        throw new Error(
            `Downloaded model is ${bytes.byteLength} bytes; expected ${expectedBytes}. `
            + `The download was incomplete or the server returned a different file (${url}).`
        );
    }
    if (cache && cacheEntry && bytes.byteLength === cacheEntry.expectedBytes) {
        await storeCached(cache, cacheEntry, bytes);
    }
    return bytes;
}

/**
 * Delete every model artifact stored by ``Separator.load``'s Cache Storage
 * caching. Loaded separators keep working (their weights already live in
 * ORT); the next load downloads again. Resolves ``true`` if a cache existed
 * and was removed, ``false`` if there was nothing to remove or Cache Storage
 * is unavailable in this context.
 */
export async function clearModelCache(): Promise<boolean> {
    if (typeof caches === 'undefined') return false;
    try {
        return await caches.delete(MODEL_CACHE_NAME);
    } catch {
        return false;
    }
}

async function openModelCache(): Promise<Cache | null> {
    if (typeof caches === 'undefined') return null;
    try {
        return await caches.open(MODEL_CACHE_NAME);
    } catch {
        // Cache Storage can exist yet refuse access (e.g. opaque origins).
        return null;
    }
}

async function readCached(
    cache: Cache,
    entry: ModelCacheEntry,
    onProgress: (loaded: number, total: number) => void,
): Promise<Uint8Array<ArrayBuffer> | null> {
    let response: Response | undefined;
    try {
        response = await cache.match(entry.key);
    } catch {
        return null;
    }
    if (!response) return null;
    try {
        const bytes = await readModelResponse(response, onProgress);
        if (bytes.byteLength === entry.expectedBytes) return bytes;
    } catch {
        // Unreadable entry; drop it and download afresh below.
    }
    await cache.delete(entry.key).catch(() => {});
    return null;
}

async function storeCached(
    cache: Cache,
    entry: ModelCacheEntry,
    bytes: Uint8Array<ArrayBuffer>,
): Promise<void> {
    // A new pinned revision supersedes older cached copies of the same file
    // (they are never read again); drop them first, so they neither leave
    // hundreds of megabytes behind per release nor take the quota the new
    // copy needs.
    const fileName = entry.key.slice(entry.key.lastIndexOf('/') + 1);
    try {
        for (const request of await cache.keys()) {
            if (request.url !== entry.key && request.url.endsWith(`/${fileName}`)) {
                await cache.delete(request);
            }
        }
    } catch {
        // Pruning is housekeeping; storing the new entry can still go ahead.
    }
    try {
        // A BufferSource body is copied when the Response is constructed, which
        // for the largest artifacts would briefly hold a second ~1 GB copy.
        // Streaming views of the same buffer lets Cache Storage consume it
        // incrementally instead.
        await cache.put(entry.key, new Response(streamBytes(bytes), {
            headers: {
                'Content-Type': 'application/octet-stream',
                'Content-Length': String(bytes.byteLength),
            },
        }));
    } catch (error) {
        console.warn('[unblend] could not cache the model; continuing without caching:', error);
    }
}

/** Chunk size for streaming a downloaded model into Cache Storage. */
const CACHE_WRITE_CHUNK_BYTES = 4 * 1024 * 1024;

/** A stream of zero-copy subarray views over ``bytes``, pulled on demand. */
function streamBytes(bytes: Uint8Array<ArrayBuffer>): ReadableStream<Uint8Array<ArrayBuffer>> {
    let offset = 0;
    return new ReadableStream<Uint8Array<ArrayBuffer>>({
        pull(controller) {
            if (offset >= bytes.byteLength) {
                controller.close();
                return;
            }
            const end = Math.min(offset + CACHE_WRITE_CHUNK_BYTES, bytes.byteLength);
            controller.enqueue(bytes.subarray(offset, end));
            offset = end;
        },
    });
}

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
): Promise<Uint8Array<ArrayBuffer>> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }
    return readModelResponse(response, onProgress);
}

async function readModelResponse(
    response: Response,
    onProgress: (loaded: number, total: number) => void
): Promise<Uint8Array<ArrayBuffer>> {
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
