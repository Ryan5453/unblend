// Minimal store-only (no compression) ZIP writer. Enough to bundle a handful
// of WAV stems into a single downloadable archive without a dependency.
// There is no ZIP64 support: archives that would exceed the classic format's
// 32-bit sizes/offsets or 16-bit counts are rejected up front instead of
// being written with silently truncated fields.

let crcTable: Uint32Array | null = null;
function getCrcTable(): Uint32Array {
    if (crcTable) return crcTable;
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) {
            c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        }
        table[n] = c >>> 0;
    }
    crcTable = table;
    return table;
}

function crc32(data: Uint8Array): number {
    const table = getCrcTable();
    let crc = 0xffffffff;
    for (let i = 0; i < data.length; i++) {
        crc = (crc >>> 8) ^ table[(crc ^ data[i]) & 0xff];
    }
    return (crc ^ 0xffffffff) >>> 0;
}

export interface ZipEntry {
    name: string;
    data: Uint8Array;
}

/** Largest value a classic (non-ZIP64) 32-bit size/offset field can hold. */
const MAX_UINT32 = 0xffffffff;
/** Largest value a 16-bit count/length field can hold. */
const MAX_UINT16 = 0xffff;
/**
 * General purpose bit 11 (EFS): names are UTF-8. Without it readers assume
 * CP437 and mangle non-ASCII track names such as "Beyoncé".
 */
const FLAG_UTF8 = 0x0800;

/** Throws before any bytes are written if the archive needs ZIP64. */
function assertFitsClassicZip(entries: ZipEntry[], names: Uint8Array[]): void {
    const tooLarge = (what: string) => new Error(
        `Cannot create ZIP: ${what} exceeds the 4 GiB / 65,535-entry limits of ` +
        'the ZIP format without ZIP64, which this writer does not support. ' +
        'Export fewer or shorter stems.'
    );
    if (entries.length > MAX_UINT16) {
        throw tooLarge(`${entries.length} entries`);
    }
    let offset = 0;
    let centralSize = 0;
    entries.forEach((entry, i) => {
        const nameLength = names[i].length;
        if (nameLength > MAX_UINT16) {
            throw new Error(`Cannot create ZIP: entry name is too long (${nameLength} bytes)`);
        }
        // 0xffffffff is reserved as the ZIP64 marker, so it is not a valid size.
        if (entry.data.length >= MAX_UINT32) {
            throw tooLarge(`entry "${entry.name}" (${entry.data.length} bytes)`);
        }
        if (offset >= MAX_UINT32) {
            throw tooLarge(`the offset of entry "${entry.name}"`);
        }
        offset += 30 + nameLength + entry.data.length;
        centralSize += 46 + nameLength;
    });
    if (offset >= MAX_UINT32) {
        throw tooLarge('the central directory offset');
    }
    if (centralSize >= MAX_UINT32) {
        throw tooLarge('the central directory size');
    }
}

export function makeZip(entries: ZipEntry[]): Blob {
    const encoder = new TextEncoder();
    const names = entries.map(entry => encoder.encode(entry.name));
    assertFitsClassicZip(entries, names);
    const chunks: Uint8Array[] = [];
    const central: Uint8Array[] = [];
    let offset = 0;

    for (const [i, entry] of entries.entries()) {
        const nameBytes = names[i];
        const crc = crc32(entry.data);
        const size = entry.data.length;

        // Local file header
        const local = new Uint8Array(30 + nameBytes.length);
        const lv = new DataView(local.buffer);
        lv.setUint32(0, 0x04034b50, true); // signature
        lv.setUint16(4, 20, true); // version needed
        lv.setUint16(6, FLAG_UTF8, true); // flags
        lv.setUint16(8, 0, true); // method: store
        lv.setUint16(10, 0, true); // mod time
        lv.setUint16(12, 0, true); // mod date
        lv.setUint32(14, crc, true);
        lv.setUint32(18, size, true);
        lv.setUint32(22, size, true);
        lv.setUint16(26, nameBytes.length, true);
        lv.setUint16(28, 0, true); // extra length
        local.set(nameBytes, 30);

        chunks.push(local, entry.data);

        // Central directory record
        const cd = new Uint8Array(46 + nameBytes.length);
        const cv = new DataView(cd.buffer);
        cv.setUint32(0, 0x02014b50, true);
        cv.setUint16(4, 20, true); // version made by
        cv.setUint16(6, 20, true); // version needed
        cv.setUint16(8, FLAG_UTF8, true); // flags
        cv.setUint16(10, 0, true);
        cv.setUint16(12, 0, true);
        cv.setUint16(14, 0, true);
        cv.setUint32(16, crc, true);
        cv.setUint32(20, size, true);
        cv.setUint32(24, size, true);
        cv.setUint16(28, nameBytes.length, true);
        cv.setUint16(30, 0, true);
        cv.setUint16(32, 0, true);
        cv.setUint16(34, 0, true);
        cv.setUint16(36, 0, true);
        cv.setUint32(38, 0, true);
        cv.setUint32(42, offset, true);
        cd.set(nameBytes, 46);
        central.push(cd);

        offset += local.length + size;
    }

    const centralSize = central.reduce((n, c) => n + c.length, 0);
    const end = new Uint8Array(22);
    const ev = new DataView(end.buffer);
    ev.setUint32(0, 0x06054b50, true);
    ev.setUint16(8, entries.length, true);
    ev.setUint16(10, entries.length, true);
    ev.setUint32(12, centralSize, true);
    ev.setUint32(16, offset, true);

    return new Blob([...chunks, ...central, end] as BlobPart[], { type: 'application/zip' });
}
