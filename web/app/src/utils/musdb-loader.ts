/**
 * MUSDB18-HQ track discovery from the directory picker (Chromium) or
 * drag-and-drop (any browser). Both paths produce ``MusdbTrack[]``.
 */

export const MUSDB_STEMS = ['drums', 'bass', 'other', 'vocals'] as const;
export type MusdbStem = typeof MUSDB_STEMS[number];

export interface MusdbTrack {
    name: string;
    /** ``mixture.wav`` for this track. */
    mixture: File;
    /** Reference stems keyed by stem name. Missing stems are simply absent. */
    stems: Partial<Record<MusdbStem, File>>;
}

export async function readTracksFromDirectory(
    root: FileSystemDirectoryHandle
): Promise<MusdbTrack[]> {
    const tracks: MusdbTrack[] = [];
    const dirIter = (root as unknown as {
        values: () => AsyncIterable<FileSystemHandle>;
    }).values();

    for await (const entry of dirIter) {
        if (entry.kind !== 'directory') continue;

        const trackDir = entry as FileSystemDirectoryHandle;
        const files = new Map<string, File>();
        const trackIter = (trackDir as unknown as {
            values: () => AsyncIterable<FileSystemHandle>;
        }).values();

        for await (const child of trackIter) {
            if (child.kind !== 'file') continue;
            const fh = child as FileSystemFileHandle;
            const name = fh.name.toLowerCase();
            if (!name.endsWith('.wav')) continue;
            files.set(name, await fh.getFile());
        }

        const mixture = files.get('mixture.wav');
        if (!mixture) continue;

        const stems: Partial<Record<MusdbStem, File>> = {};
        for (const stem of MUSDB_STEMS) {
            const f = files.get(`${stem}.wav`);
            if (f) stems[stem] = f;
        }

        tracks.push({ name: trackDir.name, mixture, stems });
    }

    tracks.sort((a, b) => a.name.localeCompare(b.name));
    return tracks;
}

export function readTracksFromFileList(files: File[]): MusdbTrack[] {
    const byTrack = new Map<string, { mixture?: File; stems: Partial<Record<MusdbStem, File>> }>();

    for (const file of files) {
        const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
        // Expect "<root>/<trackName>/<stem>.wav" or "<trackName>/<stem>.wav".
        const parts = rel.split('/');
        if (parts.length < 2) continue;

        const filename = parts[parts.length - 1].toLowerCase();
        if (!filename.endsWith('.wav')) continue;

        const trackName = parts[parts.length - 2];
        if (!byTrack.has(trackName)) {
            byTrack.set(trackName, { stems: {} });
        }
        const entry = byTrack.get(trackName)!;

        if (filename === 'mixture.wav') {
            entry.mixture = file;
        } else {
            const stem = filename.replace(/\.wav$/, '');
            if ((MUSDB_STEMS as readonly string[]).includes(stem)) {
                entry.stems[stem as MusdbStem] = file;
            }
        }
    }

    const tracks: MusdbTrack[] = [];
    for (const [name, { mixture, stems }] of byTrack) {
        if (!mixture) continue;
        tracks.push({ name, mixture, stems });
    }
    tracks.sort((a, b) => a.name.localeCompare(b.name));
    return tracks;
}

export function supportsDirectoryPicker(): boolean {
    // The dev-only benchmark can force the file-input fallback so automated
    // browser runs can provide a directory without opening a native dialog.
    const forceFileInput = new URLSearchParams(window.location.search).has('fileInput');
    return !forceFileInput
        && typeof (window as unknown as { showDirectoryPicker?: unknown }).showDirectoryPicker === 'function';
}

export async function pickMusdbDirectory(): Promise<FileSystemDirectoryHandle> {
    const showDirectoryPicker = (window as unknown as {
        showDirectoryPicker: (opts?: { mode?: string }) => Promise<FileSystemDirectoryHandle>;
    }).showDirectoryPicker;
    return await showDirectoryPicker({ mode: 'read' });
}
