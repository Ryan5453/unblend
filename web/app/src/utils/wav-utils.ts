export function createWavBlob(
    audioData: Float32Array,
    numChannels: number,
    sampleRate: number
): Blob {
    const numSamples = Math.floor(audioData.length / numChannels);
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numSamples * blockAlign;

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset: number, str: string) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
        for (let c = 0; c < numChannels; c++) {
            let sample = audioData[i * numChannels + c];
            sample = Math.max(-1, Math.min(1, sample));
            // Round (setInt16 would otherwise truncate toward zero) and use
            // the full negative range: -1 maps to -32768, +1 to +32767. The
            // clamp keeps the +1 case from overflowing to -32768.
            const scaled = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
            const q = Math.max(-32768, Math.min(32767, Math.round(scaled)));
            view.setInt16(offset, q, true);
            offset += 2;
        }
    }

    return new Blob([buffer], { type: 'audio/wav' });
}
