import type { ModelType } from './constants.js';
import { deepFreeze } from './constants.js';

export type ArtifactPrecision = 'fp32' | 'fp16';

export interface ModelArtifact {
    /** Immutable Hugging Face revision URL for the complete ONNX file. */
    readonly url: string;
    /** Exact byte length attested before publication. */
    readonly sizeBytes: number;
    /** SHA-256 of the published ONNX bytes. */
    readonly sha256: string;
}

/**
 * Browser model artifacts published at one immutable Hugging Face revision.
 *
 * The onnx worker fetches these URLs itself (rather than handing them to
 * `InferenceSession.create` directly) so it can report real download
 * progress and keep them in Cache Storage keyed by URL; this briefly doubles
 * peak memory (fetched buffer + ORT's parsed copy) instead of ORT streaming
 * the file on its own. The checked-in
 * size/digest contract is verified by `npm run verify:model-artifacts` before
 * a release, not by hashing the buffer at load time.
 */
export const MODEL_ARTIFACTS: Readonly<
    Record<ModelType, Readonly<Record<ArtifactPrecision, ModelArtifact>>>
> = {
    htdemucs: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/htdemucs_fp32.onnx',
            sizeBytes: 168679107,
            sha256: '2ccd45eeef8dda37e877ddfbb05ca9a77b5a9f17f3ee5f6091206f42b0700ff4',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/htdemucs_fp16.onnx',
            sizeBytes: 85034606,
            sha256: '742a4f8594a48d6306da303b8b2dd8be2b54389b88557db034408b436effd775',
        },
    },
    htdemucs_6s: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/htdemucs_6s_fp32.onnx',
            sizeBytes: 110395774,
            sha256: '4d22f22858ee831f9165ecc8ee63439666e4d0a99ff9088770508b94896437e9',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/htdemucs_6s_fp16.onnx',
            sizeBytes: 55845005,
            sha256: '8cde625ece6fddb022a1a021cc1f2bef65a16711c20710e0e96e317f38fe994c',
        },
    },
    bs_roformer_sw: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/bs_roformer_sw_fp32.onnx',
            sizeBytes: 700419210,
            sha256: '09585bc2a70dec895ff8a2dd316314850cce4e620304eb46d6926a4e62d12461',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/bs_roformer_sw_fp16.onnx',
            sizeBytes: 351215164,
            sha256: '560ef7646b40c0ed4ae958caacba4fcdc17f08649f8fd56034e5895a7e26ecae',
        },
    },
    melband_roformer_kim: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/melband_roformer_kim_fp32.onnx',
            sizeBytes: 946125448,
            sha256: '8e41cda95e7772ef789740080bc41e9bb3ffd7c489b530644a0b6ab231fdfe99',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/melband_roformer_kim_fp16.onnx',
            sizeBytes: 473549669,
            sha256: '7600639eee90a7e2732b71af90618defac0cacaca5ddfed7b783babfdf8474bf',
        },
    },
    scnet_small: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/scnet_small_fp32.onnx',
            sizeBytes: 48101059,
            sha256: 'eee1bf82b7c9d756a388e00fc142ba1c427444b53a07d626cfa903a450f83281',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/scnet_small_fp16.onnx',
            sizeBytes: 27057704,
            sha256: '2fbb85960a8899f81e1363ccff69cc7299f2059ecad11477c2321eb5e168813d',
        },
    },
    scnet_xl_wide_v5: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/scnet_xl_wide_v5_fp32.onnx',
            sizeBytes: 221482626,
            sha256: '6e2e46b669fa368247afbe867aaa61317425f3f0568309fd6e0c79671c39f0b6',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/30b749fd691312b1e4b3f8fe79df0b51cde836cd/scnet_xl_wide_v5_fp16.onnx',
            sizeBytes: 114857118,
            sha256: 'd97f0e7d945bfa46a32dca5b3dcd231003ff7f14fc9ab676637e4adab77b91c7',
        },
    },
};

deepFreeze(MODEL_ARTIFACTS);
