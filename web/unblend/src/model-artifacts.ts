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
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/htdemucs_fp32.onnx',
            sizeBytes: 168679100,
            sha256: '34f88bbb86740c8d9ed5eed2c212e125180465a0154ba51b27b9c00b520b34f0',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/htdemucs_fp16.onnx',
            sizeBytes: 85034599,
            sha256: '7ece9b83307c12120d163bdf558cc2ad1750e4d7dab19846cfa0bc5cb0f22ae9',
        },
    },
    htdemucs_6s: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/htdemucs_6s_fp32.onnx',
            sizeBytes: 110395767,
            sha256: 'c5c1bfe109fcf5d78f72d1633e129c5bc99ce1789031962487041a2b03fdfea0',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/htdemucs_6s_fp16.onnx',
            sizeBytes: 55844998,
            sha256: '7199b39a4dd7cb73fb7debfef81fc6cf36ed78a4cdab8f6fbdf2cb9c07960f0f',
        },
    },
    bs_roformer_sw: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/bs_roformer_sw_fp32.onnx',
            sizeBytes: 700419210,
            sha256: '09585bc2a70dec895ff8a2dd316314850cce4e620304eb46d6926a4e62d12461',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/bs_roformer_sw_fp16.onnx',
            sizeBytes: 351215164,
            sha256: '560ef7646b40c0ed4ae958caacba4fcdc17f08649f8fd56034e5895a7e26ecae',
        },
    },
    melband_roformer_kim: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/melband_roformer_kim_fp32.onnx',
            sizeBytes: 946125448,
            sha256: '8e41cda95e7772ef789740080bc41e9bb3ffd7c489b530644a0b6ab231fdfe99',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/melband_roformer_kim_fp16.onnx',
            sizeBytes: 473549669,
            sha256: '7600639eee90a7e2732b71af90618defac0cacaca5ddfed7b783babfdf8474bf',
        },
    },
    scnet_small: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/scnet_small_fp32.onnx',
            sizeBytes: 48101059,
            sha256: 'eee1bf82b7c9d756a388e00fc142ba1c427444b53a07d626cfa903a450f83281',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/scnet_small_fp16.onnx',
            sizeBytes: 27057704,
            sha256: '2fbb85960a8899f81e1363ccff69cc7299f2059ecad11477c2321eb5e168813d',
        },
    },
    scnet_xl_wide_v5: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/scnet_xl_wide_v5_fp32.onnx',
            sizeBytes: 221482626,
            sha256: '6e2e46b669fa368247afbe867aaa61317425f3f0568309fd6e0c79671c39f0b6',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/45ab0266ad9bacd1fca47cc9bc9bc7f87ca703e3/scnet_xl_wide_v5_fp16.onnx',
            sizeBytes: 114857118,
            sha256: 'd97f0e7d945bfa46a32dca5b3dcd231003ff7f14fc9ab676637e4adab77b91c7',
        },
    },
};

deepFreeze(MODEL_ARTIFACTS);
