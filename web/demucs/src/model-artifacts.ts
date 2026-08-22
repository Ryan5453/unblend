import type { ModelType } from './constants.js';

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
 * progress; this briefly doubles peak memory (fetched buffer + ORT's parsed
 * copy) instead of ORT streaming the file on its own. The checked-in
 * size/digest contract is verified by `npm run verify:model-artifacts` before
 * a release, not by hashing the buffer at load time.
 */
export const MODEL_ARTIFACTS: Record<
    ModelType,
    Record<ArtifactPrecision, ModelArtifact>
> = {
    htdemucs: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/htdemucs_fp32.onnx',
            sizeBytes: 168678764,
            sha256: 'b067d9ca7f3a93a0c41920a864481dd7a308ce16d20ed144ba41490d5e31a3ce',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/htdemucs_fp16.onnx',
            sizeBytes: 91324835,
            sha256: 'a7efcbad9625cbdde3f00967f75d6ba728384d825c2c92ab479938570007ab17',
        },
    },
    htdemucs_6s: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/htdemucs_6s_fp32.onnx',
            sizeBytes: 110395431,
            sha256: '38ad2757bd1a9aca34ecb68af38106fa75efc6e018a24f62dd1993ec74acf25d',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/htdemucs_6s_fp16.onnx',
            sizeBytes: 59382714,
            sha256: '0fcaed84ca1f48781db053a5dc44f379cefc29734e36200cf05941aa03a40388',
        },
    },
    bs_roformer_sw: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/a80a71b41face40edc91178c07edfedeca4cbb19/bs_roformer_sw_fp32.onnx',
            sizeBytes: 713020597,
            sha256: 'ec8f26334000e982a05365a88fb77672d9ddb10a140adfeb72e9ef572082be8f',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/a80a71b41face40edc91178c07edfedeca4cbb19/bs_roformer_sw_fp16.onnx',
            sizeBytes: 363867964,
            sha256: '3c687f57679321e4c8ab35c267630a74605a8ce786f27b071000adca3c16218e',
        },
    },
    melband_roformer_kim: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/a80a71b41face40edc91178c07edfedeca4cbb19/melband_roformer_kim_fp32.onnx',
            sizeBytes: 951444823,
            sha256: 'fab1113cdfee5c8ab724223e8329d91b6a76767d58df36d1e2ab245c3413af9f',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/a80a71b41face40edc91178c07edfedeca4cbb19/melband_roformer_kim_fp16.onnx',
            sizeBytes: 478901267,
            sha256: '96d42889773713979b2c7e2f6b168942357a009fe516cc554df021321a1a89c6',
        },
    },
    scnet_small: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/ac4b06164d974e1242bd9fc7585305e5ea022d0f/scnet_small_fp32.onnx',
            sizeBytes: 50197643,
            sha256: 'ff6ee6bba0f64d5ded6b540ea2fbc29b4ba169bddba44c287d56a6e4c06aaeec',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/ac4b06164d974e1242bd9fc7585305e5ea022d0f/scnet_small_fp16.onnx',
            sizeBytes: 29081412,
            sha256: 'dd421539061d3b2909be4fa7aa18d66e95de44ad75c63c147c90a8e6fc12f62a',
        },
    },
};
