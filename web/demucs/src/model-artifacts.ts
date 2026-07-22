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
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/bs_roformer_sw_fp32.onnx',
            sizeBytes: 708641242,
            sha256: '987f402a9aa572518633408e83b2b0f31600f4f382afc23964e3a19362651f51',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/bs_roformer_sw_fp16.onnx',
            sizeBytes: 360381798,
            sha256: '06e8d57ec29f99d54039bbdd738416f2f430abe37a48dbfc4b4df45412048996',
        },
    },
    melband_roformer_kim: {
        fp32: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/melband_roformer_kim_fp32.onnx',
            sizeBytes: 949386939,
            sha256: 'd09b0337efe649666decd73be39bed6e1b5b69e230e43da0f9d16a518871f46e',
        },
        fp16: {
            url: 'https://huggingface.co/Ryan5453/unblend/resolve/eda32466a76dc81c5e66af6577dbc20fb219e959/melband_roformer_kim_fp16.onnx',
            sizeBytes: 477311000,
            sha256: '701bb8771efe4488aa7af784f6cecf7b171da5c61e520e353174304f84698156',
        },
    },
};
