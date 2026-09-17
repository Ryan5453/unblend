#!/usr/bin/env bash
# Provision the MSST (ZFTurbo/Music-Source-Separation-Training) upstream
# comparison: repo, isolated venv, checkpoints and configs.
#
# unblend's built-in --include-upstream only drives adefossez/demucs, which
# resolves the HTDemucs family and nothing else. RoFormer and SCNet checkpoints
# come from MSST, so comparing them needs this second harness.
#
# Only the four architectures we actually benchmark are installed for. MSST's
# requirements.txt pins a large training stack (asteroid, spafe, timm,
# transformers, segmentation_models_pytorch...) that inference never imports --
# model modules are imported lazily per model_type, so bs_roformer /
# mel_band_roformer need einops + beartype + rotary_embedding_torch + librosa,
# and the SCNets are pure torch.
set -euo pipefail

ROOT="/Users/ryan/Developer/unblend"
HERE="$ROOT/benchmarks/msst_upstream"
REPO="$HERE/msst"
VENV="$HERE/.venv"
CKPT="$HERE/checkpoints"
CFG="$ROOT/benchmarks/msst_configs"

mkdir -p "$HERE" "$CKPT" "$CFG"

if [ ! -d "$REPO/.git" ]; then
    echo "==> cloning MSST"
    git clone --depth 1 -q https://github.com/ZFTurbo/Music-Source-Separation-Training.git "$REPO"
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "==> creating venv"
    uv venv --python 3.11 "$VENV"
    echo "==> installing inference-only deps"
    uv pip install --python "$VENV/bin/python" -q \
        torch torchaudio numpy soundfile librosa einops beartype \
        "rotary_embedding_torch==0.3.5" ml_collections omegaconf tqdm matplotlib scipy pyyaml
fi

dl() { # dl <url> <dest>
    if [ -s "$2" ]; then echo "    have $(basename "$2")"; return; fi
    echo "    fetching $(basename "$2")"
    curl -sL --fail "$1" -o "$2.part" && mv "$2.part" "$2"
}

echo "==> SCNet checkpoints + configs (GitHub release assets)"
B=https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download
dl "$B/v1.0.16/model_scnet_masked_ep_156_sdr_8.8149.ckpt" "$CKPT/scnet_small.ckpt"
dl "$B/v1.0.16/config_musdb18_scnet_small.yaml"           "$CFG/config_musdb18_scnet_small.yaml"
dl "$B/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt"        "$CKPT/scnet_xl_wide_v5.ckpt"
dl "$B/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml" "$CFG/config_musdb18_scnet_xl_more_wide_v5.yaml"

echo "==> RoFormer checkpoints already converted into unblend's cache"
# These two were imported into the model cache as their original .ckpt files;
# reuse rather than re-download ~1.6 GB. Identified by size + tensor layout.
[ -s "$CKPT/bs_roformer_sw.ckpt" ]       || cp ~/.unblend/models/24e7d35ee9c64415.ckpt "$CKPT/bs_roformer_sw.ckpt"
[ -s "$CKPT/melband_roformer_kim.ckpt" ] || cp ~/.unblend/models/87201f4d31afb5bc.ckpt "$CKPT/melband_roformer_kim.ckpt"

echo "==> RoFormer configs"
# anvuew publishes its own config; the SW checkpoint survives only as the
# enerjazzer mirror (jarredou's original account is deleted), which ships the
# matching yaml. Kim's repo has the checkpoint but no config, so MSST's own
# vocals mel-band config stands in -- that is the config MSST itself pairs with
# this checkpoint.
dl "https://huggingface.co/anvuew/BS-RoFormer/resolve/main/config.yaml" \
   "$CFG/config_bs_roformer_anvuew.yaml"
dl "https://huggingface.co/anvuew/BS-RoFormer/resolve/main/bs_roformer_anvuew_sdr_12.45.ckpt" \
   "$CKPT/bs_roformer_anvuew.ckpt"
dl "https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml" \
   "$CFG/config_bs_roformer_sw.yaml"
cp "$REPO/configs/config_vocals_mel_band_roformer.yaml" \
   "$CFG/config_melband_roformer_kim.yaml"

echo "==> done"
ls -la "$CKPT"
echo
echo "Verify each config against unblend's registry before trusting a result:"
echo "  .venv/bin/python benchmarks/msst_upstream/check_configs.py"
