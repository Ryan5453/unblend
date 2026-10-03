"""Generate the small DSP (STFT) and chunking parity fixtures in test/.

Run from the repo root: uv run python web/unblend/scripts/gen-dsp-fixtures.py web/unblend/test
"""

import json
import math
import sys

import torch

from unblend.blocks import ispectro
from unblend.onnx import (
    compute_roformer_stft_for_export,
    compute_scnet_stft_for_export,
    compute_stft_for_export,
)

out_dir = sys.argv[1]
torch.manual_seed(0)


def r(t):
    return [float(f"{v:.6g}") for v in t.flatten().tolist()]


def mask(C, F, T):
    c = torch.arange(C, dtype=torch.float64).view(C, 1, 1)
    b = torch.arange(F, dtype=torch.float64).view(1, F, 1)
    f = torch.arange(T, dtype=torch.float64).view(1, 1, T)
    return (0.5 + 0.5 * torch.cos(0.3 * b + 0.7 * f + c)).float()


# HTDemucs: nfft/4 hop, segment not a hop multiple (exercises right padding).
nfft, hop, seg = 32, 8, 100
audio = torch.randn(1, 2, seg)
real, imag = compute_stft_for_export(audio, nfft, hop)
m = mask(2, real.shape[-2], real.shape[-1])
z = torch.complex(real[0] * m, imag[0] * m)
# HTDemucs._ispec
zp = torch.nn.functional.pad(z, (0, 0, 0, 1))
zp = torch.nn.functional.pad(zp, (2, 2))
pad = hop // 2 * 3
le = hop * int(math.ceil(seg / hop)) + 2 * pad
x = ispectro(zp, hop, length=le)[..., pad : pad + seg]
json.dump(
    {
        "nfft": nfft,
        "hopLength": hop,
        "segmentSamples": seg,
        "numBins": real.shape[-2],
        "numFrames": real.shape[-1],
        "audio": r(audio[0].T),
        "real": r(real),
        "imag": r(imag),
        "istft": r(x),
    },
    open(f"{out_dir}/htdemucs-stft-fixture.json", "w"),
)

# RoFormer: plain centred Hann STFT, unnormalized; hop not nfft/4.
nfft, hop, seg = 32, 10, 160
audio = torch.randn(1, 2, seg)
real, imag = compute_roformer_stft_for_export(audio, nfft, hop, nfft, False)
m = mask(2, real.shape[-2], real.shape[-1])
z = torch.complex(real[0] * m, imag[0] * m)
x = torch.istft(
    z,
    n_fft=nfft,
    hop_length=hop,
    win_length=nfft,
    window=torch.hann_window(nfft),
    normalized=False,
    length=seg,
)
json.dump(
    {
        "nfft": nfft,
        "hopLength": hop,
        "segmentSamples": seg,
        "numBins": real.shape[-2],
        "numFrames": real.shape[-1],
        "audio": r(audio[0].T),
        "real": r(real),
        "imag": r(imag),
        "istft": r(x),
    },
    open(f"{out_dir}/roformer-stft-fixture.json", "w"),
)

# SCNet: centred, unwindowed (all-ones) STFT, sqrt(N)-normalized. Values are
# stored at full float32 precision; the JS test compares against them directly.
nfft, hop, seg = 256, 64, 1024
torch.manual_seed(0)
audio = torch.randn(1, 2, seg)
real, imag = compute_scnet_stft_for_export(audio, nfft, hop, nfft, True, "none")
json.dump(
    {
        "nfft": nfft,
        "hopLength": hop,
        "segmentSamples": seg,
        "audio": audio[0].T.flatten().tolist(),
        "real": real.flatten().tolist(),
        "imag": imag.flatten().tolist(),
        "numBins": real.shape[-2],
        "numFrames": real.shape[-1],
    },
    open(f"{out_dir}/scnet-stft-fixture.json", "w"),
)

# Chunking / overlap-add parity against apply_model_multi. The fake model
# scales its input by a function of the sample's position inside the model
# window, so the output depends on exactly where every chunk starts, how a
# short final chunk is centered, the triangular weights and the shift offsets.
# SEGMENT_OVERLAP in the JS pipeline is 0.25, matching overlap= below.
from unblend.apply import apply_model_multi  # noqa: E402

seg, n = 100, 290  # stride 75: the last unshifted chunk is 65 samples long
shift_offsets = [0, 22050, 5000]  # max_shift is int(0.5 * 44100) = 22050


class PositionModel(torch.nn.Module):
    samplerate = 44100
    audio_channels = 2
    sources = ["a", "b"]
    max_allowed_segment = seg / 44100

    def forward(self, x):
        length = x.shape[-1]
        g = 1 + torch.arange(length, dtype=x.dtype) / length
        return torch.stack([x * g, x * g * g], dim=1)


mix = 0.5 * torch.randn(1, 2, n, generator=torch.Generator().manual_seed(1))
outputs = {}
for name, offsets in [("shifts0", None), ("shifts3", shift_offsets)]:
    y = apply_model_multi(
        PositionModel(),
        [mix],
        device="cpu",
        shifts=0 if offsets is None else len(offsets),
        overlap=0.25,
        _shift_offsets=None if offsets is None else [[o] for o in offsets],
    )[0]
    outputs[name] = [
        [float(f"{v:.8g}") for v in row] for row in y[0].reshape(-1, n).tolist()
    ]
json.dump(
    {
        "segmentSamples": seg,
        "numSamples": n,
        "shiftOffsets": shift_offsets,
        # float32 values round-trip exactly with 9 significant digits.
        "mix": [[float(f"{v:.9g}") for v in row] for row in mix[0].tolist()],
        # [source * channels + channel][sample]
        "outputs": outputs,
    },
    open(f"{out_dir}/chunking-fixture.json", "w"),
)
