"""
Check each upstream MSST config against unblend's registry entry.

The upstream comparison is only meaningful if both sides build the *same*
model. unblend's registry carries a translated copy of each upstream ``model:``
section, so any disagreement here means one of the two is misconfigured and the
resulting numbers would compare different networks.

This ran clean for both SCNets before the campaign: every key matched, the only
difference being that upstream nests ``sources`` inside ``model:`` while the
registry keeps it at entry level.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CFG = ROOT / "benchmarks" / "msst_configs"

#: Registry model -> upstream config filename.
PAIRS = {
    "bs_roformer_sw": "config_bs_roformer_sw.yaml",
    "bs_roformer_anvuew": "config_bs_roformer_anvuew.yaml",
    "melband_roformer_kim": "config_melband_roformer_kim.yaml",
    "scnet_small": "config_musdb18_scnet_small.yaml",
    "scnet_xl_wide_v5": "config_musdb18_scnet_xl_more_wide_v5.yaml",
}


class _Loader(yaml.SafeLoader):
    """SafeLoader that also accepts ml_collections' python/tuple tag."""


_Loader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def _constructor_defaults(architecture: str) -> dict[str, object]:
    """
    Default value of every constructor argument for an architecture.

    A registry entry only spells out what it needs to change, while an upstream
    config tends to write every key explicitly. Comparing the two literally
    reports a difference for every key we simply left at its default, which
    buries the handful that actually matter.

    :param architecture: Registry architecture name.
    :return: Parameter name to default value.
    """
    from unblend.roformer import BSRoformer, MelBandRoformer
    from unblend.scnet import SCNet, SCNetMasked

    klass = {
        "bs_roformer": BSRoformer,
        "mel_band_roformer": MelBandRoformer,
        "scnet": SCNet,
        "scnet_masked": SCNetMasked,
    }[architecture]
    return {
        name: parameter.default
        for name, parameter in inspect.signature(klass).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _normalise(value: object) -> object:
    """
    Make tuples and lists comparable.

    :param value: Any config value.
    :return: The value with tuples flattened to lists.
    """
    return list(value) if isinstance(value, tuple) else value


def main() -> int:
    """
    Compare every available pair and report differences.

    :return: 0 if every present config agrees, 1 otherwise.
    """
    registry = yaml.safe_load((ROOT / "unblend" / "metadata.yaml").read_text())[
        "models"
    ]
    problems = 0
    for name, config_name in PAIRS.items():
        path = CFG / config_name
        if not path.exists():
            print(f"{name}: SKIP (no {config_name}; run setup.sh)")
            continue
        upstream = yaml.load(path.read_text(), Loader=_Loader)
        ours = registry[name]["config"]
        theirs = dict(upstream.get("model", {}))

        # ``sources`` legitimately lives at entry level for us and inside
        # ``model:`` for upstream; compare it against the entry instead.
        theirs_sources = theirs.pop("sources", None)
        defaults = _constructor_defaults(registry[name]["architecture"])
        # What our side would actually build with: the entry's value if it
        # states one, else the constructor default.
        effective = {**defaults, **ours}
        diffs = []
        ignored = 0
        for key in sorted(set(effective) | set(theirs)):
            if key not in theirs:
                continue  # upstream silent: it takes its own default
            mine, upstream_value = (
                _normalise(effective.get(key)),
                _normalise(theirs.get(key)),
            )
            if mine == upstream_value:
                continue
            if key not in ours and key not in defaults:
                # Training-only key (loss resolutions, checkpointing) that our
                # constructor does not accept at all.
                ignored += 1
                continue
            diffs.append((key, mine, upstream_value))
        audio = upstream.get("audio", {}) or {}
        training = upstream.get("training", {}) or {}
        chunk = audio.get("chunk_size")
        instruments = training.get("instruments") or theirs_sources

        print(f"\n{name}")
        print(
            f"  chunk_size {chunk} vs segment_samples "
            f"{registry[name]['segment_samples']}"
            f"{'  OK' if chunk == registry[name]['segment_samples'] else '  MISMATCH'}"
        )
        if instruments is not None:
            match = list(instruments) == list(registry[name]["sources"])
            print(
                f"  instruments {list(instruments)} vs sources "
                f"{registry[name]['sources']}{'  OK' if match else '  MISMATCH'}"
            )
            problems += not match
        problems += chunk != registry[name]["segment_samples"]
        print(
            f"  model-section differences: {len(diffs)}"
            f"  ({ignored} training-only key(s) ignored)"
        )
        for key, mine, upstream_value in diffs:
            print(f"    {key}: registry={mine!r} upstream={upstream_value!r}")
        problems += len(diffs)
    print(
        f"\n{'OK: all present configs agree' if not problems else f'{problems} disagreement(s)'}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
