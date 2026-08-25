# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Reading and writing the mapping files Unblend uses for configuration.

The registry, user model files, and the training configs ``models import``
translates are all YAML. It is what the ecosystem's configs are already written
in, and it lets a registry carry comments and readable multi-line prose — which
matters for a file whose job includes stating each model's licence.

JSON files are read with the JSON parser rather than YAML's. YAML 1.1 is a
near-superset of JSON, not an exact one, and a file that says ``.json`` deserves
the parser that matches it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_mapping(path: Path) -> Any:
    """
    Parse a YAML (or JSON) file.

    :param path: File to read.
    :return: Whatever it contained; callers check the shape.
    :raises OSError: If the file cannot be read.
    :raises ValueError: If it cannot be parsed.
    """
    text = path.read_text()
    if path.suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(str(exc)) from exc


#: Width at which a string is folded into a block scalar rather than left on
#: one line. Long licence and provenance notes are the reason the registry is
#: YAML at all, so they should read like prose.
_FOLD_WIDTH = 88


class RegistryDumper(yaml.SafeDumper):
    """
    Dumper that writes registries the way they are hand-written: short scalar
    lists inline, long prose folded into block scalars.
    """


def _represent_str(dumper: yaml.Dumper, value: str) -> yaml.Node:
    """
    Fold long single-line strings into block scalars.

    :param dumper: The active dumper.
    :param value: The string being represented.
    :return: Its node.
    """
    style = ">" if len(value) > _FOLD_WIDTH and "\n" not in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


def _represent_list(dumper: yaml.Dumper, value: list) -> yaml.Node:
    """
    Keep short lists of scalars on one line.

    :param dumper: The active dumper.
    :param value: The list being represented.
    :return: Its node.
    """
    inline = (
        bool(value)
        and all(isinstance(item, (int, float, str)) for item in value)
        and len(repr(value)) <= _FOLD_WIDTH
    )
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=inline)


RegistryDumper.add_representer(str, _represent_str)
RegistryDumper.add_representer(list, _represent_list)


def dump_mapping(payload: Any, path: Path) -> str:
    """
    Serialise a mapping in the format ``path`` names.

    :param payload: The mapping to write.
    :param path: Destination, whose suffix picks the format.
    :return: The serialised text.
    """
    if path.suffix == ".json":
        return json.dumps(payload, indent=2) + "\n"
    return yaml.dump(
        payload,
        Dumper=RegistryDumper,
        sort_keys=False,
        allow_unicode=True,
        width=_FOLD_WIDTH,
    )
