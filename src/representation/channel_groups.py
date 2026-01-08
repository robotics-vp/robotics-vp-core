"""Channel group specs and validation for channel-set encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable, List
import json
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    required: bool = False
    dim_hint: Optional[int] = None
    rate_hint: Optional[str] = None
    token_kind: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class ChannelGroupSpec:
    version: str
    channels: Dict[str, ChannelSpec]

    def required_channels(self) -> List[str]:
        return [name for name, spec in self.channels.items() if spec.required]

    def optional_channels(self) -> List[str]:
        return [name for name, spec in self.channels.items() if not spec.required]


class ChannelSpecError(ValueError):
    """Raised when required channels are missing in channel-set inputs."""


def _parse_channel_spec(raw: Dict[str, Any]) -> ChannelSpec:
    return ChannelSpec(
        name=str(raw.get("name", "")),
        required=bool(raw.get("required", False)),
        dim_hint=raw.get("dim_hint"),
        rate_hint=raw.get("rate_hint"),
        token_kind=raw.get("token_kind"),
        notes=raw.get("notes"),
    )


def load_channel_groups(path: str) -> ChannelGroupSpec:
    if not path:
        raise ValueError("Channel group path is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Channel group config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            payload = json.load(f)
        else:
            if yaml is None:
                raise RuntimeError("pyyaml not installed; use JSON for channel groups")
            payload = yaml.safe_load(f)

    version = str(payload.get("version", "v1"))
    channels_raw = payload.get("channels", []) or []
    channels: Dict[str, ChannelSpec] = {}
    for entry in channels_raw:
        spec = _parse_channel_spec(entry)
        if not spec.name:
            continue
        channels[spec.name] = spec

    if not channels:
        raise ValueError("No channels found in channel group config")

    return ChannelGroupSpec(version=version, channels=channels)


def validate_required_channels(
    tokens_by_channel: Dict[str, Any],
    spec: ChannelGroupSpec,
    mode: str = "eval",
) -> List[str]:
    """Validate required channels for channel-set encoding.

    Args:
        tokens_by_channel: Mapping of channel name -> tokens or provider output.
        spec: Channel group spec.
        mode: train|eval|test (test allows missing required channels).

    Returns:
        List of missing required channels (empty if all present).

    Raises:
        ChannelSpecError if required channels are missing in train/eval mode.
    """
    missing = []
    for name in spec.required_channels():
        if name not in tokens_by_channel:
            missing.append(name)
            continue
        value = tokens_by_channel.get(name)
        if value is None:
            missing.append(name)

    if missing and mode in {"train", "eval"}:
        raise ChannelSpecError(
            "Missing required channels for channel-set encoding: "
            f"{', '.join(sorted(missing))}. "
            "Ensure providers emit these channels or enable synthetic test mode."
        )

    return missing


__all__ = ["ChannelSpec", "ChannelGroupSpec", "ChannelSpecError", "load_channel_groups", "validate_required_channels"]
