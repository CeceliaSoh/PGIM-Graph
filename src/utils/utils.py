from __future__ import annotations

from collections.abc import Mapping, Sequence


def check_config_keys(config: Mapping, required_keys: Sequence[str], config_name: str = "config") -> None:
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing {config_name} config keys: {missing_keys}")
