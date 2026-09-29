import os
from typing import Any


def get_config(
    key: str, value: Any | None = None, default: Any | None = None
) -> Any | None:
    """
    Get a configuration value from the environment or a default.
    """
    if value is None:
        value = os.getenv(f"NEWARE_{key.upper()}", default)
    if value is None:
        raise ValueError(
            f"Configuration value for {key} is not set and no default is provided."
        )
    return value
