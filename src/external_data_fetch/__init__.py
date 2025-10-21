"""External data fetching utilities for land cover and other auxiliary datasets."""

from .fetch_esa_worldcover import (
    fetch_and_process_worldcover,
    one_hot_encode_worldcover,
)

try:
    from .fetch_dynamic_world import (
        fetch_and_process_dynamic_world,
        one_hot_encode_dynamic_world,
    )

    DYNAMIC_WORLD_AVAILABLE = True
except ImportError:
    DYNAMIC_WORLD_AVAILABLE = False

__all__ = [
    "fetch_and_process_worldcover",
    "one_hot_encode_worldcover",
]

if DYNAMIC_WORLD_AVAILABLE:
    __all__.extend(
        [
            "fetch_and_process_dynamic_world",
            "one_hot_encode_dynamic_world",
        ]
    )
