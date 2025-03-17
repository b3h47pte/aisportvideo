from typing import cast

import distinctipy

UNIQUE_COLORS = distinctipy.get_colors(36)


def get_fp_unique_color(index: int) -> tuple[float, float, float]:
    return UNIQUE_COLORS[index % len(UNIQUE_COLORS)]


def get_byte_unique_color(index: int) -> tuple[int, int, int]:
    return cast(
        "tuple[int, int, int]",
        tuple(int(min(max(255 * x, 0), 255)) for x in get_fp_unique_color(index)),
    )
