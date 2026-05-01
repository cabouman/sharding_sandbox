"""Render JAX sharded arrays as rich panels showing device placement and values."""
from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _start(s) -> int:
    if isinstance(s, slice):
        return s.start if s.start is not None else 0
    return int(s)


def _starts(index: tuple, ndim: int) -> tuple[int, ...]:
    return tuple(_start(index[i] if i < len(index) else slice(None)) for i in range(ndim))


# (border/accent color, background fill color) — backgrounds are dark so default text stays legible.
_PALETTE = (
    ("cyan", "dark_cyan"),
    ("magenta", "dark_magenta"),
    ("green", "dark_green"),
    ("yellow", "yellow4"),
    ("blue", "dark_blue"),
    ("red", "dark_red"),
    ("bright_cyan", "deep_sky_blue4"),
    ("bright_magenta", "deep_pink4"),
)


def _format(data: np.ndarray) -> str:
    return np.array2string(data, precision=3, separator=" ", suppress_small=True)


class _HostDevice:
    id = -1
    def __str__(self) -> str:
        return "host"


class _HostShard:
    def __init__(self, data: np.ndarray, ndim: int) -> None:
        self.data = data
        self.device = _HostDevice()
        self.index = (slice(None),) * ndim


def visualize_with_values(arr, *, title: str | None = None, console: Console | None = None) -> None:
    """Show each addressable shard's values inside a panel labeled with its device, laid out
    in a grid that mirrors the sharding. Supports 1-D and 2-D arrays. Replicated shards
    (same index on multiple devices) are coalesced into one panel listing all devices.
    Plain numpy arrays (no sharding) render as a single panel labeled 'host'."""
    # Treat as "host" if it's a plain numpy array OR a jax array that hasn't been explicitly
    # placed on a device (uncommitted). Only deliberately-placed arrays show device labels.
    is_host = not hasattr(arr, "addressable_shards") or not getattr(arr, "committed", True)
    arr_np = np.asarray(arr) if is_host else None
    ndim = arr_np.ndim if arr_np is not None else arr.ndim
    if ndim not in (1, 2):
        raise ValueError(f"Only 1-D and 2-D arrays supported, got ndim={ndim}")

    console = console or Console()

    by_pos: dict[tuple[int, ...], list] = {}
    if arr_np is not None:
        by_pos[(0,) * ndim] = [_HostShard(arr_np, ndim)]
    else:
        for s in arr.addressable_shards:
            by_pos.setdefault(_starts(s.index, ndim), []).append(s)

    if ndim == 1:
        row_keys = sorted(by_pos.keys())
        col_keys = [(0,)]
        lookup = lambda r, c: by_pos.get(row_keys[r])
    else:
        row_keys = sorted({k[0] for k in by_pos})
        col_keys = sorted({k[1] for k in by_pos})
        lookup = lambda r, c: by_pos.get((row_keys[r], col_keys[c]))

    color_for: dict[int, tuple[str, str]] = {}
    for entry in by_pos.values():
        for s in entry:
            if s.device.id == _HostDevice.id:
                color_for.setdefault(s.device.id, ("yellow", "black"))
            else:
                color_for.setdefault(s.device.id, _PALETTE[len(color_for) % len(_PALETTE)])

    def _panel_for(shard) -> Panel:
        data = np.asarray(shard.data)
        accent, bg = color_for[shard.device.id]
        return Panel(
            _format(data),
            title=f"[bold {accent}]{shard.device}[/]",
            title_align="left",
            border_style=accent,
            style=f"on {bg}",
            expand=False,
        )

    grid = Table.grid(padding=(0, 1))
    for _ in range(len(col_keys)):
        grid.add_column()
    for r in range(len(row_keys)):
        cells = []
        for c in range(len(col_keys)):
            entry = lookup(r, c)
            if entry is None:
                cells.append("")
                continue
            # Each device at this position gets its own panel; replicated shards
            # render side-by-side so the duplicated values are visible.
            if len(entry) == 1:
                cells.append(_panel_for(entry[0]))
            else:
                sub = Table.grid(padding=(0, 1))
                for _ in range(len(entry)):
                    sub.add_column()
                sub.add_row(*(_panel_for(s) for s in entry))
                cells.append(sub)
        grid.add_row(*cells)

    console.print(Panel(grid, title=title) if title else grid)
