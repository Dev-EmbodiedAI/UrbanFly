from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Panel:
    name: str
    x: int
    y: int
    width: int
    height: int
    content_aspect: float

    @property
    def aspect(self) -> float:
        return self.width / self.height


CANVAS_4K = (3840, 2160)
PANELS_4K = (
    Panel("front", 0, 0, 2560, 1440, 16.0 / 9.0),
    Panel("candidates", 2560, 0, 1280, 1440, 8.0 / 9.0),
    Panel("chase", 0, 1440, 1280, 720, 16.0 / 9.0),
    Panel("depth", 1280, 1440, 1200, 720, 5.0 / 3.0),
    Panel("telemetry", 2480, 1440, 1360, 720, 17.0 / 9.0),
)
CANVAS_1080 = (1920, 1080)
PANELS_1080 = tuple(
    Panel(panel.name, panel.x // 2, panel.y // 2, panel.width // 2, panel.height // 2, panel.content_aspect)
    for panel in PANELS_4K
)


def validate_layout(
    canvas: tuple[int, int], panels: tuple[Panel, ...], aspect_tolerance: float = 1e-3
) -> None:
    width, height = canvas
    occupied = set()
    for panel in panels:
        if panel.x < 0 or panel.y < 0 or panel.x + panel.width > width or panel.y + panel.height > height:
            raise ValueError(f"panel {panel.name} is outside the canvas")
        if abs(panel.aspect - panel.content_aspect) > aspect_tolerance:
            raise ValueError(
                f"panel {panel.name} aspect {panel.aspect:.6f} differs from content "
                f"{panel.content_aspect:.6f}"
            )
        for y in range(panel.y, panel.y + panel.height, max(panel.height // 20, 1)):
            for x in range(panel.x, panel.x + panel.width, max(panel.width // 20, 1)):
                point = (x, y)
                if point in occupied:
                    raise ValueError(f"panel {panel.name} overlaps another panel")
                occupied.add(point)
