from uav_wm_navigation.evaluation.video_layout import (
    CANVAS_4K,
    CANVAS_1080,
    PANELS_4K,
    PANELS_1080,
    validate_layout,
)


def test_4k_and_1080_dashboard_panels_are_exact_half_scale() -> None:
    validate_layout(CANVAS_4K, PANELS_4K)
    validate_layout(CANVAS_1080, PANELS_1080)
    for full, half in zip(PANELS_4K, PANELS_1080):
        assert (half.x, half.y, half.width, half.height) == (
            full.x // 2,
            full.y // 2,
            full.width // 2,
            full.height // 2,
        )


def test_metric_depth_panel_preserves_five_by_three_pixel_aspect() -> None:
    depth_4k = next(panel for panel in PANELS_4K if panel.name == "depth")
    depth_1080 = next(panel for panel in PANELS_1080 if panel.name == "depth")
    assert abs(depth_4k.aspect - 160 / 96) < 1e-12
    assert abs(depth_1080.aspect - 160 / 96) < 1e-12
