"""Quality-check image exports for the native Femtonics pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .roi_core import compute_roi_layout, metadata_path_for_tiff, parse_metadata
    from .summary_images import (
        auto_display_range,
        compute_summary_images,
        export_png_scale_factor,
        save_png,
        scale_to_uint8,
        upscale_export_image,
    )
except ImportError:  # pragma: no cover - supports direct script-style imports.
    from roi_core import compute_roi_layout, metadata_path_for_tiff, parse_metadata
    from summary_images import (
        auto_display_range,
        compute_summary_images,
        export_png_scale_factor,
        save_png,
        scale_to_uint8,
        upscale_export_image,
    )


def save_raw_average_png(tiff_path: Path, output_path: Path) -> Path:
    """Write the raw mean projection used as a quick input QC image."""
    summary = compute_summary_images(tiff_path)
    return save_png(summary.mean, output_path)


def _grayscale_rgb(values: np.ndarray, display_range: tuple[float, float] | None = None) -> np.ndarray:
    black, white = display_range if display_range is not None else auto_display_range(values)
    gray = scale_to_uint8(values, black, white)
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def _difference_display_limit(values: np.ndarray) -> float:
    diff = np.asarray(values, dtype=np.float64)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), 99.0))
    if limit <= 0:
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    return limit if limit > 0 else 1.0


def _difference_heatmap(values: np.ndarray, limit: float | None = None) -> np.ndarray:
    diff = np.asarray(values, dtype=np.float64)
    display_limit = _difference_display_limit(diff) if limit is None else float(limit)
    if display_limit <= 0:
        display_limit = 1.0

    normalized = np.clip(diff / display_limit, -1.0, 1.0)
    magnitude = np.power(np.abs(normalized), 0.85)
    neutral = np.array([255.0, 255.0, 255.0], dtype=np.float64)
    negative_color = np.array([59.0, 129.0, 191.0], dtype=np.float64)
    positive_color = np.array([203.0, 74.0, 65.0], dtype=np.float64)
    rgb = np.broadcast_to(neutral, (*diff.shape, 3)).copy()
    positive = normalized > 0
    negative = normalized < 0
    rgb[positive] = neutral + (positive_color - neutral) * magnitude[positive][:, np.newaxis]
    rgb[negative] = neutral + (negative_color - neutral) * magnitude[negative][:, np.newaxis]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), int(size))
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return max(0, int(box[2] - box[0])), max(0, int(box[3] - box[1]))


def _format_scale_value(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _scaled_image_for_qc(rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    scale = export_png_scale_factor(image.width, image.height, min_long_edge=360, max_scale=8)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.BILINEAR)
    return image


def _difference_colorbar(width: int, limit: float) -> Image.Image:
    width = max(180, int(width))
    label_font = _font(13)
    caption_font = _font(13, bold=True)
    bar_height = 16
    tick_height = 5
    label_height = 22
    caption_height = 20
    padding = 12

    gradient = np.linspace(-float(limit), float(limit), width - (2 * padding), dtype=np.float64)
    gradient_rgb = _difference_heatmap(gradient[np.newaxis, :], limit=float(limit))
    gradient_img = Image.fromarray(gradient_rgb, mode="RGB").resize(
        (width - (2 * padding), bar_height),
        resample=Image.Resampling.BILINEAR,
    )

    image = Image.new("RGB", (width, bar_height + tick_height + label_height + caption_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    image.paste(gradient_img, (padding, 0))
    draw.rectangle([padding, 0, width - padding - 1, bar_height - 1], outline=(120, 130, 145), width=1)

    tick_y0 = bar_height
    tick_y1 = bar_height + tick_height
    for x in (padding, width // 2, width - padding - 1):
        draw.line([(x, tick_y0), (x, tick_y1)], fill=(75, 85, 99), width=1)

    labels = [f"-{_format_scale_value(limit)}", "0", f"+{_format_scale_value(limit)}"]
    anchors = [padding, width // 2, width - padding - 1]
    for text, x in zip(labels, anchors):
        text_w, _text_h = _text_size(draw, text, label_font)
        if x == padding:
            text_x = x
        elif x == width - padding - 1:
            text_x = x - text_w
        else:
            text_x = x - text_w // 2
        draw.text((text_x, bar_height + tick_height + 2), text, fill=(55, 65, 81), font=label_font)

    caption = "Delta intensity (corrected - raw, a.u.)"
    caption_w, _caption_h = _text_size(draw, caption, caption_font)
    draw.text(
        ((width - caption_w) // 2, bar_height + tick_height + label_height),
        caption,
        fill=(17, 24, 39),
        font=caption_font,
    )
    return image


def _grayscale_colorbar(width: int, black_level: float, white_level: float) -> Image.Image:
    width = max(220, int(width))
    label_font = _font(13)
    caption_font = _font(13, bold=True)
    bar_height = 16
    tick_height = 5
    label_height = 22
    caption_height = 20
    padding = 12

    gradient = np.linspace(0, 255, width - (2 * padding), dtype=np.uint8)
    gradient_rgb = np.repeat(gradient[np.newaxis, :, np.newaxis], 3, axis=2)
    gradient_img = Image.fromarray(gradient_rgb, mode="RGB").resize(
        (width - (2 * padding), bar_height),
        resample=Image.Resampling.BILINEAR,
    )

    image = Image.new("RGB", (width, bar_height + tick_height + label_height + caption_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    image.paste(gradient_img, (padding, 0))
    draw.rectangle([padding, 0, width - padding - 1, bar_height - 1], outline=(120, 130, 145), width=1)

    tick_y0 = bar_height
    tick_y1 = bar_height + tick_height
    for x in (padding, width - padding - 1):
        draw.line([(x, tick_y0), (x, tick_y1)], fill=(75, 85, 99), width=1)

    labels = [_format_scale_value(black_level), _format_scale_value(white_level)]
    anchors = [padding, width - padding - 1]
    for text, x in zip(labels, anchors):
        text_w, _text_h = _text_size(draw, text, label_font)
        text_x = x if x == padding else x - text_w
        draw.text((text_x, bar_height + tick_height + 2), text, fill=(55, 65, 81), font=label_font)

    caption = "Shared intensity scale (a.u.)"
    caption_w, _caption_h = _text_size(draw, caption, caption_font)
    draw.text(
        ((width - caption_w) // 2, bar_height + tick_height + label_height),
        caption,
        fill=(17, 24, 39),
        font=caption_font,
    )
    return image


def _motion_comparison_figure(
    raw_rgb: np.ndarray,
    corrected_rgb: np.ndarray,
    difference_rgb: np.ndarray,
    *,
    grayscale_range: tuple[float, float],
    difference_limit: float,
) -> Image.Image:
    images = [
        _scaled_image_for_qc(raw_rgb),
        _scaled_image_for_qc(corrected_rgb),
        _scaled_image_for_qc(difference_rgb),
    ]
    panel_width = max(image.width for image in images)
    panel_height = max(image.height for image in images)
    normalized_images: list[Image.Image] = []
    for image in images:
        if image.size != (panel_width, panel_height):
            padded = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
            padded.paste(image, ((panel_width - image.width) // 2, (panel_height - image.height) // 2))
            image = padded
        normalized_images.append(image)

    margin = 32
    gutter = 28
    title_height = 42
    image_y = margin + title_height
    scale_gap = 18
    scale_y = image_y + panel_height + scale_gap
    grayscale_bar = _grayscale_colorbar((panel_width * 2) + gutter, grayscale_range[0], grayscale_range[1])
    difference_bar = _difference_colorbar(panel_width, difference_limit)
    footer_height = max(grayscale_bar.height, difference_bar.height)
    width = (panel_width * 3) + (gutter * 2) + (margin * 2)
    height = margin + title_height + panel_height + scale_gap + footer_height + margin

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18, bold=True)
    titles = ["Raw", "Motion Corrected", "Difference map"]

    x_positions = [margin + idx * (panel_width + gutter) for idx in range(3)]
    for x, image, title in zip(x_positions, normalized_images, titles):
        title_w, _title_h = _text_size(draw, title, title_font)
        draw.text((x + (panel_width - title_w) // 2, margin), title, fill=(17, 24, 39), font=title_font)
        canvas.paste(image, (x, image_y))
        draw.rectangle([x, image_y, x + panel_width - 1, image_y + panel_height - 1], outline=(31, 41, 55), width=1)

    canvas.paste(grayscale_bar, (x_positions[0], scale_y))
    canvas.paste(difference_bar, (x_positions[2], scale_y))
    return canvas


def _labeled_panel(
    rgb: np.ndarray,
    label: str,
    *,
    subtitle: str = "",
    footer: Image.Image | None = None,
) -> Image.Image:
    image = _scaled_image_for_qc(rgb)
    title_font = _font(20, bold=True)
    subtitle_font = _font(13)
    measure = Image.new("RGB", (1, 1), (255, 255, 255))
    measure_draw = ImageDraw.Draw(measure)
    label_width, _label_height = _text_size(measure_draw, label, title_font)
    subtitle_width = _text_size(measure_draw, subtitle, subtitle_font)[0] if subtitle else 0
    footer_width = int(footer.width) if footer is not None else 0
    header_height = 54 if subtitle else 42
    footer_height = int(footer.height) + 12 if footer is not None else 0
    panel_width = max(image.width, label_width + 24, subtitle_width + 24, footer_width + 16)
    panel = Image.new("RGB", (panel_width, image.height + header_height + footer_height), (255, 255, 255))
    panel.paste(image, ((panel_width - image.width) // 2, header_height))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, panel_width, header_height - 1], fill=(244, 246, 248))
    draw.line([(0, header_height - 1), (panel_width, header_height - 1)], fill=(210, 216, 224), width=1)
    draw.text((12, 9), label, fill=(17, 24, 39), font=title_font)
    if subtitle:
        draw.text((12, 34), subtitle, fill=(75, 85, 99), font=subtitle_font)
    if footer is not None:
        footer_y = header_height + image.height + 6
        panel.paste(footer, ((panel_width - footer.width) // 2, footer_y))
    return panel


def save_motion_comparison_png(raw_tiff: Path, corrected_tiff: Path, output_path: Path) -> Path:
    """Write raw/corrected mean projections and a corrected-minus-raw difference map."""
    raw_summary = compute_summary_images(raw_tiff)
    corrected_summary = compute_summary_images(corrected_tiff)
    if raw_summary.mean.shape != corrected_summary.mean.shape:
        raise ValueError(
            "Cannot compare motion correction averages with different shapes: "
            f"{raw_summary.mean.shape} vs {corrected_summary.mean.shape}."
        )

    combined_range = auto_display_range(np.stack([raw_summary.mean, corrected_summary.mean], axis=0))
    difference = corrected_summary.mean - raw_summary.mean
    difference_limit = _difference_display_limit(difference)
    raw_rgb = _grayscale_rgb(raw_summary.mean, combined_range)
    corrected_rgb = _grayscale_rgb(corrected_summary.mean, combined_range)
    heatmap = _difference_heatmap(difference, difference_limit)
    canvas = _motion_comparison_figure(
        raw_rgb,
        corrected_rgb,
        heatmap,
        grayscale_range=combined_range,
        difference_limit=difference_limit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if max(canvas.size) < 1024:
        canvas = upscale_export_image(canvas, resample=Image.Resampling.LANCZOS)
    canvas.save(output_path)
    return output_path


def save_roi_label_map_png(tiff_path: Path, output_path: Path) -> Path:
    """Write a mean-projection ROI map with acquisition rectangles labeled by ROI id."""
    metadata_path = metadata_path_for_tiff(tiff_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file is missing: {metadata_path}")

    metadata = parse_metadata(metadata_path)
    layout = compute_roi_layout(metadata)
    summary = compute_summary_images(tiff_path)
    rgb = _grayscale_rgb(summary.mean)
    image = Image.fromarray(rgb, mode="RGB")
    scale = export_png_scale_factor(image.width, image.height)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    for box in layout.boxes:
        rectangle = [
            int(box.left) * scale,
            int(box.upper) * scale,
            (int(box.right) * scale) - 1,
            (int(box.lower) * scale) - 1,
        ]
        draw.rectangle(rectangle, outline=(0, 220, 255), width=2)
        label = f"ROI{box.roi_index:02d}"
        text_x = int(box.left) * scale + 6
        text_y = int(box.upper) * scale + 6
        text_box = draw.textbbox((text_x, text_y), label)
        draw.rectangle(
            [text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2],
            fill=(0, 0, 0),
        )
        draw.text((text_x, text_y), label, fill=(255, 255, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
