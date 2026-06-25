"""Core TIFF/ROI utilities for Femtonics multi-ROI processing."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

try:
    from .tiff_backend import dtype_name_from_pil_image, iter_tiff_frames
except ImportError:  # pragma: no cover - supports direct script execution.
    from tiff_backend import dtype_name_from_pil_image, iter_tiff_frames


TIFF_SUFFIXES = {".tif", ".tiff"}
OUTPUT_FOLDER_NAME = "split_rois"

LogCallback = Callable[[str], None]
FrameProgress = Callable[[int, int], None]


class UnsupportedFastTiffError(RuntimeError):
    """Raised when the optimized TIFF writer cannot safely handle an image."""


@dataclass(frozen=True)
class RoiMetadata:
    width: int
    height: int
    average: int | None
    pattern_num: int | None
    roi_count: int
    roi_sizes: list[tuple[int, int]]
    roi_indices: list[int]
    channel: str | None
    data_type: str | None
    acquisition_type: str | None = None
    comment: str | None = None
    measurement_date: str | None = None
    sampling_rate_hz: float | None = None
    raw_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TiffInfo:
    frame_width: int
    frame_height: int
    frame_count: int
    mode: str
    dtype: str


@dataclass(frozen=True)
class RoiBox:
    ordinal: int
    roi_index: int
    row: int
    column: int
    left: int
    upper: int
    right: int
    lower: int
    output_file: str = ""
    shape_type: str = "rectangle"

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        return (self.left, self.upper, self.right, self.lower)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.lower - self.upper

    def with_output_file(self, output_file: Path) -> "RoiBox":
        return RoiBox(
            ordinal=self.ordinal,
            roi_index=self.roi_index,
            row=self.row,
            column=self.column,
            left=self.left,
            upper=self.upper,
            right=self.right,
            lower=self.lower,
            output_file=str(output_file),
            shape_type=self.shape_type,
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "roi_index": self.roi_index,
            "shape_type": self.shape_type,
            "row": self.row,
            "column": self.column,
            "crop_box": {
                "left": self.left,
                "upper": self.upper,
                "right": self.right,
                "lower": self.lower,
            },
            "display_range": {
                "x": [self.left, self.right - 1],
                "y": [self.upper, self.lower - 1],
            },
            "width": self.width,
            "height": self.height,
            "output_file": self.output_file,
        }


@dataclass(frozen=True)
class RoiLayout:
    order: str
    columns: int
    rows: int
    roi_width: int
    roi_height: int
    boxes: list[RoiBox]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "columns": self.columns,
            "rows": self.rows,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
        }


@dataclass(frozen=True)
class SplitResult:
    ok: bool
    message: str
    tiff_path: Path
    output_dir: Path
    manifest: dict[str, Any]
    error: str | None = None


def read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def parse_int(value: str, field_name: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {field_name}: {value!r}") from exc


def parse_optional_int(fields: Mapping[str, str], key: str) -> int | None:
    if key not in fields:
        return None
    return parse_int(fields[key], key)


def parse_literal(value: str, field_name: str) -> Any:
    try:
        return ast.literal_eval(value.strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid literal for {field_name}: {value!r}") from exc


def extract_sampling_rate_hz(fields_or_text: Mapping[str, str] | str) -> float | None:
    """Extract a sampling rate from text like '6rois-556hz', '556 Hz', or '2.08 kHz'."""
    if isinstance(fields_or_text, str):
        search_values = [fields_or_text]
    else:
        preferred = [
            value
            for key, value in fields_or_text.items()
            if key.lower() in {"comment", "samplingrate", "samplingratehz", "framerate", "frameratehz"}
        ]
        remaining = [value for key, value in fields_or_text.items() if value not in preferred]
        search_values = preferred + remaining

    pattern = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(k(?:ilo)?hz|kilohertz|hz|hertz)\b")
    for value in search_values:
        match = pattern.search(value)
        if match:
            rate = float(match.group(1))
            unit = match.group(2).lower()
            if unit in {"khz", "kilohz", "kilohertz"}:
                rate *= 1000.0
            if rate > 0:
                return rate
    return None


def normalize_roi_sizes(raw_sizes: Any, roi_count: int) -> list[tuple[int, int]]:
    if (
        isinstance(raw_sizes, (list, tuple))
        and len(raw_sizes) == 2
        and all(isinstance(item, (int, float)) for item in raw_sizes)
    ):
        return [(int(raw_sizes[0]), int(raw_sizes[1])) for _ in range(roi_count)]

    if not isinstance(raw_sizes, (list, tuple)):
        raise ValueError("RoiSize must be a list, such as [[10, 10], [10, 10]].")

    sizes: list[tuple[int, int]] = []
    for item in raw_sizes:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid RoiSize entry: {item!r}")
        sizes.append((int(item[0]), int(item[1])))

    if len(sizes) != roi_count:
        raise ValueError(f"RoiSize has {len(sizes)} entries but RoiCount is {roi_count}.")
    return sizes


def normalize_roi_indices(raw_indices: Any, roi_count: int) -> list[int]:
    if raw_indices is None:
        return list(range(1, roi_count + 1))
    if not isinstance(raw_indices, (list, tuple)):
        raise ValueError("RoiIndex must be a list, such as [1, 2, 3].")
    indices = [int(item) for item in raw_indices]
    if len(indices) != roi_count:
        raise ValueError(f"RoiIndex has {len(indices)} entries but RoiCount is {roi_count}.")
    return indices


def parse_metadata(path: Path) -> RoiMetadata:
    text = read_text_lossy(path)
    fields: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith(">") or line.startswith("<"):
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key.startswith("RoiSize"):
            fields["RoiSize"] = value
        else:
            fields[key] = value

    required = ["Width", "Height", "RoiCount", "RoiSize"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Metadata is missing required field(s): {', '.join(missing)}.")

    roi_count = parse_int(fields["RoiCount"], "RoiCount")
    if roi_count <= 0:
        raise ValueError("RoiCount must be greater than zero.")

    raw_roi_indices = parse_literal(fields["RoiIndex"], "RoiIndex") if "RoiIndex" in fields else None
    roi_sizes = normalize_roi_sizes(parse_literal(fields["RoiSize"], "RoiSize"), roi_count)
    roi_indices = normalize_roi_indices(raw_roi_indices, roi_count)

    return RoiMetadata(
        width=parse_int(fields["Width"], "Width"),
        height=parse_int(fields["Height"], "Height"),
        average=parse_optional_int(fields, "Average"),
        pattern_num=parse_optional_int(fields, "PatternNum"),
        roi_count=roi_count,
        roi_sizes=roi_sizes,
        roi_indices=roi_indices,
        channel=fields.get("Channel"),
        data_type=fields.get("DataType"),
        acquisition_type=fields.get("Type"),
        comment=fields.get("Comment"),
        measurement_date=fields.get("MeasurementDate"),
        sampling_rate_hz=extract_sampling_rate_hz(fields),
        raw_fields=dict(fields),
    )


def is_tiff_path(path: Path) -> bool:
    return path.suffix.lower() in TIFF_SUFFIXES


def metadata_path_for_tiff(tiff_path: Path) -> Path:
    return Path(str(tiff_path) + ".metadata.txt")


def tiff_path_for_metadata(metadata_path: Path) -> Path:
    name = metadata_path.name
    if not name.endswith(".metadata.txt"):
        raise ValueError(f"Metadata path does not end with .metadata.txt: {metadata_path}")
    return metadata_path.with_name(name[: -len(".metadata.txt")])


def iter_tiff_inputs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.name.endswith(".metadata.txt"):
            return [tiff_path_for_metadata(input_path)]
        if is_tiff_path(input_path):
            return [input_path]
        raise ValueError(f"Input file is not a TIFF or metadata file: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    found: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(input_path):
            dirnames[:] = [name for name in dirnames if name != OUTPUT_FOLDER_NAME]
            for filename in filenames:
                path = Path(dirpath) / filename
                if is_tiff_path(path):
                    found.append(path)
    else:
        for path in input_path.iterdir():
            if path.is_file() and is_tiff_path(path):
                found.append(path)

    return sorted(found)


def default_split_output_dir(tiff_path: Path, output_root: Path | None = None) -> Path:
    if output_root is None:
        return tiff_path.parent / OUTPUT_FOLDER_NAME / tiff_path.stem
    return output_root / tiff_path.stem


def compute_roi_layout(
    metadata: RoiMetadata,
    output_dir: Path | None = None,
    tiff_stem: str | None = None,
) -> RoiLayout:
    first_size = metadata.roi_sizes[0]
    if any(size != first_size for size in metadata.roi_sizes):
        raise ValueError("Mixed ROI sizes are not supported by this splitter version.")

    roi_width, roi_height = first_size
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError(f"ROI size must be positive, got {first_size}.")

    if metadata.width % roi_width != 0 or metadata.height % roi_height != 0:
        raise ValueError(
            f"ROI size {roi_width}x{roi_height} does not evenly tile "
            f"metadata size {metadata.width}x{metadata.height}."
        )

    columns = metadata.width // roi_width
    rows = metadata.height // roi_height
    if columns * rows != metadata.roi_count:
        raise ValueError(
            f"Computed grid has {columns * rows} tiles ({columns} columns x {rows} rows), "
            f"but RoiCount is {metadata.roi_count}."
        )

    boxes: list[RoiBox] = []
    for ordinal, roi_index in enumerate(metadata.roi_indices, start=1):
        zero_based = ordinal - 1
        row = zero_based // columns
        column = zero_based % columns
        left = column * roi_width
        upper = row * roi_height
        output_file = ""
        if output_dir is not None and tiff_stem is not None:
            output_file = str(output_dir / f"{tiff_stem}_ROI{roi_index:02d}.tiff")
        boxes.append(
            RoiBox(
                ordinal=ordinal,
                roi_index=roi_index,
                row=row,
                column=column,
                left=left,
                upper=upper,
                right=left + roi_width,
                lower=upper + roi_height,
                output_file=output_file,
            )
        )

    return RoiLayout(
        order="row-major",
        columns=columns,
        rows=rows,
        roi_width=roi_width,
        roi_height=roi_height,
        boxes=boxes,
    )


def image_dtype_name(image: Image.Image) -> str:
    return dtype_name_from_pil_image(image)


def load_tiff_info(tiff_path: Path) -> TiffInfo:
    with Image.open(tiff_path) as image:
        return TiffInfo(
            frame_width=image.size[0],
            frame_height=image.size[1],
            frame_count=int(getattr(image, "n_frames", 1)),
            mode=image.mode,
            dtype=image_dtype_name(image),
        )


def validate_tiff_matches_metadata(info: TiffInfo, metadata: RoiMetadata) -> None:
    if (info.frame_width, info.frame_height) != (metadata.width, metadata.height):
        raise ValueError(
            f"TIFF dimensions {info.frame_width}x{info.frame_height} do not match "
            f"metadata size {metadata.width}x{metadata.height}."
        )


def collect_roi_frames(tiff_path: Path, crop_box: tuple[int, int, int, int], frame_count: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    left, upper, right, lower = crop_box
    for frame_number, frame in enumerate(iter_tiff_frames(tiff_path)):
        if frame_number >= frame_count:
            break
        frames.append(Image.fromarray(np.asarray(frame)[upper:lower, left:right]).copy())
    return frames


_FAST_TIFF_DTYPES: dict[np.dtype, tuple[int, int]] = {
    np.dtype("uint8"): (8, 1),
    np.dtype("uint16"): (16, 1),
    np.dtype("int16"): (16, 2),
    np.dtype("uint32"): (32, 1),
    np.dtype("int32"): (32, 2),
    np.dtype("float32"): (32, 3),
}


def _fast_tiff_dtype_spec(dtype: np.dtype) -> tuple[int, int]:
    normalized = np.dtype(dtype).newbyteorder("=")
    if normalized not in _FAST_TIFF_DTYPES:
        raise UnsupportedFastTiffError(f"Fast TIFF writer does not support dtype {dtype}.")
    return _FAST_TIFF_DTYPES[normalized]


def _ifd_entry(tag: int, field_type: int, count: int, value: int) -> bytes:
    if field_type == 3 and count == 1:
        packed_value = struct.pack("<H", int(value)) + b"\x00\x00"
    elif field_type == 4 and count == 1:
        packed_value = struct.pack("<I", int(value))
    else:
        raise UnsupportedFastTiffError("Fast TIFF writer only supports scalar SHORT/LONG IFD values.")
    return struct.pack("<HHI", tag, field_type, count) + packed_value


class ClassicTiffStackWriter:
    """Stream grayscale frames to a classic multi-page TIFF file."""

    def __init__(self, output_path: Path, frame_count: int, width: int, height: int, dtype: np.dtype) -> None:
        if frame_count <= 0:
            raise ValueError("Cannot save an empty TIFF stack.")
        if width <= 0 or height <= 0:
            raise UnsupportedFastTiffError(f"Invalid TIFF frame size {width}x{height}.")

        self.output_path = output_path
        self.frame_count = int(frame_count)
        self.width = int(width)
        self.height = int(height)
        self.dtype = np.dtype(dtype)
        self.bits_per_sample, self.sample_format = _fast_tiff_dtype_spec(self.dtype)
        self.little_dtype = self.dtype.newbyteorder("<")
        self.frame_bytes = int(self.width * self.height * self.dtype.itemsize)
        self.entries_per_frame = 10
        self.ifd_size = 2 + self.entries_per_frame * 12 + 4
        total_size = 8 + self.frame_count * (self.ifd_size + self.frame_bytes)
        if total_size > 0xFFFFFFFF:
            raise UnsupportedFastTiffError("Fast TIFF writer only supports classic TIFF files under 4 GB.")
        self.handle = None

    def __enter__(self) -> "ClassicTiffStackWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output_path.open("wb")
        self.handle.write(b"II")
        self.handle.write(struct.pack("<H", 42))
        self.handle.write(struct.pack("<I", 8))
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj) -> None:
        self.close()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def write_frame(self, frame_number: int, frame: np.ndarray) -> None:
        if self.handle is None:
            raise RuntimeError("TIFF writer is not open.")
        if frame_number < 0 or frame_number >= self.frame_count:
            raise ValueError(f"Frame number {frame_number} is outside 0..{self.frame_count - 1}.")
        array = np.asarray(frame)
        if array.shape != (self.height, self.width):
            raise UnsupportedFastTiffError(
                f"Expected ROI frame shape {(self.height, self.width)}, got {array.shape}."
            )
        if np.dtype(array.dtype) != self.dtype:
            raise UnsupportedFastTiffError(f"Expected ROI dtype {self.dtype}, got {array.dtype}.")

        array = np.ascontiguousarray(array.astype(self.little_dtype, copy=False))
        ifd_offset = self.handle.tell()
        data_offset = ifd_offset + self.ifd_size
        next_ifd_offset = data_offset + self.frame_bytes if frame_number + 1 < self.frame_count else 0
        entries = [
            _ifd_entry(256, 4, 1, self.width),
            _ifd_entry(257, 4, 1, self.height),
            _ifd_entry(258, 3, 1, self.bits_per_sample),
            _ifd_entry(259, 3, 1, 1),
            _ifd_entry(262, 3, 1, 1),
            _ifd_entry(273, 4, 1, data_offset),
            _ifd_entry(277, 3, 1, 1),
            _ifd_entry(278, 4, 1, self.height),
            _ifd_entry(279, 4, 1, self.frame_bytes),
            _ifd_entry(339, 3, 1, self.sample_format),
        ]
        self.handle.write(struct.pack("<H", len(entries)))
        for entry in entries:
            self.handle.write(entry)
        self.handle.write(struct.pack("<I", next_ifd_offset))
        self.handle.write(array.tobytes(order="C"))


def save_tiff_array_stack(output_path: Path, stack: np.ndarray) -> None:
    """Write a grayscale multi-page TIFF stack without thousands of PIL page objects."""
    array = np.asarray(stack)
    if array.ndim != 3:
        raise UnsupportedFastTiffError(f"Expected stack shape (frames, height, width), got {array.shape}.")
    frame_count, height, width = array.shape
    with ClassicTiffStackWriter(output_path, frame_count, width, height, np.dtype(array.dtype)) as writer:
        for frame_number in range(frame_count):
            writer.write_frame(frame_number, array[frame_number])


def stream_roi_stacks_to_tiffs(
    tiff_path: Path,
    boxes: list[RoiBox],
    frame_count: int,
    progress: FrameProgress | None = None,
) -> None:
    """Read one source frame at a time and append each ROI page to its output stack."""
    if not boxes:
        return

    writers: list[ClassicTiffStackWriter] = []
    source_shape: tuple[int, int] | None = None
    source_dtype: np.dtype | None = None

    try:
        for frame_number, frame in enumerate(iter_tiff_frames(tiff_path)):
            if frame_number >= frame_count:
                break
            frame = np.asarray(frame)
            if frame.ndim != 2:
                raise UnsupportedFastTiffError(
                    f"Fast splitter only supports single-channel grayscale frames, got shape {frame.shape}."
                )

            dtype = np.dtype(frame.dtype)
            _fast_tiff_dtype_spec(dtype)
            if source_shape is None:
                source_shape = frame.shape
                source_dtype = dtype
                for box in boxes:
                    writer = ClassicTiffStackWriter(
                        Path(box.output_file),
                        frame_count,
                        box.width,
                        box.height,
                        dtype,
                    )
                    writers.append(writer.__enter__())
            elif frame.shape != source_shape or dtype != source_dtype:
                raise UnsupportedFastTiffError("Fast splitter requires all frames to share shape and dtype.")

            for writer, box in zip(writers, boxes):
                writer.write_frame(frame_number, frame[box.upper : box.lower, box.left : box.right])

            if progress is not None and (
                frame_number == 0
                or (frame_number + 1) % 500 == 0
                or frame_number + 1 == frame_count
            ):
                progress(frame_number + 1, frame_count)

        if source_shape is None:
            raise ValueError("Cannot split an empty TIFF stack.")
    finally:
        for writer in writers:
            writer.close()


def save_tiff_stack(output_path: Path, frames: list[Image.Image]) -> None:
    if not frames:
        raise ValueError("Cannot save an empty TIFF stack.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:])


def format_roi_debug_line(box: RoiBox) -> str:
    return (
        f"ROI{box.roi_index:02d}: "
        f"x={box.left}..{box.right - 1}, y={box.upper}..{box.lower - 1}"
    )


def debug_print(enabled: bool, message: str, log: LogCallback | None = None) -> None:
    if enabled:
        if log is not None:
            log(message)
        else:
            print(message)


def write_debug_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "debug_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_split_manifest(tiff_path: Path, metadata_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "status": "failed",
        "source_tiff": str(tiff_path),
        "metadata_file": str(metadata_path),
        "output_dir": str(output_dir),
        "metadata": None,
        "tiff": None,
        "roi_grid": None,
        "rois": [],
        "errors": [],
    }


def split_tiff(
    tiff_path: Path,
    debug: bool = False,
    output_root: Path | None = None,
    write_manifest: bool | None = None,
    log: LogCallback | None = None,
) -> SplitResult:
    metadata_path = metadata_path_for_tiff(tiff_path)
    output_dir = default_split_output_dir(tiff_path, output_root)
    manifest = build_split_manifest(tiff_path, metadata_path, output_dir)
    should_write_manifest = debug if write_manifest is None else write_manifest

    debug_print(debug, "", log)
    debug_print(debug, f"TIFF: {tiff_path}", log)
    debug_print(debug, f"Metadata: {metadata_path}", log)

    try:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file is missing: {metadata_path}")

        metadata = parse_metadata(metadata_path)
        manifest["metadata"] = asdict(metadata)

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_path, output_dir / metadata_path.name)

        layout = compute_roi_layout(metadata, output_dir, tiff_path.stem)
        info = load_tiff_info(tiff_path)

        manifest["tiff"] = asdict(info)
        manifest["roi_grid"] = layout.as_manifest()
        manifest["rois"] = [box.as_manifest() for box in layout.boxes]

        validate_tiff_matches_metadata(info, metadata)

        debug_print(
            debug,
            f"TIFF info: size={info.frame_width}x{info.frame_height}, "
            f"frames={info.frame_count}, mode={info.mode}, dtype={info.dtype}",
            log,
        )
        debug_print(debug, f"Parsed metadata: {json.dumps(asdict(metadata), sort_keys=True)}", log)
        debug_print(debug, f"Computed ROI grid: columns={layout.columns}, rows={layout.rows}", log)
        for box in layout.boxes:
            debug_print(debug, format_roi_debug_line(box), log)

        try:
            debug_print(debug, "Streaming TIFF frames into ROI stacks...", log)

            def progress(done: int, total: int) -> None:
                debug_print(debug, f"Streamed frames {done}/{total}", log)

            stream_roi_stacks_to_tiffs(
                tiff_path,
                layout.boxes,
                info.frame_count,
                progress=progress if debug else None,
            )
        except UnsupportedFastTiffError as exc:
            debug_print(debug, f"Optimized splitter unavailable ({exc}); using Pillow fallback.", log)
            for box in layout.boxes:
                output_path = Path(box.output_file)
                debug_print(debug, f"Writing {output_path}", log)
                frames = collect_roi_frames(tiff_path, box.crop_box, info.frame_count)
                try:
                    save_tiff_stack(output_path, frames)
                finally:
                    for frame in frames:
                        frame.close()

        manifest["status"] = "success"
        if should_write_manifest:
            write_debug_manifest(output_dir, manifest)

        message = f"[OK] {tiff_path} -> {output_dir} ({len(layout.boxes)} ROI stacks)"
        return SplitResult(True, message, tiff_path, output_dir, manifest)

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        manifest["errors"].append(error_message)
        if debug:
            error_line = f"[ERROR] {tiff_path}: {error_message}"
            if log is not None:
                log(error_line)
            else:
                print(error_line, file=sys.stderr)
        if should_write_manifest and manifest["metadata"] is not None:
            write_debug_manifest(output_dir, manifest)
        message = f"[ERROR] {tiff_path}: {error_message}"
        return SplitResult(False, message, tiff_path, output_dir, manifest, error_message)


def process_tiff(tiff_path: Path, debug: bool) -> tuple[bool, str]:
    """Compatibility wrapper used by the original CLI/test shape."""
    result = split_tiff(tiff_path, debug=debug)
    return result.ok, result.message
