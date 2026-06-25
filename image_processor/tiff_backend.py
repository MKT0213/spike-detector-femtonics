"""Fast TIFF access and preview helpers for interactive image processing."""

from __future__ import annotations

import math
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable, Iterator

import numpy as np
from PIL import Image

try:  # Optional fast path for scientific TIFF stacks.
    import tifffile
except Exception:  # pragma: no cover - depends on the user's environment.
    tifffile = None

try:  # Optional persistent preview cache.
    import zarr
except Exception:  # pragma: no cover - depends on the user's environment.
    zarr = None


PIL_MODE_DTYPES = {
    "1": "bool",
    "L": "uint8",
    "P": "uint8",
    "I;16": "uint16",
    "I;16L": "uint16",
    "I;16B": "uint16",
    "I;16N": "uint16",
    "I": "int32",
    "F": "float32",
    "RGB": "uint8",
    "RGBA": "uint8",
}

PREVIEW_CACHE_VERSION = 1
PREVIEW_CACHE_FOLDER = ".image_processor_cache"


@dataclass(frozen=True)
class PreviewFrame:
    tiff_path: Path
    frame_index: int
    frame_count: int
    source_shape: tuple[int, ...]
    source_dtype: str
    preview_frame: np.ndarray
    downsample: int
    observed_range: tuple[float, float]
    auto_range: tuple[float, float]
    dtype_range: tuple[float, float]
    backend: str
    elapsed_ms: float


def available_backend_name() -> str:
    return "tifffile" if tifffile is not None else "pillow"


def persistent_preview_cache_available() -> bool:
    return zarr is not None


def dtype_name_from_pil_image(image: Image.Image) -> str:
    dtype = PIL_MODE_DTYPES.get(image.mode)
    if dtype is not None:
        return dtype
    return str(np.asarray(image).dtype)


def _series_pages(tiff) -> object | None:
    if not getattr(tiff, "series", None):
        return None
    series = tiff.series[0]
    pages = getattr(series, "pages", None)
    if pages is not None and len(pages) > 0:
        return pages
    return None


def read_tiff_frame(tiff_path: Path, frame_index: int) -> np.ndarray:
    """Read one frame from a TIFF stack, preferring tifffile when available."""
    frame_index = int(frame_index)
    if frame_index < 0:
        raise IndexError(f"Frame index must be non-negative, got {frame_index}.")

    if tifffile is not None:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            pages = _series_pages(tif)
            if pages is not None:
                if frame_index >= len(pages):
                    raise IndexError(f"Frame {frame_index} is outside 0..{len(pages) - 1}.")
                return np.asarray(pages[frame_index].asarray())
            data = tif.asarray()
            if data.ndim <= 2:
                if frame_index != 0:
                    raise IndexError("Single-frame TIFF has only frame 0.")
                return np.asarray(data)
            if frame_index >= data.shape[0]:
                raise IndexError(f"Frame {frame_index} is outside 0..{data.shape[0] - 1}.")
            return np.asarray(data[frame_index])

    with Image.open(tiff_path) as image:
        image.seek(frame_index)
        return np.array(image, copy=True)


def iter_tiff_frames(tiff_path: Path) -> Iterator[np.ndarray]:
    """Yield TIFF frames while keeping the file open for efficient sequential access."""
    if tifffile is not None:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            pages = _series_pages(tif)
            if pages is not None:
                for page in pages:
                    yield np.asarray(page.asarray())
                return
            data = tif.asarray()
            if data.ndim <= 2:
                yield np.asarray(data)
                return
            for frame_index in range(data.shape[0]):
                yield np.asarray(data[frame_index])
        return

    with Image.open(tiff_path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        for frame_number in range(frame_count):
            image.seek(frame_number)
            yield np.array(image, copy=True)


def finite_min_max(frame: np.ndarray) -> tuple[float, float]:
    values = np.asarray(frame)
    if values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def dtype_display_bounds(frame: np.ndarray) -> tuple[float, float]:
    dtype = np.asarray(frame).dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return finite_min_max(frame)


def auto_display_range(
    frame: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[float, float]:
    values = np.asarray(frame)
    if values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, lower_percentile))
    high = float(np.percentile(finite, upper_percentile))
    if high <= low:
        low, high = finite_min_max(frame)
    if high <= low:
        high = low + 1.0
    return low, high


def downsample_for_preview(frame: np.ndarray, max_edge: int) -> tuple[np.ndarray, int]:
    """Return a strided preview frame and the source-pixel step used."""
    if max_edge <= 0 or frame.ndim < 2:
        return np.asarray(frame), 1
    height, width = int(frame.shape[0]), int(frame.shape[1])
    step = max(1, int(math.ceil(max(height, width) / float(max_edge))))
    if step <= 1:
        return np.asarray(frame), 1
    if frame.ndim == 2:
        return np.asarray(frame[::step, ::step]).copy(), step
    return np.asarray(frame[::step, ::step, ...]).copy(), step


def load_preview_frame(tiff_path: Path, frame_index: int, frame_count: int, max_edge: int) -> PreviewFrame:
    start = time.perf_counter()
    frame = read_tiff_frame(tiff_path, frame_index)
    preview, downsample = downsample_for_preview(frame, max_edge)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PreviewFrame(
        tiff_path=Path(tiff_path),
        frame_index=int(frame_index),
        frame_count=int(frame_count),
        source_shape=tuple(int(value) for value in frame.shape),
        source_dtype=str(frame.dtype),
        preview_frame=np.asarray(preview),
        downsample=int(downsample),
        observed_range=finite_min_max(frame),
        auto_range=auto_display_range(frame),
        dtype_range=dtype_display_bounds(frame),
        backend=available_backend_name(),
        elapsed_ms=elapsed_ms,
    )


def persistent_preview_cache_path(
    tiff_path: Path,
    max_edge: int,
    cache_root: Path | None = None,
) -> Path:
    source = Path(tiff_path)
    stat = source.stat()
    root = Path(cache_root) if cache_root is not None else source.parent / PREVIEW_CACHE_FOLDER
    identity = "|".join(
        [
            str(source.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(max_edge),
            str(PREVIEW_CACHE_VERSION),
        ]
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source.stem)
    return root / f"{safe_stem}_{digest}_preview.zarr"


def read_persistent_preview_frame(
    tiff_path: Path,
    frame_index: int,
    frame_count: int,
    max_edge: int,
    cache_root: Path | None = None,
) -> PreviewFrame | None:
    if zarr is None:
        return None
    cache_path = persistent_preview_cache_path(tiff_path, max_edge, cache_root)
    if not cache_path.exists():
        return None

    start = time.perf_counter()
    try:
        group = zarr.open_group(str(cache_path), mode="r")
        attrs = dict(group.attrs)
        if not attrs.get("complete", False):
            return None
        if int(attrs.get("version", -1)) != PREVIEW_CACHE_VERSION:
            return None
        if int(attrs.get("max_edge", -1)) != int(max_edge):
            return None
        if int(attrs.get("frame_count", -1)) != int(frame_count):
            return None
        frames = group["frames"]
        if frame_index < 0 or frame_index >= frames.shape[0]:
            raise IndexError(f"Frame {frame_index} is outside 0..{frames.shape[0] - 1}.")
        preview = np.asarray(frames[frame_index])
        observed = np.asarray(group["observed_range"][frame_index], dtype=np.float64)
        auto = np.asarray(group["auto_range"][frame_index], dtype=np.float64)
        dtype_range = tuple(float(value) for value in attrs["dtype_range"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return PreviewFrame(
            tiff_path=Path(tiff_path),
            frame_index=int(frame_index),
            frame_count=int(frame_count),
            source_shape=tuple(int(value) for value in attrs["source_shape"]),
            source_dtype=str(attrs["source_dtype"]),
            preview_frame=preview,
            downsample=int(attrs["downsample"]),
            observed_range=(float(observed[0]), float(observed[1])),
            auto_range=(float(auto[0]), float(auto[1])),
            dtype_range=(dtype_range[0], dtype_range[1]),
            backend="zarr-preview",
            elapsed_ms=elapsed_ms,
        )
    except (KeyError, OSError, ValueError, TypeError):
        return None


def build_persistent_preview_cache(
    tiff_path: Path,
    frame_count: int,
    max_edge: int,
    cache_root: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    if zarr is None:
        raise RuntimeError("zarr is required to build a persistent preview cache.")

    cache_path = persistent_preview_cache_path(tiff_path, max_edge, cache_root)
    existing = read_persistent_preview_frame(tiff_path, 0, frame_count, max_edge, cache_root)
    if existing is not None:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(cache_path), mode="w")
    group.attrs.update(
        {
            "version": PREVIEW_CACHE_VERSION,
            "complete": False,
            "source_path": str(Path(tiff_path).resolve()),
            "frame_count": int(frame_count),
            "max_edge": int(max_edge),
            "created_at": time.time(),
        }
    )

    frames_array = None
    observed_array = None
    auto_array = None
    source_shape: tuple[int, ...] | None = None
    source_dtype = ""
    dtype_range = (0.0, 1.0)
    downsample = 1
    frames_written = 0

    for frame_number, frame in enumerate(iter_tiff_frames(tiff_path)):
        if frame_number >= frame_count:
            break
        frame = np.asarray(frame)
        preview, frame_downsample = downsample_for_preview(frame, max_edge)
        observed = finite_min_max(frame)
        auto = auto_display_range(frame)

        if frames_array is None:
            source_shape = tuple(int(value) for value in frame.shape)
            source_dtype = str(frame.dtype)
            dtype_range = dtype_display_bounds(frame)
            downsample = int(frame_downsample)
            preview = np.asarray(preview)
            frames_array = group.create_array(
                "frames",
                shape=(int(frame_count), *preview.shape),
                chunks=(1, *preview.shape),
                dtype=preview.dtype,
            )
            observed_array = group.create_array(
                "observed_range",
                shape=(int(frame_count), 2),
                chunks=(min(int(frame_count), 1024), 2),
                dtype=np.float64,
            )
            auto_array = group.create_array(
                "auto_range",
                shape=(int(frame_count), 2),
                chunks=(min(int(frame_count), 1024), 2),
                dtype=np.float64,
            )
            group.attrs.update(
                {
                    "source_shape": list(source_shape),
                    "source_dtype": source_dtype,
                    "preview_shape": list(preview.shape),
                    "preview_dtype": str(preview.dtype),
                    "downsample": downsample,
                    "dtype_range": [float(dtype_range[0]), float(dtype_range[1])],
                }
            )
        elif tuple(int(value) for value in frame.shape) != source_shape:
            raise ValueError("Cannot build preview cache for frames with mixed shapes.")
        elif str(frame.dtype) != source_dtype:
            raise ValueError("Cannot build preview cache for frames with mixed dtypes.")
        elif int(frame_downsample) != downsample:
            raise ValueError("Cannot build preview cache for frames with mixed downsample factors.")

        frames_array[frame_number] = preview
        observed_array[frame_number] = observed
        auto_array[frame_number] = auto
        frames_written += 1
        if progress is not None and (
            frame_number == 0
            or (frame_number + 1) % 500 == 0
            or frame_number + 1 == frame_count
        ):
            progress(frame_number + 1, frame_count)

    if frames_written != frame_count:
        raise ValueError(f"Expected {frame_count} TIFF frames, wrote {frames_written} preview frames.")
    group.attrs["complete"] = True
    group.attrs["completed_at"] = time.time()
    return cache_path


class PreviewFrameCache:
    """Thread-safe small LRU cache for decoded preview frames."""

    def __init__(
        self,
        max_frames: int = 6,
        max_edge: int = 1536,
        disk_cache_root: Path | None = None,
    ) -> None:
        self.max_frames = max(1, int(max_frames))
        self.max_edge = max(1, int(max_edge))
        self.disk_cache_root = disk_cache_root
        self._frames: OrderedDict[tuple[str, int], PreviewFrame] = OrderedDict()
        self._lock = RLock()

    def _key(self, tiff_path: Path, frame_index: int) -> tuple[str, int]:
        return (str(Path(tiff_path).resolve()), int(frame_index))

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def get_cached(self, tiff_path: Path, frame_index: int) -> PreviewFrame | None:
        key = self._key(tiff_path, frame_index)
        with self._lock:
            cached = self._frames.get(key)
            if cached is not None:
                self._frames.move_to_end(key)
            return cached

    def get(self, tiff_path: Path, frame_index: int, frame_count: int) -> PreviewFrame:
        key = self._key(tiff_path, frame_index)
        cached = self.get_cached(tiff_path, frame_index)
        if cached is not None:
            return cached

        loaded = read_persistent_preview_frame(
            Path(tiff_path),
            int(frame_index),
            int(frame_count),
            self.max_edge,
            self.disk_cache_root,
        )
        if loaded is None:
            loaded = load_preview_frame(Path(tiff_path), int(frame_index), int(frame_count), self.max_edge)

        with self._lock:
            self._frames[key] = loaded
            self._frames.move_to_end(key)
            while len(self._frames) > self.max_frames:
                self._frames.popitem(last=False)
        return loaded
