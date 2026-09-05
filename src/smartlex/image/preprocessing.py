"""Image detection, safe loading and deterministic preprocessing.

This is the first stage of the image indexing/retrieval pipeline:

    Image -> Preprocessing -> pHash -> Candidate Retrieval -> ORB -> Ranking

The :class:`ImagePreprocessor` is responsible for validating that a file is a
supported image, loading it safely (guarding against corrupted/oversized
files) and turning it into a deterministic grayscale ``numpy`` array that both
the pHash and ORB extractors consume.  Determinism matters: the exact same
bytes must always yield the exact same features so that a re-indexed image and
a query image compare identically.
"""

import os

import cv2
import numpy as np

from smartlex.core.config import load_config
from smartlex.core.logger import setup_logger

logger = setup_logger("image_preprocessing")


class UnsupportedImageError(ValueError):
    """Raised when a path is not a supported image format."""


class ImageLoadError(ValueError):
    """Raised when an image exists but cannot be decoded (corrupted/invalid)."""


class ImagePreprocessor:
    """Validate, load and normalise images for visual feature extraction.

    Parameters
    ----------
    supported_formats:
        Iterable of lower-case extensions (with leading dot) that are treated
        as images.  Defaults to ``IMAGE_SUPPORTED_FORMATS`` from the config.
    max_dimension:
        Longest edge (in pixels) an image is downscaled to before feature
        extraction.  Prevents very large images from being loaded whole into
        memory.  Defaults to ``IMAGE_MAX_DIMENSION`` from the config.
    """

    def __init__(self, supported_formats=None, max_dimension=None):
        cfg = load_config()
        if supported_formats is None:
            supported_formats = cfg["IMAGE_SUPPORTED_FORMATS"]
        if max_dimension is None:
            max_dimension = cfg["IMAGE_MAX_DIMENSION"]

        self.supported_formats = tuple(f.lower() for f in supported_formats)
        self.max_dimension = int(max_dimension)

    def is_supported(self, file_path):
        """Return ``True`` if *file_path* has a supported image extension."""
        return str(file_path).lower().endswith(self.supported_formats)

    def load_grayscale(self, file_path):
        """Load *file_path* as a deterministic, size-bounded grayscale image.

        Returns a 2-D ``uint8`` ``numpy`` array.

        Raises
        ------
        UnsupportedImageError
            If the extension is not a supported image format.
        ImageLoadError
            If the file is missing, empty or cannot be decoded.
        """
        if not self.is_supported(file_path):
            raise UnsupportedImageError(f"Unsupported image format: {file_path}")

        if not os.path.exists(file_path):
            raise ImageLoadError(f"Image file not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ImageLoadError(f"Image file is empty: {file_path}")

        # Decode via numpy buffer so unicode paths work on every platform.
        try:
            raw = np.fromfile(file_path, dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        except Exception as exc:  # pragma: no cover - defensive
            raise ImageLoadError(f"Failed to decode image {file_path}: {exc}") from exc

        if image is None or image.size == 0:
            raise ImageLoadError(f"Corrupted or undecodable image: {file_path}")

        return self._downscale(image)

    def _downscale(self, image):
        """Downscale so the longest edge is at most ``max_dimension`` px.

        Uses ``INTER_AREA`` (deterministic) and only ever shrinks, never
        enlarges, so small images are left untouched.
        """
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.max_dimension:
            return image

        scale = self.max_dimension / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
