"""Image indexing orchestration: preprocessing -> pHash -> ORB -> record.

``index_image`` runs the *indexing half* of the pipeline for a single image and
returns a persistable record (see
:class:`~smartlex.image.feature_store.ImageFeatureStore`).  Descriptors are
computed exactly once here, at index time, so query-time search never has to
recompute features for already-indexed images.

Corrupted, unsupported or unreadable images are handled gracefully: a warning
is logged and ``None`` is returned instead of raising, so a bad file never
aborts a whole batch.
"""

import os

from smartlex.core.logger import setup_logger
from smartlex.image.orb_features import OrbFeatureExtractor, serialize_descriptors
from smartlex.image.phash import PhashExtractor
from smartlex.image.preprocessing import (
    ImageLoadError,
    ImagePreprocessor,
    UnsupportedImageError,
)

logger = setup_logger("image_indexer")


def build_image_record(
    file_path,
    preprocessor=None,
    phash_extractor=None,
    orb_extractor=None,
):
    """Extract visual features for *file_path* and return a record dict.

    Raises the underlying preprocessing errors
    (:class:`UnsupportedImageError` / :class:`ImageLoadError`) so callers that
    want strict behaviour (e.g. the query path) can react to them.  For batch
    indexing use :func:`index_image`, which swallows these into ``None``.
    """
    preprocessor = preprocessor or ImagePreprocessor()
    phash_extractor = phash_extractor or PhashExtractor()
    orb_extractor = orb_extractor or OrbFeatureExtractor()

    gray = preprocessor.load_grayscale(file_path)
    height, width = gray.shape[:2]

    phash = phash_extractor.compute(gray)
    keypoints, descriptors = orb_extractor.extract(gray)

    try:
        size_bytes = os.path.getsize(file_path)
    except OSError:
        size_bytes = None

    return {
        "path": os.path.abspath(file_path),
        "phash": phash,
        "num_bits": phash_extractor.num_bits,
        "orb": serialize_descriptors(descriptors),
        "num_keypoints": 0 if keypoints is None else len(keypoints),
        "width": int(width),
        "height": int(height),
        "size_bytes": size_bytes,
    }


def index_image(
    file_path,
    preprocessor=None,
    phash_extractor=None,
    orb_extractor=None,
):
    """Index a single image, returning its record or ``None`` on failure.

    This is the batch-safe entry point: unsupported/corrupted/missing images
    log a warning and return ``None`` rather than raising.
    """
    try:
        return build_image_record(
            file_path,
            preprocessor=preprocessor,
            phash_extractor=phash_extractor,
            orb_extractor=orb_extractor,
        )
    except UnsupportedImageError as exc:
        logger.warning(f"Skipping unsupported image {file_path}: {exc}")
    except ImageLoadError as exc:
        logger.warning(f"Skipping unreadable image {file_path}: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Unexpected error indexing image {file_path}: {exc}")
    return None
