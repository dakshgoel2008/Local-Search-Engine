"""ORB feature extraction, (de)serialization and matching.

ORB (Oriented FAST and Rotated BRIEF) provides *local* feature verification:
where pHash asks "do these look alike overall?", ORB asks "do these share the
same distinctive corners/patches?".  ORB is robust to rotation, moderate
scaling and minor illumination changes, which is why it is used to verify the
coarse pHash candidates.

Descriptors are stored once at index time (they are expensive to compute) and
persisted as base64 so they round-trip losslessly through the JSON index.
"""

import base64

import cv2
import numpy as np

from smartlex.core.config import load_config
from smartlex.core.logger import setup_logger

logger = setup_logger("image_orb")


class OrbFeatureExtractor:
    """Detect ORB keypoints and compute descriptors for an image.

    All tunables come from configuration (never hard-coded at call sites):
    ``ORB_N_FEATURES``, ``ORB_SCALE_FACTOR``, ``ORB_N_LEVELS`` and
    ``ORB_FAST_THRESHOLD``.
    """

    def __init__(
        self,
        n_features=None,
        scale_factor=None,
        n_levels=None,
        fast_threshold=None,
    ):
        cfg = load_config()
        self.n_features = int(
            n_features if n_features is not None else cfg["ORB_N_FEATURES"]
        )
        self.scale_factor = float(
            scale_factor if scale_factor is not None else cfg["ORB_SCALE_FACTOR"]
        )
        self.n_levels = int(n_levels if n_levels is not None else cfg["ORB_N_LEVELS"])
        self.fast_threshold = int(
            fast_threshold if fast_threshold is not None else cfg["ORB_FAST_THRESHOLD"]
        )

        self._orb = cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            fastThreshold=self.fast_threshold,
        )

    def extract(self, gray_image):
        """Return ``(keypoints, descriptors)`` for *gray_image*.

        ``descriptors`` is a ``(N, 32) uint8`` array, or ``None`` when the
        detector found no keypoints (e.g. a blank image).  Callers must handle
        the ``None`` case.
        """
        keypoints, descriptors = self._orb.detectAndCompute(gray_image, None)
        return keypoints, descriptors


def serialize_descriptors(descriptors):
    """Serialize an ORB descriptor array to a JSON-safe dict.

    Returns ``None`` when there are no descriptors so the caller can persist a
    plain ``null``.  The representation preserves dtype and shape so it can be
    loaded back without any loss required for matching.
    """
    if descriptors is None or getattr(descriptors, "size", 0) == 0:
        return None
    descriptors = np.ascontiguousarray(descriptors, dtype=np.uint8)
    return {
        "data": base64.b64encode(descriptors.tobytes()).decode("ascii"),
        "shape": list(descriptors.shape),
        "dtype": "uint8",
    }


def deserialize_descriptors(payload):
    """Inverse of :func:`serialize_descriptors`.

    Returns an ``(N, 32) uint8`` array, or ``None`` if *payload* is ``None`` or
    represents an empty descriptor set.
    """
    if not payload:
        return None
    try:
        raw = base64.b64decode(payload["data"])
        array = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
        return array.reshape(tuple(payload["shape"]))
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to deserialize ORB descriptors: {exc}")
        return None


def match_descriptors(query_desc, candidate_desc, ratio=None):
    """Match two ORB descriptor sets using a BF matcher + Lowe's ratio test.

    Returns the list of "good" matches surviving the ratio test.  Gracefully
    returns an empty list when either side has no/too-few descriptors, so it
    never crashes on empty or incompatible inputs.
    """
    if ratio is None:
        ratio = load_config()["ORB_MATCH_RATIO"]

    if query_desc is None or candidate_desc is None:
        return []
    if len(query_desc) < 2 or len(candidate_desc) < 2:
        # knnMatch with k=2 needs at least two candidates to compare against.
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(query_desc, candidate_desc, k=2)
    except cv2.error as exc:  # pragma: no cover - defensive
        logger.error(f"ORB matching failed: {exc}")
        return []

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    return good


def orb_similarity(query_desc, candidate_desc, ratio=None):
    """Compute a normalised ORB similarity score in ``[0, 1]``.

    The score is the fraction of the *larger* descriptor set that found a good
    match::

        similarity = good_matches / max(len(query_desc), len(candidate_desc))

    Why ``max`` and not ``min``:  ``knnMatch(query, candidate)`` yields at most
    one match per *query* descriptor, but with ``crossCheck=False`` several
    query descriptors can map onto the same few candidate descriptors.  So the
    number of good matches can exceed ``min(len_q, len_c)``.  Normalising by the
    smaller set would then saturate the score at ``1.0`` whenever one image is
    much sparser than the other -- e.g. a near-blank or unrelated candidate with
    a handful of descriptors could score ~1.0 against a feature-rich query.
    Normalising by the *larger* set is guaranteed to stay in ``[0, 1]``
    (``good <= len_q <= max``) and does not inflate for asymmetric keypoint
    counts.  Missing/empty descriptors yield ``0.0`` rather than an error.
    """
    if query_desc is None or candidate_desc is None:
        return 0.0
    denom = max(len(query_desc), len(candidate_desc))
    if denom == 0:
        return 0.0
    good = match_descriptors(query_desc, candidate_desc, ratio=ratio)
    # good <= len(query_desc) <= denom, so this is already within [0, 1];
    # min() is a defensive belt-and-braces guard only.
    return min(1.0, len(good) / float(denom))
