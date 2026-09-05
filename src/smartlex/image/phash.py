"""Perceptual hashing (pHash) for coarse visual similarity.

pHash captures the *overall* visual appearance of an image, so two images that
look alike (even after resizing, mild compression or small edits) produce
hashes that are close in Hamming distance.  This is deliberately **not** a
cryptographic hash (SHA-256/MD5): those change completely with a single
altered pixel and are useless for similarity.

Algorithm (the classic DCT-based pHash):

1. Resize the grayscale image to ``hash_size * highfreq_factor`` square.
2. Take the 2-D Discrete Cosine Transform (DCT).
3. Keep the top-left ``hash_size x hash_size`` block (the low frequencies that
   describe the broad structure of the image).
4. Compare each coefficient against the median of that block (excluding the DC
   term at ``[0, 0]``, which only encodes overall brightness) to produce a bit.

The result is a ``hash_size**2``-bit hash, stored as a hex string so it fits
naturally into the existing JSON index.
"""

import cv2
import numpy as np

from smartlex.core.config import load_config


class PhashExtractor:
    """Compute and compare DCT-based perceptual hashes.

    Parameters
    ----------
    hash_size:
        Side length of the low-frequency block kept from the DCT.  The hash is
        ``hash_size**2`` bits (default 8 -> 64-bit hash).
    highfreq_factor:
        The image is resized to ``hash_size * highfreq_factor`` before the DCT
        so that meaningful low-frequency content is retained.
    """

    def __init__(self, hash_size=None, highfreq_factor=None):
        cfg = load_config()
        self.hash_size = int(hash_size if hash_size is not None else cfg["PHASH_HASH_SIZE"])
        self.highfreq_factor = int(
            highfreq_factor
            if highfreq_factor is not None
            else cfg["PHASH_HIGHFREQ_FACTOR"]
        )

    @property
    def num_bits(self):
        """Total number of bits in a hash produced by this extractor."""
        return self.hash_size * self.hash_size

    def compute(self, gray_image):
        """Return the perceptual hash of *gray_image* as a hex string.

        *gray_image* must be a 2-D ``numpy`` array (as produced by
        :class:`~smartlex.image.preprocessing.ImagePreprocessor`).
        """
        img_size = self.hash_size * self.highfreq_factor
        resized = cv2.resize(
            gray_image, (img_size, img_size), interpolation=cv2.INTER_AREA
        )

        dct = cv2.dct(np.float32(resized))
        low_freq = dct[: self.hash_size, : self.hash_size]

        # Exclude the DC term (overall brightness) from the median so the hash
        # reflects structure rather than exposure.
        coeffs = low_freq.flatten()
        median = np.median(coeffs[1:])

        bits = (low_freq > median).flatten()
        return self._bits_to_hex(bits)

    def _bits_to_hex(self, bits):
        value = 0
        for bit in bits:
            value = (value << 1) | int(bool(bit))
        width = (self.num_bits + 3) // 4
        return format(value, "0{}x".format(width))


def hamming_distance(phash_a, phash_b):
    """Hamming distance between two hex-string perceptual hashes.

    A smaller distance means the images are more visually similar.  ``0`` means
    the hashes are identical.
    """
    if phash_a is None or phash_b is None:
        raise ValueError("Cannot compare a missing pHash")
    return bin(int(phash_a, 16) ^ int(phash_b, 16)).count("1")


def similarity_from_distance(distance, num_bits):
    """Normalise an already-computed Hamming *distance* into a ``[0, 1]`` score.

    ``similarity = 1 - distance / num_bits`` so distance ``0`` -> ``1.0`` and a
    fully-opposite hash -> ``0.0``.  Kept separate from :func:`phash_similarity`
    so callers that already know the distance (e.g. after candidate retrieval)
    do not recompute it.
    """
    if num_bits <= 0:
        return 0.0
    return max(0.0, 1.0 - distance / float(num_bits))


def phash_similarity(phash_a, phash_b, num_bits):
    """Normalise the Hamming distance between two hashes into a ``[0, 1]`` score.

    Identical images score ``1.0`` and completely opposite hashes score ``0.0``.
    """
    return similarity_from_distance(hamming_distance(phash_a, phash_b), num_bits)
