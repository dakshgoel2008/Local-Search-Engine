"""Tests for perceptual hashing (pHash) and Hamming-distance similarity."""

import cv2
import numpy as np
import pytest

from smartlex.image.phash import PhashExtractor, hamming_distance, phash_similarity
from smartlex.image.preprocessing import ImagePreprocessor


@pytest.fixture
def phash():
    return PhashExtractor()


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


def _phash_of(path, preprocessor, phash):
    return phash.compute(preprocessor.load_grayscale(path))


def test_same_image_distance_zero(base_image_path, preprocessor, phash):
    h1 = _phash_of(base_image_path, preprocessor, phash)
    h2 = _phash_of(base_image_path, preprocessor, phash)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0


def test_resized_image_low_distance(base_image_path, resized_image_path, preprocessor, phash):
    h_base = _phash_of(base_image_path, preprocessor, phash)
    h_resized = _phash_of(resized_image_path, preprocessor, phash)
    distance = hamming_distance(h_base, h_resized)
    # A mild resize should barely perturb the perceptual hash.
    assert distance <= phash.num_bits * 0.15


def test_modified_image_relatively_low_distance(
    base_image_path, modified_image_path, preprocessor, phash
):
    h_base = _phash_of(base_image_path, preprocessor, phash)
    h_mod = _phash_of(modified_image_path, preprocessor, phash)
    distance = hamming_distance(h_base, h_mod)
    # A small local edit should stay well below "completely different".
    assert distance <= phash.num_bits * 0.30


def test_different_image_large_distance(
    base_image_path, different_image_path, modified_image_path, preprocessor, phash
):
    h_base = _phash_of(base_image_path, preprocessor, phash)
    h_diff = _phash_of(different_image_path, preprocessor, phash)
    h_mod = _phash_of(modified_image_path, preprocessor, phash)

    dist_diff = hamming_distance(h_base, h_diff)
    dist_mod = hamming_distance(h_base, h_mod)

    # An unrelated image must be clearly farther than a small local edit.
    assert dist_diff > dist_mod
    assert dist_diff >= phash.num_bits * 0.20


def test_phash_is_deterministic(base_image_path, preprocessor, phash):
    hashes = {_phash_of(base_image_path, preprocessor, phash) for _ in range(5)}
    assert len(hashes) == 1


def test_similarity_monotonic_with_distance():
    num_bits = 64
    assert phash_similarity("0" * 16, "0" * 16, num_bits) == 1.0
    assert phash_similarity("f" * 16, "0" * 16, num_bits) == 0.0
    mid = phash_similarity("f" * 8 + "0" * 8, "0" * 16, num_bits)
    assert 0.0 < mid < 1.0


def test_hash_width_matches_num_bits(base_image_path, preprocessor, phash):
    h = _phash_of(base_image_path, preprocessor, phash)
    # hex string length == num_bits / 4.
    assert len(h) == phash.num_bits // 4


def test_hamming_distance_missing_hash_raises():
    with pytest.raises(ValueError):
        hamming_distance(None, "0000")


def test_not_a_cryptographic_hash(preprocessor, phash, tmp_path):
    """One-pixel change must barely move the hash (unlike SHA/MD5)."""
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (200, 200), 255, -1)
    p1 = tmp_path / "a.png"
    cv2.imwrite(str(p1), img)

    img2 = img.copy()
    img2[0, 0] = 255 - img2[0, 0]
    p2 = tmp_path / "b.png"
    cv2.imwrite(str(p2), img2)

    h1 = _phash_of(str(p1), preprocessor, phash)
    h2 = _phash_of(str(p2), preprocessor, phash)
    assert hamming_distance(h1, h2) <= 2
