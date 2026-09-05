"""Tests for ORB extraction, (de)serialization and matching."""

import numpy as np
import pytest

from smartlex.image.orb_features import (
    OrbFeatureExtractor,
    deserialize_descriptors,
    match_descriptors,
    orb_similarity,
    serialize_descriptors,
)
from smartlex.image.preprocessing import ImagePreprocessor


@pytest.fixture
def orb():
    return OrbFeatureExtractor()


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


def _descriptors(path, preprocessor, orb):
    gray = preprocessor.load_grayscale(path)
    _, desc = orb.extract(gray)
    return desc


def test_same_image_strong_matching(base_image_path, preprocessor, orb):
    desc = _descriptors(base_image_path, preprocessor, orb)
    score = orb_similarity(desc, desc)
    assert score > 0.9  # identical descriptors -> near-perfect self match


def test_resized_matching_reasonable(base_image_path, resized_image_path, preprocessor, orb):
    d_base = _descriptors(base_image_path, preprocessor, orb)
    d_resized = _descriptors(resized_image_path, preprocessor, orb)
    score = orb_similarity(d_base, d_resized)
    assert score > 0.1
    assert len(match_descriptors(d_base, d_resized)) >= 10


def test_rotated_matching_reasonable(base_image_path, rotated_image_path, preprocessor, orb):
    d_base = _descriptors(base_image_path, preprocessor, orb)
    d_rot = _descriptors(rotated_image_path, preprocessor, orb)
    # ORB is designed to be rotation invariant.
    assert len(match_descriptors(d_base, d_rot)) >= 10


def test_similar_images_meaningful_matches(
    base_image_path, modified_image_path, preprocessor, orb
):
    d_base = _descriptors(base_image_path, preprocessor, orb)
    d_mod = _descriptors(modified_image_path, preprocessor, orb)
    assert len(match_descriptors(d_base, d_mod)) >= 10


def test_unrelated_images_weak_matching(
    base_image_path, different_image_path, modified_image_path, preprocessor, orb
):
    d_base = _descriptors(base_image_path, preprocessor, orb)
    d_diff = _descriptors(different_image_path, preprocessor, orb)
    d_mod = _descriptors(modified_image_path, preprocessor, orb)

    score_diff = orb_similarity(d_base, d_diff)
    score_similar = orb_similarity(d_base, d_mod)
    # Unrelated images should match more weakly than genuinely similar ones.
    assert score_diff < score_similar


def test_image_with_no_keypoints_no_crash(blank_image_path, preprocessor, orb):
    desc = _descriptors(blank_image_path, preprocessor, orb)
    # Either None or a tiny descriptor set; matching must not crash and scores 0.
    assert orb_similarity(desc, desc) == 0.0 or desc is None or len(desc) < 2


def test_matching_handles_none_descriptors():
    assert match_descriptors(None, None) == []
    assert orb_similarity(None, None) == 0.0


def test_matching_handles_too_few_descriptors():
    one = np.zeros((1, 32), dtype=np.uint8)
    assert match_descriptors(one, one) == []
    assert orb_similarity(one, one) == 0.0


def test_asymmetric_sizes_do_not_inflate_score():
    """A tiny candidate must not score ~1.0 against a feature-rich query.

    Regression test for the normalization: dividing good-matches by the
    *smaller* set would saturate the score to 1.0 here; dividing by the larger
    set keeps it low and bounded.
    """
    rng = np.random.RandomState(0)
    query = rng.randint(0, 256, (500, 32), dtype=np.uint8)
    # Two genuine descriptors lifted from the query (guaranteed exact matches).
    candidate = query[:2].copy()

    score = orb_similarity(query, candidate)
    assert score <= 1.0
    # With min-normalization this would be ~1.0; with max-normalization it is tiny.
    assert score < 0.2


def test_similarity_never_exceeds_one():
    rng = np.random.RandomState(7)
    a = rng.randint(0, 256, (300, 32), dtype=np.uint8)
    b = rng.randint(0, 256, (12, 32), dtype=np.uint8)
    assert 0.0 <= orb_similarity(a, b) <= 1.0
    assert 0.0 <= orb_similarity(b, a) <= 1.0


def test_serialization_roundtrip(base_image_path, preprocessor, orb):
    desc = _descriptors(base_image_path, preprocessor, orb)
    payload = serialize_descriptors(desc)
    restored = deserialize_descriptors(payload)
    assert restored is not None
    assert restored.dtype == np.uint8
    assert restored.shape == desc.shape
    assert np.array_equal(restored, desc)
    # Restored descriptors match exactly like the originals.
    assert orb_similarity(desc, restored) > 0.9


def test_serialization_of_empty_is_none():
    assert serialize_descriptors(None) is None
    assert serialize_descriptors(np.zeros((0, 32), dtype=np.uint8)) is None
    assert deserialize_descriptors(None) is None


def test_orb_config_is_respected():
    custom = OrbFeatureExtractor(n_features=123, scale_factor=1.5, n_levels=4, fast_threshold=10)
    assert custom.n_features == 123
    assert custom.scale_factor == 1.5
    assert custom.n_levels == 4
    assert custom.fast_threshold == 10
