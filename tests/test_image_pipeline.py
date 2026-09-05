"""End-to-end tests for the image indexing + search pipeline.

Covers: indexing -> preprocessing -> pHash -> ORB -> persistence -> query ->
candidate retrieval -> ORB matching -> final ranking, plus the required edge
cases (corrupted/unsupported/missing files, empty descriptors, duplicates,
large images, insufficient candidates).
"""

import numpy as np
import pytest

from smartlex.image.feature_store import ImageFeatureStore
from smartlex.image.indexer import build_image_record, index_image
from smartlex.image.search import ImageSearchService, search_images


def _build_store(paths, index_file):
    store = ImageFeatureStore(index_file=str(index_file))
    for path in paths:
        record = index_image(path)
        assert record is not None
        store.add(record)
    store.save()
    return store


def test_index_image_record_shape(base_image_path):
    record = index_image(base_image_path)
    assert record is not None
    assert record["phash"]
    assert record["num_bits"] == 64
    assert record["orb"] is not None
    assert record["num_keypoints"] > 0
    assert record["width"] > 0 and record["height"] > 0


def test_index_unsupported_returns_none(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("not an image")
    assert index_image(str(p)) is None


def test_index_corrupted_returns_none(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"\x89PNG not really")
    assert index_image(str(p)) is None


def test_index_missing_returns_none():
    assert index_image("/does/not/exist.jpg") is None


def test_store_persistence_roundtrip(base_image_path, different_image_path, tmp_path):
    index_file = tmp_path / "image_index.json"
    _build_store([base_image_path, different_image_path], index_file)

    reloaded = ImageFeatureStore.load(str(index_file))
    assert len(reloaded) == 2
    for record in reloaded.all().values():
        assert record["phash"]
        # ORB descriptors survive the JSON round-trip.
        assert record["orb"] is not None


def test_full_pipeline_ranks_similar_highest(
    base_image_path, resized_image_path, different_image_path, tmp_path
):
    index_file = tmp_path / "image_index.json"
    store = _build_store(
        [base_image_path, resized_image_path, different_image_path], index_file
    )

    service = ImageSearchService(store=store)
    results = service.search(base_image_path, top_k=5)

    assert results
    # The exact same image indexed should rank first with the top score.
    assert results[0]["path"] == build_image_record(base_image_path)["path"]
    assert results[0]["rank"] == 1
    # Scores are sorted descending.
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # The unrelated image should not outrank the resized (similar) one.
    by_path = {r["path"]: r for r in results}
    diff_rec = build_image_record(different_image_path)
    resized_rec = build_image_record(resized_image_path)
    assert by_path[resized_rec["path"]]["final_score"] >= by_path[diff_rec["path"]]["final_score"]


def test_result_contains_required_fields(base_image_path, tmp_path):
    index_file = tmp_path / "image_index.json"
    store = _build_store([base_image_path], index_file)
    results = ImageSearchService(store=store).search(base_image_path)
    assert results
    r = results[0]
    for field in (
        "path",
        "phash_distance",
        "phash_similarity",
        "orb_similarity",
        "final_score",
        "rank",
    ):
        assert field in r


def test_search_images_convenience_function(base_image_path, different_image_path, tmp_path):
    index_file = tmp_path / "image_index.json"
    store = _build_store([base_image_path, different_image_path], index_file)
    results = search_images(base_image_path, image_index=store, top_k=1)
    assert len(results) == 1
    assert results[0]["rank"] == 1


def test_top_k_limits_candidates(tmp_path, image_factory, make_image):
    paths = [image_factory(f"img{i}", make_image(seed=i)) for i in range(6)]
    index_file = tmp_path / "image_index.json"
    store = _build_store(paths, index_file)

    results = ImageSearchService(store=store).search(paths[0], top_k=3)
    assert len(results) <= 3


def test_duplicate_images_both_returned(base_image_path, image_factory, tmp_path, make_image):
    duplicate = image_factory("dup", make_image(seed=1))  # same content as base
    index_file = tmp_path / "image_index.json"
    store = _build_store([base_image_path, duplicate], index_file)

    results = ImageSearchService(store=store).search(base_image_path, top_k=5)
    # Both the original and its duplicate should score at/near the top.
    top_two = results[:2]
    assert all(r["phash_distance"] == 0 for r in top_two)


def test_query_against_empty_index_returns_empty(base_image_path, tmp_path):
    index_file = tmp_path / "image_index.json"
    store = ImageFeatureStore(index_file=str(index_file))
    results = ImageSearchService(store=store).search(base_image_path)
    assert results == []


def test_insufficient_candidates(base_image_path, different_image_path, tmp_path):
    """Asking for more candidates than exist returns only what is available."""
    index_file = tmp_path / "image_index.json"
    store = _build_store([base_image_path, different_image_path], index_file)
    results = ImageSearchService(store=store).search(base_image_path, top_k=50)
    assert 0 < len(results) <= 2


def test_large_image_indexes_without_error(image_factory, tmp_path):
    big = np.random.RandomState(3).randint(0, 256, (2600, 2600, 3), dtype=np.uint8)
    path = image_factory("huge", big)
    record = index_image(path)
    assert record is not None
    # Downscaled: longest edge bounded by IMAGE_MAX_DIMENSION (default 1024).
    assert max(record["width"], record["height"]) <= 1024


def test_empty_descriptor_candidate_does_not_crash(blank_image_path, base_image_path, tmp_path):
    index_file = tmp_path / "image_index.json"
    store = ImageFeatureStore(index_file=str(index_file))
    for path in (blank_image_path, base_image_path):
        record = index_image(path)
        assert record is not None
        store.add(record)
    store.save()

    # Querying with a blank (no-keypoint) image must not crash.
    results = ImageSearchService(store=store).search(blank_image_path, top_k=5)
    assert isinstance(results, list)


def test_query_unsupported_image_raises(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("nope")
    store = ImageFeatureStore(index_file=str(tmp_path / "idx.json"))
    with pytest.raises(Exception):
        ImageSearchService(store=store).search(str(p))
