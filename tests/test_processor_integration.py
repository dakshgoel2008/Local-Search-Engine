"""Integration test for the image hook inside the core batch processor.

The processor module depends on the text NLP stack (``rake_nltk``/``nltk``);
where that is not installed this test is skipped, since it exercises the
integration seam rather than the image feature logic itself (covered
elsewhere).
"""

import pytest

# The core processor pulls in the full text stack (rake_nltk/nltk, PyMuPDF's
# ``fitz``, python-docx).  Skip cleanly if any of it is unavailable rather than
# failing collection -- this test exercises the integration seam, not the image
# feature logic itself (covered by the other test modules).
try:
    from smartlex.core.processor import process_batch
except Exception as exc:  # ImportError or transitive dependency failure
    pytest.skip(f"text stack unavailable: {exc}", allow_module_level=True)


def test_process_batch_extracts_image_features(tmp_path, base_image_path, different_image_path):
    batch_file = tmp_path / "batch_1.txt"
    batch_file.write_text(f"{base_image_path}\n{different_image_path}\n")

    keywords, deferred, image_features = process_batch(str(batch_file))

    # Both images produce persistable visual-feature records...
    assert len(image_features) == 2
    for record in image_features.values():
        assert record["phash"]
        assert record["num_bits"] == 64
    # ...and are still queued for deferred OCR (existing behaviour preserved).
    assert set(deferred) == {base_image_path, different_image_path}


def test_process_batch_missing_file_is_reported_gracefully(tmp_path):
    batch_file = tmp_path / "batch_2.txt"
    batch_file.write_text("/no/such/image.png\n")
    keywords, deferred, image_features = process_batch(str(batch_file))
    assert image_features == {}
