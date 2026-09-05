"""Tests for image detection, safe loading and preprocessing."""

import numpy as np
import pytest

from smartlex.image.preprocessing import (
    ImageLoadError,
    ImagePreprocessor,
    UnsupportedImageError,
)


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.jpg", True),
        ("a.jpeg", True),
        ("a.PNG", True),
        ("a.webp", True),
        ("a.bmp", True),
        ("a.tif", True),
        ("a.tiff", True),
        ("a.pdf", False),
        ("a.docx", False),
        ("a.txt", False),
        ("a", False),
    ],
)
def test_is_supported(preprocessor, name, expected):
    assert preprocessor.is_supported(name) is expected


def test_load_grayscale_returns_2d_uint8(base_image_path, preprocessor):
    gray = preprocessor.load_grayscale(base_image_path)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_large_image_is_downscaled(image_factory):
    big = np.random.RandomState(0).randint(0, 256, (3000, 2000), dtype=np.uint8)
    path = image_factory("big", big)
    pre = ImagePreprocessor(max_dimension=512)
    gray = pre.load_grayscale(path)
    assert max(gray.shape) == 512


def test_small_image_not_upscaled(image_factory):
    small = np.random.RandomState(0).randint(0, 256, (100, 80), dtype=np.uint8)
    path = image_factory("small", small)
    pre = ImagePreprocessor(max_dimension=1024)
    gray = pre.load_grayscale(path)
    assert gray.shape == (100, 80)


def test_unsupported_format_raises(tmp_path, preprocessor):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(UnsupportedImageError):
        preprocessor.load_grayscale(str(p))


def test_missing_file_raises(preprocessor):
    with pytest.raises(ImageLoadError):
        preprocessor.load_grayscale("/no/such/image.png")


def test_empty_file_raises(tmp_path, preprocessor):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        preprocessor.load_grayscale(str(p))


def test_corrupted_file_raises(tmp_path, preprocessor):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n not a real image at all")
    with pytest.raises(ImageLoadError):
        preprocessor.load_grayscale(str(p))


def test_preprocessing_is_deterministic(base_image_path, preprocessor):
    a = preprocessor.load_grayscale(base_image_path)
    b = preprocessor.load_grayscale(base_image_path)
    assert np.array_equal(a, b)
