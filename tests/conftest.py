"""Shared pytest fixtures for the SmartLex image pipeline tests.

Images are generated synthetically (deterministically, from a seed) with
OpenCV/numpy so the suite needs no binary test assets on disk.  Generated
images are feature-rich (many corners/edges) so ORB has real keypoints to work
with, while remaining stable enough for pHash.
"""

import os
import sys

import cv2
import numpy as np
import pytest

# Make the ``smartlex`` package importable, mirroring run.py / test_patent_logic.py.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def make_textured_image(seed=0, size=512):
    """Create a deterministic, feature-rich BGR image for a given seed."""
    rng = np.random.RandomState(seed)

    # Smooth diagonal gradient background (gives pHash something global).
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    background = (ramp[None, :] + ramp[:, None]) / 2.0
    image = np.stack([background] * 3, axis=-1).astype(np.uint8)

    # Draw many shapes -> lots of corners/edges for ORB keypoints.
    for _ in range(40):
        x1, y1 = rng.randint(0, size, size=2)
        x2, y2 = rng.randint(0, size, size=2)
        color = tuple(int(c) for c in rng.randint(0, 256, size=3))
        shape = rng.randint(0, 3)
        if shape == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=rng.randint(1, 5))
        elif shape == 1:
            radius = int(rng.randint(5, 40))
            cv2.circle(image, (x1, y1), radius, color, thickness=rng.randint(1, 4))
        else:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness=rng.randint(1, 4))

    # A little seeded noise for texture (still deterministic).
    noise = rng.randint(0, 20, size=image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    return image


def write_image(path, image):
    cv2.imwrite(str(path), image)
    return str(path)


@pytest.fixture
def make_image():
    """Expose the deterministic textured-image generator to tests."""
    return make_textured_image


@pytest.fixture
def image_factory(tmp_path):
    """Return a factory that writes a named image and returns its path."""

    def _factory(name, image, ext=".png"):
        path = tmp_path / f"{name}{ext}"
        return write_image(path, image)

    return _factory


@pytest.fixture
def base_image_path(image_factory):
    return image_factory("base", make_textured_image(seed=1))


@pytest.fixture
def resized_image_path(image_factory):
    img = make_textured_image(seed=1)
    resized = cv2.resize(img, (410, 410), interpolation=cv2.INTER_AREA)  # ~0.8x
    return image_factory("resized", resized)


@pytest.fixture
def rotated_image_path(image_factory):
    img = make_textured_image(seed=1)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)  # rotate 12 degrees
    rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return image_factory("rotated", rotated)


@pytest.fixture
def modified_image_path(image_factory):
    """The base image with a small local edit (a filled box + text)."""
    img = make_textured_image(seed=1).copy()
    cv2.rectangle(img, (20, 20), (90, 90), (0, 0, 0), thickness=-1)
    cv2.putText(img, "X", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return image_factory("modified", img)


@pytest.fixture
def different_image_path(image_factory):
    return image_factory("different", make_textured_image(seed=999))


@pytest.fixture
def blank_image_path(image_factory):
    """A uniform image -> effectively no ORB keypoints."""
    blank = np.full((256, 256, 3), 127, dtype=np.uint8)
    return image_factory("blank", blank)
