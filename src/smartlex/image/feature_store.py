"""Persistence layer for image visual features.

Reuses the project's existing JSON index mechanism
(:mod:`smartlex.core.index_manager`) rather than introducing a new database.
The store maps an image path to a record::

    {
        "path": "/abs/path/to/image.jpg",
        "phash": "d1e2...",             # hex perceptual hash
        "num_bits": 64,                  # hash width (for normalisation)
        "orb": {"data": "<b64>", "shape": [N, 32], "dtype": "uint8"} | null,
        "num_keypoints": 342,
        "width": 800,
        "height": 600,
        "size_bytes": 51234
    }

The whole store is a ``{path: record}`` dict, mirroring the shape of the text
index (``output.json``), so it loads/saves through the same helpers.
"""

from smartlex.core.config import load_config
from smartlex.core.index_manager import load_index, save_index
from smartlex.core.logger import setup_logger

logger = setup_logger("image_feature_store")


class ImageFeatureStore:
    """In-memory image feature index backed by a JSON file."""

    def __init__(self, index_file=None, records=None):
        cfg = load_config()
        self.index_file = index_file or cfg["IMAGE_INDEX_FILE"]
        self._records = dict(records) if records else {}

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, index_file=None):
        """Load the store from disk, returning an empty store if none exists."""
        cfg = load_config()
        path = index_file or cfg["IMAGE_INDEX_FILE"]
        try:
            data = load_index(path)
            if not isinstance(data, dict):
                data = {}
        except (FileNotFoundError, ValueError):
            data = {}
        return cls(index_file=path, records=data)

    def save(self):
        """Persist the store to its JSON file."""
        save_index(self._records, self.index_file)

    # -- record access -------------------------------------------------------
    def add(self, record):
        """Insert/replace a record keyed by its ``path``."""
        self._records[record["path"]] = record

    def update(self, records):
        """Merge a ``{path: record}`` mapping into the store."""
        self._records.update(records)

    def get(self, path):
        return self._records.get(path)

    def remove(self, path):
        self._records.pop(path, None)

    def all(self):
        """Return the underlying ``{path: record}`` mapping."""
        return self._records

    def __len__(self):
        return len(self._records)

    def __contains__(self, path):
        return path in self._records
