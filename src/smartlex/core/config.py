# core/config.py
import json
from pathlib import Path

CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG = {
    "NUM_PROCESSES": 8,
    "TOP_KEYWORDS": 150,
    "AUTOCOMPLETE_WORDS": 100,
    "SUPPORTED_FORMATS": [
        ".pdf",
        ".docx",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".mp4",
        ".mp3",
        ".wav",
    ],
    "INDEX_FOLDER": "all",
    "OUTPUT_FILE": "output.json",
    "AUTOCOMPLETE_FILE": "autocomplete_words.json",
    "DEFERRED_INDEX_FILE": "deferred_index.json",
    "DEFERRED_INDEXING_MAX_CPU": 75.0,
    # --- Image processing & visual-similarity indexing ---
    # Persisted store for image visual features (pHash + ORB descriptors).
    # Reuses the existing JSON index mechanism (index_manager), no new DB.
    "IMAGE_INDEX_FILE": "image_index.json",
    # Image formats accepted by the visual indexing pipeline.
    "IMAGE_SUPPORTED_FORMATS": [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    ],
    # Largest edge (px) an image is downscaled to before feature extraction.
    # Keeps memory bounded and makes ORB/pHash cost predictable.
    "IMAGE_MAX_DIMENSION": 1024,
    # Perceptual hash geometry. HASH_SIZE=8 -> 64-bit hash; the DCT is
    # computed on a HASH_SIZE * HIGHFREQ_FACTOR square then its low
    # frequencies are kept.
    "PHASH_HASH_SIZE": 8,
    "PHASH_HIGHFREQ_FACTOR": 4,
    # Candidate retrieval (coarse pHash stage).
    "IMAGE_CANDIDATE_TOP_K": 10,
    # Candidates whose pHash Hamming distance exceeds this are discarded
    # before the (expensive) ORB stage. 64 == the full 64-bit hash width,
    # i.e. "no cut-off" by default.
    "IMAGE_PHASH_MAX_DISTANCE": 64,
    # ORB detector/descriptor configuration.
    "ORB_N_FEATURES": 500,
    "ORB_SCALE_FACTOR": 1.2,
    "ORB_N_LEVELS": 8,
    "ORB_FAST_THRESHOLD": 20,
    # Lowe ratio-test threshold used when matching ORB descriptors.
    "ORB_MATCH_RATIO": 0.75,
    # Final score = ALPHA * phash_similarity + BETA * orb_similarity.
    # ALPHA + BETA should sum to 1.0 (normalised defensively at runtime).
    "IMAGE_SCORE_ALPHA": 0.4,
    "IMAGE_SCORE_BETA": 0.6,
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            user_config = json.load(f)
        return {**DEFAULT_CONFIG, **user_config}
    return DEFAULT_CONFIG
