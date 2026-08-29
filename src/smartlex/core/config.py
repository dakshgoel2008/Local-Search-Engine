# core/config.py
import json
from pathlib import Path

CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG = {
    "NUM_PROCESSES": 8,
    "TOP_KEYWORDS": 150,
    "AUTOCOMPLETE_WORDS": 100,
    "SUPPORTED_FORMATS": [".pdf", ".docx", ".png", ".jpg", ".mp4", ".mp3", ".wav"],
    "INDEX_FOLDER": "all",
    "OUTPUT_FILE": "output.json",
    "AUTOCOMPLETE_FILE": "autocomplete_words.json",
    "DEFERRED_INDEX_FILE": "deferred_index.json",
    "DEFERRED_INDEXING_MAX_CPU": 75.0,
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            user_config = json.load(f)
        return {**DEFAULT_CONFIG, **user_config}
    return DEFAULT_CONFIG
