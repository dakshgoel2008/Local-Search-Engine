import concurrent.futures
import os
from collections import Counter
from pathlib import Path

from smartlex.core.config import load_config
from smartlex.text.keyword_extraction import rake_keywords
from smartlex.core.logger import setup_logger
from smartlex.text.text_extraction import extract_text_docx, extract_text_pdf

logger = setup_logger(__name__)

# Loaded once per (sub)process so worker processes pick up user config too.
_CFG = load_config()
_IMAGE_FORMATS = tuple(f.lower() for f in _CFG["IMAGE_SUPPORTED_FORMATS"])


def _is_supported_image(path):
    return path.lower().endswith(_IMAGE_FORMATS)


def extract_keywords_from_file(file_path):
    lower_path = file_path.lower()
    if lower_path.endswith(".pdf"):
        text = extract_text_pdf(file_path)
        return rake_keywords(text), False
    elif lower_path.endswith(".docx"):
        text = extract_text_docx(file_path)
        return rake_keywords(text), False
    elif _is_supported_image(file_path):
        from smartlex.image.image_extraction import fast_metadata_extract

        keywords = fast_metadata_extract(file_path)
        return keywords, True  # True means deferred deep (OCR) processing needed
    else:
        return [], False


def process_batch(batch_file):
    """Process one batch file, returning keywords, deferred files and image features.

    The keyword index (text search) is built exactly as before.  In addition,
    every supported image also has its *visual* features (pHash + ORB
    descriptors) extracted here, once, so they are persisted at index time and
    never recomputed during search.
    """
    result = {}
    deferred_files = []
    image_features = {}
    if not os.path.exists(batch_file):
        logger.error(f"Batch file not found: {batch_file}")
        return result, deferred_files, image_features

    with open(batch_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f]

    # Visual-feature extractors are created lazily, once per batch (only if the
    # batch actually contains images), and reused across images to avoid
    # re-instantiating cv2.ORB for every file.
    image_extractors = None

    for path in paths:
        if os.path.exists(path):
            try:
                kws, needs_defer = extract_keywords_from_file(path)
                if kws:
                    result[path] = kws
                if needs_defer:
                    deferred_files.append(path)
                if _is_supported_image(path):
                    if image_extractors is None:
                        image_extractors = _build_image_extractors()
                    from smartlex.image.indexer import index_image

                    record = index_image(path, *image_extractors)
                    if record is not None:
                        image_features[record["path"]] = record
            except Exception as e:
                logger.error(f"Error processing file {path}: {e}")
    return result, deferred_files, image_features


def _build_image_extractors():
    """Instantiate the (preprocessor, phash, orb) extractor trio once per batch."""
    from smartlex.image.orb_features import OrbFeatureExtractor
    from smartlex.image.phash import PhashExtractor
    from smartlex.image.preprocessing import ImagePreprocessor

    return ImagePreprocessor(), PhashExtractor(), OrbFeatureExtractor()


def refine_keywords(data, top_n):
    for k, v in data.items():
        c = Counter(v)
        top_values = [w for w, _ in c.most_common(top_n)]
        data[k] = list(set(top_values))
    return data


def process_all_batches(batch_files, top_n):
    D = {}
    all_deferred = []
    all_image_features = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_batch, batch_files))
    for res, deferred, image_features in results:
        D.update(res)
        all_deferred.extend(deferred)
        all_image_features.update(image_features)
    return refine_keywords(D, top_n), all_deferred, all_image_features
