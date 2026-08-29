import concurrent.futures
import os
from collections import Counter
from pathlib import Path

from smartlex.text.keyword_extraction import rake_keywords
from smartlex.core.logger import setup_logger
from smartlex.text.text_extraction import extract_text_docx, extract_text_pdf

logger = setup_logger(__name__)


def extract_keywords_from_file(file_path):
    lower_path = file_path.lower()
    if lower_path.endswith(".pdf"):
        text = extract_text_pdf(file_path)
        return rake_keywords(text), False
    elif lower_path.endswith(".docx"):
        text = extract_text_docx(file_path)
        return rake_keywords(text), False
    elif lower_path.endswith((".png", ".jpg")):
        from smartlex.image.image_extraction import fast_metadata_extract

        keywords = fast_metadata_extract(file_path)
        return keywords, True  # True means deferred deep processing needed
    else:
        return [], False


def process_batch(batch_file):
    result = {}
    deferred_files = []
    if not os.path.exists(batch_file):
        logger.error(f"Batch file not found: {batch_file}")
        return result, deferred_files

    with open(batch_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f]

    for path in paths:
        if os.path.exists(path):
            try:
                kws, needs_defer = extract_keywords_from_file(path)
                if kws:
                    result[path] = kws
                if needs_defer:
                    deferred_files.append(path)
            except Exception as e:
                logger.error(f"Error processing file {path}: {e}")
    return result, deferred_files


def refine_keywords(data, top_n):
    for k, v in data.items():
        c = Counter(v)
        top_values = [w for w, _ in c.most_common(top_n)]
        data[k] = list(set(top_values))
    return data


def process_all_batches(batch_files, top_n):
    D = {}
    all_deferred = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_batch, batch_files))
    for res, deferred in results:
        D.update(res)
        all_deferred.extend(deferred)
    return refine_keywords(D, top_n), all_deferred
