"""Image processing and visual-similarity indexing for SmartLex.

Public API (see ``docs/image_search.md`` for the full pipeline description):

    from smartlex.image import index_image, search_images, ImageSearchService

Pipeline: Image -> Preprocessing -> pHash -> Candidate Retrieval -> ORB -> Ranking.
"""

from smartlex.image.feature_store import ImageFeatureStore
from smartlex.image.indexer import build_image_record, index_image
from smartlex.image.orb_features import OrbFeatureExtractor
from smartlex.image.phash import PhashExtractor, hamming_distance
from smartlex.image.preprocessing import ImagePreprocessor
from smartlex.image.search import (
    ImageCandidateRetriever,
    ImageSearchService,
    ImageSimilarityRanker,
    search_images,
)

__all__ = [
    "ImagePreprocessor",
    "PhashExtractor",
    "hamming_distance",
    "OrbFeatureExtractor",
    "ImageFeatureStore",
    "index_image",
    "build_image_record",
    "ImageCandidateRetriever",
    "ImageSimilarityRanker",
    "ImageSearchService",
    "search_images",
]
