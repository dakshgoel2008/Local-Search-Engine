"""Query-time image similarity search.

Implements the *query half* of the pipeline::

    query image
      -> preprocessing + pHash + ORB (query features, computed once)
      -> pHash Hamming-distance candidate retrieval (coarse, cheap, top-K)
      -> ORB descriptor matching on candidates only (fine, expensive)
      -> combined similarity ranking

Splitting the work this way is what keeps search fast: pHash reduces the
candidate set to a configurable ``top_k`` before the costly ORB matching runs,
so ORB never has to compare the query against every indexed image.

Scoring
-------
For each surviving candidate::

    phash_similarity = 1 - hamming_distance / num_bits          # in [0, 1]
    orb_similarity   = good_matches / max(#query_desc, #cand_desc)  # in [0, 1]
    final_score      = alpha * phash_similarity + beta * orb_similarity

``alpha`` and ``beta`` are configurable (``IMAGE_SCORE_ALPHA`` /
``IMAGE_SCORE_BETA``) and are normalised so they sum to 1.  pHash contributes
coarse/global appearance; ORB contributes local feature verification.
"""

from smartlex.core.config import load_config
from smartlex.core.logger import setup_logger
from smartlex.image.feature_store import ImageFeatureStore
from smartlex.image.indexer import build_image_record
from smartlex.image.orb_features import deserialize_descriptors, orb_similarity
from smartlex.image.phash import (
    PhashExtractor,
    hamming_distance,
    similarity_from_distance,
)

logger = setup_logger("image_search")


class ImageCandidateRetriever:
    """Retrieve the top-K visually closest candidates by pHash distance."""

    def __init__(self, store, top_k=None, max_distance=None):
        cfg = load_config()
        self.store = store
        self.top_k = int(top_k if top_k is not None else cfg["IMAGE_CANDIDATE_TOP_K"])
        self.max_distance = int(
            max_distance if max_distance is not None else cfg["IMAGE_PHASH_MAX_DISTANCE"]
        )

    def retrieve(self, query_phash):
        """Return ``[(record, distance), ...]`` sorted by ascending distance.

        Records without a usable pHash are skipped; those beyond
        ``max_distance`` are filtered out; the closest ``top_k`` remain.
        """
        scored = []
        for record in self.store.all().values():
            candidate_phash = record.get("phash")
            if not candidate_phash:
                continue
            try:
                distance = hamming_distance(query_phash, candidate_phash)
            except ValueError:
                continue
            if distance <= self.max_distance:
                scored.append((record, distance))

        scored.sort(key=lambda item: item[1])
        return scored[: self.top_k]


class ImageSimilarityRanker:
    """Combine pHash and ORB similarity into a final ranked list."""

    def __init__(self, alpha=None, beta=None):
        cfg = load_config()
        alpha = cfg["IMAGE_SCORE_ALPHA"] if alpha is None else alpha
        beta = cfg["IMAGE_SCORE_BETA"] if beta is None else beta
        total = float(alpha) + float(beta)
        if total <= 0:
            # Degenerate config -> fall back to an even split.
            self.alpha, self.beta = 0.5, 0.5
        else:
            # Normalise defensively so alpha + beta == 1.
            self.alpha = float(alpha) / total
            self.beta = float(beta) / total

        # Fallback hash width for malformed records that lack "num_bits",
        # derived from the configured pHash size rather than hard-coded.
        self._default_num_bits = PhashExtractor().num_bits

    def rank(self, query_desc, candidates):
        """Rank ``candidates`` (``[(record, distance), ...]``).

        Returns a list of result dicts sorted by descending ``final_score``.
        """
        results = []
        for record, distance in candidates:
            num_bits = record.get("num_bits") or self._default_num_bits
            # Normalise the (already-computed) Hamming distance into similarity.
            p_sim = similarity_from_distance(distance, num_bits)

            candidate_desc = deserialize_descriptors(record.get("orb"))
            o_sim = orb_similarity(query_desc, candidate_desc)

            final_score = self.alpha * p_sim + self.beta * o_sim
            results.append(
                {
                    "path": record.get("path"),
                    "phash_distance": distance,
                    "phash_similarity": round(p_sim, 6),
                    "orb_similarity": round(o_sim, 6),
                    "final_score": round(final_score, 6),
                    "num_keypoints": record.get("num_keypoints", 0),
                }
            )

        results.sort(key=lambda item: item["final_score"], reverse=True)
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
        return results


class ImageSearchService:
    """End-to-end image similarity search over an :class:`ImageFeatureStore`."""

    def __init__(self, store=None, top_k=None, alpha=None, beta=None):
        self.store = store or ImageFeatureStore.load()
        self.retriever = ImageCandidateRetriever(self.store, top_k=top_k)
        self.ranker = ImageSimilarityRanker(alpha=alpha, beta=beta)

    def search(self, query_image_path, top_k=None):
        """Search for images visually similar to *query_image_path*.

        The query image is preprocessed with the exact same pipeline used at
        index time, guaranteeing consistent pHash/ORB features.  Returns a
        ranked list of result dicts::

            {path, phash_distance, phash_similarity, orb_similarity,
             final_score, num_keypoints, rank}
        """
        if top_k is not None:
            self.retriever.top_k = int(top_k)

        # Compute query features once (raises on unsupported/corrupted input).
        query_record = build_image_record(query_image_path)
        query_desc = deserialize_descriptors(query_record["orb"])

        candidates = self.retriever.retrieve(query_record["phash"])
        if not candidates:
            logger.info(f"No candidates found for query image {query_image_path}")
            return []
        return self.ranker.rank(query_desc, candidates)


def search_images(query_image_path, image_index=None, top_k=None, alpha=None, beta=None):
    """Convenience function mirroring :func:`smartlex.core.search_engine.search`.

    *image_index* may be an :class:`ImageFeatureStore`, a raw ``{path: record}``
    dict, or ``None`` (loads the persisted store).
    """
    if image_index is None:
        store = ImageFeatureStore.load()
    elif isinstance(image_index, ImageFeatureStore):
        store = image_index
    else:
        store = ImageFeatureStore(records=image_index)

    service = ImageSearchService(store=store, top_k=top_k, alpha=alpha, beta=beta)
    return service.search(query_image_path, top_k=top_k)
