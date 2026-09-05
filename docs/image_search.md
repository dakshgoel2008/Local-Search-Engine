# Image Processing & Visual-Similarity Indexing

This document describes the image indexing/retrieval layer added to SmartLex.
It lets the engine index image files and later find visually similar images for
a given query image — complementing the existing lexical (keyword) search
rather than replacing it.

## 1. Overall architecture

The image layer lives entirely in `src/smartlex/image/` and reuses the
project's existing abstractions:

* **Configuration** — all tunables live in `core/config.py` (`DEFAULT_CONFIG`).
* **Storage** — visual features are persisted to a JSON file (`image_index.json`)
  using the same `core/index_manager` helpers that back the text index
  (`output.json`). No new database is introduced.
* **Indexing pipeline** — image feature extraction is hooked into the existing
  batch processor (`core/processor.py`), so images are indexed in the same
  parallel scan/index pass as PDFs and DOCX files.
* **Logging / error handling** — uses `core/logger.setup_logger`, matching the
  rest of the codebase.

The pipeline:

```
        Image
          │
          ▼
   Preprocessing        (validate, safe-load, downscale, grayscale)
          │
          ▼
        pHash           (coarse / global perceptual hash)
          │
          ▼
  Candidate Retrieval   (Hamming distance, keep top-K)
          │
          ▼
    ORB Matching        (local feature verification on candidates only)
          │
          ▼
  Similarity Ranking    (final_score = α·pHash_sim + β·ORB_sim)
```

### Component map

| Responsibility            | Class / function                        | Module            |
|---------------------------|-----------------------------------------|-------------------|
| Detection & preprocessing | `ImagePreprocessor`                     | `preprocessing.py`|
| Perceptual hashing        | `PhashExtractor`, `hamming_distance`    | `phash.py`        |
| ORB features & matching   | `OrbFeatureExtractor`, `match_descriptors`, `orb_similarity`, `serialize_descriptors`/`deserialize_descriptors` | `orb_features.py` |
| Persistence               | `ImageFeatureStore`                     | `feature_store.py`|
| Index orchestration       | `index_image`, `build_image_record`     | `indexer.py`      |
| Candidate retrieval       | `ImageCandidateRetriever`               | `search.py`       |
| Ranking                   | `ImageSimilarityRanker`                 | `search.py`       |
| Search service / API      | `ImageSearchService`, `search_images`   | `search.py`       |

## 2. Preprocessing

`ImagePreprocessor` (`preprocessing.py`):

1. **Validation** — accepts `.jpg/.jpeg/.png/.webp/.bmp/.tif/.tiff`
   (`IMAGE_SUPPORTED_FORMATS`).
2. **Safe loading** — decodes via a numpy buffer + `cv2.imdecode` (unicode-path
   safe). Missing, empty and corrupted files raise `ImageLoadError`;
   unsupported extensions raise `UnsupportedImageError`.
3. **Memory bound** — images are downscaled so the longest edge is at most
   `IMAGE_MAX_DIMENSION` (default 1024 px). Small images are never upscaled.
4. **Determinism** — grayscale conversion + `INTER_AREA` resize are
   deterministic, so the same bytes always yield the same features (essential
   for query/index consistency).

## 3. pHash generation

`PhashExtractor` (`phash.py`) implements the classic **DCT-based perceptual
hash**:

1. Resize grayscale to `hash_size × highfreq_factor` (default 8×4 = 32²).
2. Compute the 2-D DCT.
3. Keep the top-left `hash_size × hash_size` low-frequency block.
4. Threshold each coefficient against the block median (excluding the DC term,
   which only encodes brightness) → one bit per coefficient.

Result: a 64-bit hash stored as a 16-char hex string. This is **not** a
cryptographic hash — a one-pixel change moves it by ≈0 bits, whereas SHA/MD5
would change completely.

## 4. Hamming-distance candidate retrieval

`hamming_distance(a, b)` counts differing bits between two hex hashes. Smaller =
more similar; 0 = identical appearance.

`ImageCandidateRetriever.retrieve(query_phash)`:

* Computes the Hamming distance from the query hash to every indexed hash.
* Discards candidates beyond `IMAGE_PHASH_MAX_DISTANCE` (default 64 = no cut).
* Returns the closest **`IMAGE_CANDIDATE_TOP_K`** (default 10) candidates.

This is the cheap, coarse stage that shrinks the candidate set before the
expensive ORB stage.

## 5. ORB extraction

`OrbFeatureExtractor` (`orb_features.py`) wraps `cv2.ORB_create` with fully
configurable parameters: `ORB_N_FEATURES`, `ORB_SCALE_FACTOR`, `ORB_N_LEVELS`,
`ORB_FAST_THRESHOLD`. `extract(gray)` returns `(keypoints, descriptors)`;
`descriptors` is an `(N, 32) uint8` array, or `None` when no keypoints were
found. Descriptors are computed **once at index time** and persisted.

## 6. ORB descriptor storage

`serialize_descriptors` encodes the `uint8` descriptor array as
`{data: base64, shape: [N, 32], dtype: "uint8"}`; `deserialize_descriptors`
reverses it exactly (verified by the test suite). Images with no descriptors
store `null`. Everything fits in the JSON `image_index.json` record:

```json
{
  "path": "/abs/path/img.jpg",
  "phash": "d1e2f3...",
  "num_bits": 64,
  "orb": {"data": "<base64>", "shape": [500, 32], "dtype": "uint8"},
  "num_keypoints": 500,
  "width": 1024, "height": 768, "size_bytes": 51234
}
```

## 7. ORB matching

For each pHash candidate, `orb_similarity(query_desc, candidate_desc)`:

* Uses a **Brute-Force Hamming matcher** (`cv2.NORM_HAMMING`) with `knnMatch(k=2)`.
* Applies **Lowe's ratio test** (`ORB_MATCH_RATIO`, default 0.75) to keep only
  distinctive matches.
* Gracefully handles missing / too-few (<2) / incompatible descriptors by
  returning an empty match list (never crashes).

Robustness: ORB is rotation-invariant and its scale pyramid handles moderate
scaling; Hamming matching + the ratio test tolerate minor illumination changes
and differing image sizes.

## 8. Final similarity score

For each candidate:

```
phash_similarity = 1 − hamming_distance / num_bits          # in [0, 1]
orb_similarity   = good_matches / max(#query_desc, #cand_desc)   # in [0, 1]
final_score      = α · phash_similarity + β · orb_similarity
```

* `α = IMAGE_SCORE_ALPHA` (default 0.4), `β = IMAGE_SCORE_BETA` (default 0.6).
* α and β are **normalised at runtime** so they always sum to 1.

**Why this normalisation.** pHash distance is normalised by hash width
(`num_bits`, derived from the configured hash size — never hard-coded) so it is
comparable across configurations. ORB is normalised by the **larger** descriptor
count. This matters because `knnMatch(query, candidate)` yields at most one
match per *query* descriptor, but with `crossCheck=False` several query
descriptors can map onto the same few candidate descriptors — so the good-match
count can exceed `min(len_q, len_c)`. Dividing by the smaller set would then
saturate the score at `1.0` whenever one image is much sparser than the other
(e.g. a near-blank or unrelated candidate with a handful of descriptors could
score ~1.0 against a feature-rich query). Dividing by the larger set is
guaranteed to stay in `[0, 1]` (`good ≤ len_q ≤ max`) and does not inflate for
asymmetric keypoint counts. Both similarities land in `[0, 1]`, so the weighted
sum is too, and the weights are directly interpretable.

**Why pHash first, ORB second.** pHash is O(1) per comparison (a 64-bit XOR +
popcount) and captures global appearance, making it ideal for cheaply pruning
millions of images down to a handful of candidates. ORB is far more expensive
(keypoint detection + descriptor matching) but verifies *local* structure and
survives rotation/scale/illumination changes. Running ORB only on the top-K
pHash candidates gives accurate ranking at a fraction of the cost of matching
every image.

## 9. Configuration parameters

All in `core/config.py` (`DEFAULT_CONFIG`); override via `config.json`.

| Key | Default | Meaning |
|-----|---------|---------|
| `IMAGE_INDEX_FILE` | `image_index.json` | Visual-feature store path |
| `IMAGE_SUPPORTED_FORMATS` | jpg/jpeg/png/webp/bmp/tif/tiff | Accepted formats |
| `IMAGE_MAX_DIMENSION` | 1024 | Downscale longest edge to this |
| `PHASH_HASH_SIZE` | 8 | Hash side → 64-bit hash |
| `PHASH_HIGHFREQ_FACTOR` | 4 | DCT input = 8×4 = 32² |
| `IMAGE_CANDIDATE_TOP_K` | 10 | Candidates kept after pHash |
| `IMAGE_PHASH_MAX_DISTANCE` | 64 | Max pHash distance to consider |
| `ORB_N_FEATURES` | 500 | Max ORB keypoints |
| `ORB_SCALE_FACTOR` | 1.2 | ORB pyramid scale factor |
| `ORB_N_LEVELS` | 8 | ORB pyramid levels |
| `ORB_FAST_THRESHOLD` | 20 | FAST corner threshold |
| `ORB_MATCH_RATIO` | 0.75 | Lowe ratio-test threshold |
| `IMAGE_SCORE_ALPHA` | 0.4 | pHash weight (α) |
| `IMAGE_SCORE_BETA` | 0.6 | ORB weight (β) |

## 10. API usage

### Indexing

Images on disk are indexed automatically as part of the normal indexing pass
(run the app via `python run.py` on first launch, or delete `output.json` to
re-index). The batch processor extracts visual features for every supported
image and `IndexingThread` persists them to `image_index.json`.

Programmatically:

```python
from smartlex.image import index_image, ImageFeatureStore

store = ImageFeatureStore.load()          # or ImageFeatureStore(index_file=...)
record = index_image("/path/to/photo.jpg")  # None on unsupported/corrupt
if record:
    store.add(record)
store.save()
```

### Searching

```python
from smartlex.image import search_images, ImageSearchService

# Convenience function (mirrors core.search_engine.search):
results = search_images("/path/to/query.jpg", top_k=10)

# Or via the service (reuse a loaded store across queries):
service = ImageSearchService()            # loads image_index.json
results = service.search("/path/to/query.jpg", top_k=10)

for r in results:
    print(r["rank"], r["final_score"], r["path"])
```

Each result dict contains:

```python
{
  "path": "...",              # image reference
  "phash_distance": 2,        # Hamming distance from query
  "phash_similarity": 0.969,
  "orb_similarity": 0.444,
  "final_score": 0.654,
  "num_keypoints": 500,
  "rank": 1
}
```

## 11. Limitations & edge cases

* **Handled gracefully** (covered by tests): corrupted/unsupported/missing
  files (skipped at index time, raise at query time), images with no keypoints
  (empty descriptors → ORB score 0, no crash), duplicate images (both
  returned, distance 0), very large images (downscaled), and asking for more
  candidates than exist (returns what is available).
* **pHash is global**: it can rate two images with similar overall layout but
  different details as close — that is exactly why ORB verification follows.
* **Heavy crops / large rotations (>~30°) / strong perspective** reduce ORB
  matches; ORB targets moderate transformations.
* **Linear scan** for candidate retrieval is O(N) in the number of indexed
  images. This is fast (64-bit popcount) for tens of thousands of images; for
  much larger corpora a BK-tree / VP-tree over the hashes would be the next
  step.
* Text keyword indexing for images (filename tokens + deferred OCR) is
  **unchanged** and independent of the visual index.
```
