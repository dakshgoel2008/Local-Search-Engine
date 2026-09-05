# Smart Lexical Search Engine - Developer Guide

##  Project Structure

Our project follows a modular, decoupled architecture allowing separate teams to develop independently without merge conflicts:

```text
SmartLex/
│
├── pyproject.toml         # Package configuration
├── run.py                 # Application and testing entry point
├── test_patent_logic.py   # Test suite for resource monitor
│
├── src/smartlex/
│   ├── main.py            # Core application entry
│   ├── text/              # Text processing, PDF/DOCX parsing, and keyword extraction
│   ├── image/             # Image processing and OCR logic
│   ├── core/              # Central orchestration, scanners, and configuration
│   └── gui/               # PyQt5 interface and background threads
│
├── requirements.txt
├── config.json
└── README.md
```

### Module Responsibilities
- **Text Developers**: Work exclusively inside the `src/smartlex/text/` directory. Updates to NLP algorithms, `fitz` (PyMuPDF) handlers, or keyword extraction logic happen here.
- **Image Developers**: Work exclusively inside the `src/smartlex/image/` directory. Updates to OCR logic, image metadata extraction, and the visual-similarity pipeline (pHash + ORB) happen here. See [`docs/image_search.md`](docs/image_search.md) for the image indexing/retrieval architecture and API.
- **Core Developers**: Work inside `src/smartlex/core/` to orchestrate multithreading (`processor.py`), config, and resource monitoring logic.

*(Note: Audio and Video processing pipelines have been removed. The current focus is entirely on Text and Image optimization).*

---

## Testing

You have multiple ways to test your changes locally:

### 1. Test the Application (GUI & End-to-End)
Run the application entry point to launch the UI and ensure your new logic hasn't broken the overarching search and indexing loops:
```bash
python run.py
```

### 2. Test Background Resource Monitor (Patent Logic)
Test that the background ML indexing thread successfully pauses when heavy system usage (e.g., a video game) is detected, and resumes safely when idle:
```bash
python test_patent_logic.py
```

### 3. Test the Image Visual-Similarity Pipeline (pytest)
Run the image pipeline suite (pHash, ORB, preprocessing, end-to-end indexing and search). The tests generate their own synthetic images, so no assets are required:
```bash
python -m pytest tests/ -q
```

---

## Manual Testing: Image Visual-Similarity Search

This walks you through indexing real image files and running a visual similarity
search **by hand**, so you can confirm the feature works end-to-end on your own
images. Full design docs are in [`docs/image_search.md`](docs/image_search.md).

### Step 0 — Install dependencies (one time)
The image pipeline needs OpenCV and NumPy (already listed in `requirements.txt`):
```bash
pip install -r requirements.txt
```

### Step 1 — Prepare a few test images
Put several images in one folder (any of `.jpg .jpeg .png .webp .bmp .tif .tiff`).
For a meaningful test, include:
- one **original** image (e.g. `photo.png`),
- a **near-duplicate** of it (resize or re-save it as `.jpg`, e.g. `photo_small.jpg`),
- one or two **unrelated** images.

Tip: to make a resized near-duplicate copy on **any OS** (Windows/macOS/Linux),
run this (uses Pillow, already installed):
```bash
python -c "from PIL import Image; im=Image.open('photo.png'); im.resize((400,400)).save('photo_small.jpg')"
```

### Step 2 — Index the images and run a search
Run this script from the project root. Edit `IMAGE_DIR` and `QUERY_IMAGE` to point
at your folder and the image you want to search with. It uses the exact same
components the app uses (`index_image`, `ImageFeatureStore`, `ImageSearchService`).

```bash
python - <<'PY'
import os, sys, glob
sys.path.insert(0, "src")
from smartlex.image.indexer import index_image
from smartlex.image.feature_store import ImageFeatureStore
from smartlex.image.search import ImageSearchService

IMAGE_DIR   = "test_images"          # <-- folder containing your images
QUERY_IMAGE = "test_images/photo.png"  # <-- image to search with

# 1. Index every supported image in the folder into image_index.json
store = ImageFeatureStore.load()     # loads existing index, or starts empty
exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
count = 0
for path in glob.glob(os.path.join(IMAGE_DIR, "*")):
    if path.lower().endswith(exts):
        record = index_image(path)   # returns None for corrupt/unsupported files
        if record:
            store.add(record)
            count += 1
store.save()
print(f"Indexed {count} image(s) -> image_index.json ({len(store)} total records)")

# 2. Search with the query image
results = ImageSearchService(store=store).search(QUERY_IMAGE, top_k=10)
print(f"\nTop matches for {QUERY_IMAGE}:\n")
print(f"{'rank':<5}{'final':<8}{'phash_dist':<12}{'orb_sim':<9}file")
for r in results:
    print(f"{r['rank']:<5}{r['final_score']:<8.3f}{r['phash_distance']:<12}{r['orb_similarity']:<9.3f}{os.path.basename(r['path'])}")
PY
```

### Step 3 — What you should see
- The **query image itself** (if it is in the folder) ranks **#1** with `final ≈ 1.000` and `phash_dist = 0`.
- The **near-duplicate** ranks **#2** with a high score and a small `phash_dist`.
- **Unrelated** images rank lower with larger `phash_dist` and low `orb_sim`.

`phash_dist` = perceptual-hash Hamming distance (smaller = more similar);
`orb_sim` = local-feature match score; `final` = the combined ranking score.

### Step 4 — Verify persistence (optional)
The index is saved to `image_index.json` in the project root. Re-running **only**
the search part of the script (Step 2 without re-indexing) still returns results,
proving features are persisted and reloaded — no re-computation needed.

### Step 5 — Try the edge cases (optional)
Drop these into your folder and re-run Step 2 to confirm nothing crashes:
- a **corrupted** file (e.g. `echo "not an image" > test_images/broken.png`) — it is skipped with a warning,
- an **unsupported** file (e.g. a `.txt`) — ignored,
- a **blank/solid-colour** image — indexed with no ORB keypoints, ranked by pHash only.