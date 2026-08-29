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
- **Image Developers**: Work exclusively inside the `src/smartlex/image/` directory. Updates to OCR logic and image metadata extraction happen here.
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