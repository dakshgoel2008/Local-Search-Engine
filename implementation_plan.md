# SmartLex Productization & Resume Upgrade Plan

To transform SmartLex from a cool script into a **"Wow" resume project and a real product**, we need to upgrade its architecture from basic keyword matching to industry-standard search engine mechanics, add modern AI features, and improve the user experience so it feels like a native OS integration (like Apple Spotlight or Windows PowerToys).

Here is the proposed roadmap to make this project stand out to recruiters and be genuinely useful to end-users.

## User Review Required

> [!IMPORTANT]
> Please review the features below. Let me know which ones you want to prioritize or if you want to implement all of them. Once approved, I will break this down into actionable tasks and we can start building!

## Proposed Features & Upgrades

### 1. 🏗️ True Search Engine Architecture (Inverted Index)
**Current:** Saves a simple mapping of `File -> Keywords` in a giant `output.json`. This is slow to load into memory and doesn't scale.
**Proposed:** Migrate the backend to use **SQLite FTS5 (Full-Text Search)** or **Whoosh**.
*   **Why for Resume:** You can say: *"Engineered a highly-scalable inverted index using SQLite FTS5, reducing search latency from O(N) to O(1) and enabling advanced BM25 ranking algorithms."*
*   **Product Value:** Instant search across millions of documents, zero memory bloat, and the ability to show "text snippets" (context of where the word was found in the document) in the UI.

### 2. 👁️ Background File System Monitoring (Real-time Updates)
**Current:** Indexes the drive once. If a user downloads a new PDF, they must manually rescan.
**Proposed:** Integrate the `watchdog` Python library to run a lightweight background daemon that listens to OS-level file creation/modification/deletion events.
*   **Why for Resume:** *"Implemented an event-driven background daemon using OS-level file watchers to maintain a real-time, synchronized index with zero user intervention."*
*   **Product Value:** This is what separates a script from a real product. The app becomes invisible and "just works."

### 3. 🧠 Semantic Search (Local AI / Vector Embeddings)
**Current:** Exact keyword matching (e.g., searching "money" won't find "finance").
**Proposed:** Introduce a "Semantic Search" mode. We can use a tiny, blazing-fast local embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and a local vector database (like `FAISS` or `ChromaDB`).
*   **Why for Resume:** This is the ultimate buzzword. *"Integrated local Natural Language Processing (NLP) and Vector Embeddings for semantic similarity search, allowing users to find documents based on meaning rather than exact keywords."*
*   **Product Value:** Massive boost in search accuracy and utility.

### 4. 🚀 Spotlight-style Global Hotkey & System Tray
**Current:** A standard desktop window that the user has to manually open.
**Proposed:** Hide the app in the **System Tray**. Add a global hotkey (e.g., `Ctrl + Space` or `Alt + Space`) to instantly summon a floating, beautiful search bar in the middle of the screen (exactly like Mac Spotlight or PowerToys Run).
*   **Why for Resume:** Shows strong product sense and UX focus.
*   **Product Value:** Unmatched convenience. Users will use it 10x more often if it's always one keystroke away.

### 5. 🧪 Engineering Maturity (CI/CD & Testing)
**Current:** No automated tests.
**Proposed:** Write a solid test suite using `pytest` and set up **GitHub Actions** to automatically run tests and build the `.exe` every time you push code.
*   **Why for Resume:** *"Established a full CI/CD pipeline using GitHub Actions for automated testing and cross-platform executable deployment."* Recruiters love seeing this.

---

## Next Steps / Open Questions

> [!QUESTION]
> 1.  **Which phase excites you the most?** We should probably start with **Phase 1 (SQLite FTS5 Inverted Index)** as it builds the foundation for everything else, followed by **Phase 4 (Global Hotkey/System Tray)** for immediate visual impact.
> 2.  **Are you open to adding new dependencies** like `watchdog` for real-time monitoring and `keyboard` or `pynput` for the global hotkeys?

Let me know what you think, and I will create a `task.md` to begin execution!
