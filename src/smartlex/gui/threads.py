import json
import os
import subprocess
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from smartlex.core.config import load_config
from smartlex.core.index_manager import generate_autocomplete, save_index
from smartlex.core.logger import setup_logger
from smartlex.core.processor import process_all_batches
from smartlex.core.scanner import scan_all_drives

logger = setup_logger(__name__)


class IndexingThread(QThread):
    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(dict, list)
    error_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = load_config()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        logger.info("Indexing cancellation requested")

    def run(self):
        try:
            if self._is_cancelled:
                return

            self._emit_progress("Starting indexing process...")

            # Step 1 & 2: Scan all drives and create batch files
            self._emit_progress(
                "Scanning all system drives for documents. This may take a moment..."
            )
            batch_files = scan_all_drives(
                extensions=self.cfg["SUPPORTED_FORMATS"],
                num_processes=self.cfg["NUM_PROCESSES"],
                output_folder=self.cfg["INDEX_FOLDER"],
                progress_callback=self._emit_progress,
            )

            if self._is_cancelled:
                self._emit_progress("Indexing cancelled")
                return

            if not batch_files:
                self._emit_error("No documents found on the system.")
                return

            # Step 3: Process files
            self._emit_progress(
                f"Processing {len(batch_files)} batch(es) in parallel..."
            )
            D, deferred_files, image_features = process_all_batches(
                batch_files, self.cfg["TOP_KEYWORDS"]
            )

            # Step 3b: Persist visual image features (pHash + ORB descriptors)
            if image_features:
                from smartlex.image.feature_store import ImageFeatureStore

                self._emit_progress(
                    f"Saving visual features for {len(image_features)} image(s)..."
                )
                image_store = ImageFeatureStore.load(self.cfg["IMAGE_INDEX_FILE"])
                image_store.update(image_features)
                image_store.save()
                logger.info(
                    f"Image feature index saved with {len(image_store)} entries"
                )

            # Save deferred files list
            if deferred_files:
                deferred_path = Path(
                    self.cfg.get("DEFERRED_INDEX_FILE", "deferred_index.json")
                )
                with open(deferred_path, "w", encoding="utf-8") as f:
                    json.dump(deferred_files, f, indent=2)

            if self._is_cancelled:
                self._emit_progress("Indexing cancelled")
                return

            if not D:
                self._emit_error(
                    "No data indexed. Check if PDF files exist in specified directories."
                )
                return

            # Step 4: Save index
            self._emit_progress("Saving index to disk...")
            save_index(D, self.cfg["OUTPUT_FILE"])
            self._emit_progress(f"Index saved with {len(D)} entries")

            if self._is_cancelled:
                self._emit_progress("Indexing cancelled")
                return

            # Step 5: Generate autocomplete
            self._emit_progress("Generating autocomplete data...")
            words = generate_autocomplete(D, self.cfg["AUTOCOMPLETE_WORDS"])

            autocomplete_path = Path(self.cfg["AUTOCOMPLETE_FILE"])
            autocomplete_path.parent.mkdir(parents=True, exist_ok=True)

            with open(autocomplete_path, "w", encoding="utf-8") as f:
                json.dump(words, f, indent=2)

            self._emit_progress(f"Autocomplete saved with {len(words)} words")

            # Finish
            self._emit_progress("✓ Indexing completed successfully!")
            self.finished_signal.emit(D, words)
            logger.info(
                f"Indexing completed: {len(D)} files indexed, {len(words)} autocomplete words"
            )

        except Exception as e:
            error_msg = f"Indexing error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._emit_error(error_msg)

    def _emit_progress(self, message):
        if not self._is_cancelled:
            self.progress.emit(message)

    def _emit_error(self, message):
        logger.error(message)
        self.error_signal.emit(message)
        self.progress.emit(f"✗ ERROR: {message}")


class DeferredIndexingThread(QThread):
    new_data_signal = pyqtSignal(str, list)  # path, keywords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = load_config()
        self._is_cancelled = False
        self.deferred_file = Path(
            self.cfg.get("DEFERRED_INDEX_FILE", "deferred_index.json")
        )

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        from smartlex.core.resource_monitor import monitor
        from smartlex.image.image_extraction import extract_image_text
        from smartlex.text.keyword_extraction import rake_keywords
        import time

        while not self._is_cancelled:
            if not self.deferred_file.exists():
                time.sleep(5)
                continue

            try:
                with open(self.deferred_file, "r", encoding="utf-8") as f:
                    deferred_files = json.load(f)
            except Exception:
                deferred_files = []

            if not deferred_files:
                time.sleep(5)
                continue

            # Process one file at a time
            file_to_process = deferred_files[0]

            # WAIT until CPU is free enough
            monitor.can_run_heavy_tasks.wait()

            if self._is_cancelled:
                break

            logger.info(f"Deep processing started for {file_to_process}")

            try:
                lower_path = file_to_process.lower()
                text = ""
                if lower_path.endswith((".png", ".jpg")):
                    text = extract_image_text(file_to_process)

                if text:
                    kws = rake_keywords(text)
                    if kws:
                        self.new_data_signal.emit(file_to_process, kws)
            except Exception as e:
                logger.error(f"Error in deep processing for {file_to_process}: {e}")

            # Remove from queue and save
            deferred_files.pop(0)
            try:
                with open(self.deferred_file, "w", encoding="utf-8") as f:
                    json.dump(deferred_files, f, indent=2)
            except Exception as e:
                logger.error(f"Error updating deferred index queue: {e}")

            # Sleep briefly to allow CPU to catch up
            time.sleep(1)
