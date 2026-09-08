import os
import sys
from pathlib import Path
from smartlex.core.logger import setup_logger

logger = setup_logger("media_extractor")

# ── Tesseract binary auto-detection ────────────────────────────────────────
# On Windows, winget installs Tesseract to a fixed path that may not yet be
# in the *current* PATH (requires shell restart).  We set the cmd explicitly
# so pytesseract works immediately after installation without a reboot.
def _find_tesseract_cmd():
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in candidates:
            if Path(p).exists():
                return p
    return "tesseract"  # fall back to PATH on Linux / macOS


def extract_image_text(file_path):
    """
    Extracts text from images using OCR.
    """
    try:
        import pytesseract
        from PIL import Image

        pytesseract.pytesseract.tesseract_cmd = _find_tesseract_cmd()
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.error(f"Failed to run OCR on {file_path}: {e}")
        return ""


def extract_audio_video_text(file_path):
    """
    Extracts text from audio or video using faster-whisper.
    For video, it first extracts audio.
    """
    try:
        import tempfile
        from faster_whisper import WhisperModel

        audio_path = file_path

        # If it's a video, extract audio first using moviepy
        if file_path.endswith(".mp4"):
            try:
                from moviepy.editor import VideoFileClip

                video = VideoFileClip(file_path)
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio_path = temp_audio.name
                temp_audio.close()
                video.audio.write_audiofile(temp_audio_path, logger=None, verbose=False)
                audio_path = temp_audio_path
            except Exception as e:
                logger.error(f"Failed to extract audio from video {file_path}: {e}")
                return ""

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)

        transcription = []
        for segment in segments:
            transcription.append(segment.text)

        # Clean up temporary audio file if we created one
        if audio_path != file_path and os.path.exists(audio_path):
            os.remove(audio_path)

        return " ".join(transcription)
    except Exception as e:
        logger.error(f"Failed to transcribe {file_path}: {e}")
        return ""


def fast_metadata_extract(file_path):
    """
    Stage 1: Fast metadata extraction for immediate indexing.
    Returns basic keywords like the filename without extension.
    """
    path = Path(file_path)
    # Split filename by spaces, underscores, dashes
    import re

    name_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", path.stem)
    keywords = [w.lower() for w in name_clean.split() if len(w) > 2]
    return keywords
