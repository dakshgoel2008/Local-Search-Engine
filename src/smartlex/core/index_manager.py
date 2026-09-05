import json
import os
import tempfile
from collections import Counter
from pathlib import Path


def save_index(data, output_file):
    """Persist *data* as JSON to *output_file* atomically.

    Writes to a temporary file in the same directory and then ``os.replace``s
    it into place.  ``os.replace`` is atomic on POSIX and Windows when the
    source and destination are on the same filesystem, so an interrupted or
    crashing write can never leave a half-written / corrupted index behind: the
    previous index remains intact until the new one is fully written.
    """
    output_path = Path(output_file)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    directory = str(output_path.parent) if str(output_path.parent) else "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    except Exception:
        # Clean up the temp file on any failure; leave the existing index alone.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_index(output_file):
    with open(output_file, "r") as f:
        return json.load(f)


def generate_autocomplete(data, top_n):
    words = []
    for kws in data.values():
        words.extend(kws)
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]
