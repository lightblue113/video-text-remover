"""Output directory handling, resumable state, and pending input discovery."""

import glob
import json
import os
import time
from datetime import datetime, timezone

from .video import is_readable_video_file, is_valid_video_file

STATE_FILENAME = ".text_remover_state.json"


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return True
        except OSError as error:
            print(f"Error creating directory {directory}: {error}")
            return False

    temp_file = os.path.join(directory, f"temp_{time.time()}.tmp")
    try:
        with open(temp_file, "w") as f:
            f.write("test")
        os.remove(temp_file)
        return True
    except Exception as e:
        print(f"No write permission in directory {directory}: {e}")
        return False


def _state_path(output_dir):
    return os.path.join(output_dir, STATE_FILENAME)


def _video_fingerprint(file):
    stat = os.stat(file)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _roi_for_state(roi):
    return list(roi) if roi is not None else None


def load_processing_state(output_dir):
    """Load the resumable processing manifest, if one exists."""
    state_path = _state_path(output_dir)
    if not os.path.exists(state_path):
        state_path = os.path.join(output_dir, ".watermark_remover_state.json")
    if not os.path.exists(state_path):
        return {"version": 1, "videos": {}}

    try:
        with open(state_path, "r", encoding="utf-8") as state_file:
            state = json.load(state_file)
        if not isinstance(state, dict) or not isinstance(state.get("videos"), dict):
            raise ValueError("state file must contain a videos object")
        return state
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Warning: could not read {state_path}; starting a new state: {error}")
        return {"version": 1, "videos": {}}


def save_processing_state(output_dir, state):
    """Atomically save the resumable processing manifest."""
    state_path = _state_path(output_dir)
    temporary_path = f"{state_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2)
        state_file.write("\n")
        state_file.flush()
        os.fsync(state_file.fileno())
    os.replace(temporary_path, state_path)


def record_video_state(
    output_dir,
    state,
    input_file,
    status,
    roi,
    output_file=None,
    video_info=None,
    processing_time=None,
    error=None,
):
    """Record one video's result immediately so an interrupted run can resume."""
    entry = {
        "input": os.path.abspath(input_file),
        "fingerprint": _video_fingerprint(input_file),
        "status": status,
        "roi": _roi_for_state(roi),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if output_file is not None:
        entry["output"] = os.path.abspath(output_file)
    if video_info is not None:
        entry["video_info"] = video_info
    if processing_time is not None:
        entry["processing_time"] = processing_time
    if error is not None:
        entry["error"] = str(error)

    state.setdefault("version", 1)
    state.setdefault("videos", {})[os.path.abspath(input_file)] = entry
    save_processing_state(output_dir, state)


def _state_entry_matches(entry, input_file, roi):
    try:
        return (
            entry.get("fingerprint") == _video_fingerprint(input_file)
            # With GUI selection, the saved ROI is the selection made on the
            # previous run and is intentionally accepted for resume checks.
            and (roi is None or entry.get("roi") == _roi_for_state(roi))
        )
    except OSError:
        return False


def get_unprocessed_videos(
    input_dir, output_dir, state=None, roi=None, retry_failed=False
):
    """Return valid inputs not already completed or recorded as failed."""
    if state is None:
        state = load_processing_state(output_dir)

    processed_stems = set()
    for output_file in glob.glob(os.path.join(output_dir, "*.mp4")):
        if is_readable_video_file(output_file):
            processed_stems.add(os.path.splitext(os.path.basename(output_file))[0])

    pending_videos = []
    for input_file in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if not os.path.isfile(input_file) or not is_valid_video_file(input_file):
            continue
        video_stem = os.path.splitext(os.path.basename(input_file))[0]
        entry = state.get("videos", {}).get(os.path.abspath(input_file))
        if entry and _state_entry_matches(entry, input_file, roi):
            if entry.get("status") == "success" and entry.get("output"):
                if is_readable_video_file(entry["output"]):
                    print(f"Skipping already processed video: {input_file}")
                    continue
            elif entry.get("status") == "failed" and not retry_failed:
                print(
                    f"Skipping previously failed video: {input_file} "
                    f"({entry.get('error', 'unknown error')})"
                )
                continue
        elif entry is None and video_stem in processed_stems:
            # Outputs produced by older versions had no state manifest.
            print(f"Skipping already processed video: {input_file}")
            continue
        pending_videos.append(input_file)
    return pending_videos
