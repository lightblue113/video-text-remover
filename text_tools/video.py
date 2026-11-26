"""Video metadata and input/output validation."""

import io
from contextlib import redirect_stdout

import cv2
from moviepy import VideoFileClip


def get_video_info(video_clip):
    info = {
        "resolution": f"{int(video_clip.w)}x{int(video_clip.h)}",
        "duration": f"{video_clip.duration:.2f}秒",
        "total_frames": int(video_clip.duration * video_clip.fps),
        "fps": f"{video_clip.fps:.2f}",
        "format": "mp4",
    }
    return info


def is_valid_video_file(file):
    try:
        with redirect_stdout(io.StringIO()):
            with VideoFileClip(file) as video_clip:
                return True
    except Exception as e:
        print(f"Invalid video file: {file}, Error: {e}")
        return False


def is_readable_video_file(file):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        cap.release()
        return False
    readable, _ = cap.read()
    cap.release()
    return readable
