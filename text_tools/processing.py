"""Frame blending, video encoding, and per-device worker execution."""

import os
import time
from datetime import timedelta

import cv2
import numpy as np
import torch
from moviepy import VideoFileClip
from tqdm import tqdm

from .detection import TextDetector
from .inpainting import initialize_lama, lama_inpaint
from .video import get_video_info, is_readable_video_file


class TextProcessor:
    def __init__(self, model, config, roi_coords, roi_mask):
        self.model = model
        self.config = config
        self.roi_coords = roi_coords
        self.roi_mask = roi_mask

    def extract_roi(self, frame_rgb):
        y_min, y_max, x_min, x_max = self.roi_coords
        return frame_rgb[y_min:y_max, x_min:x_max]

    def process_frame(self, frame_rgb, frame_index):
        y_min, y_max, x_min, x_max = self.roi_coords
        roi = self.extract_roi(frame_rgb)

        processed_roi = lama_inpaint(roi, self.roi_mask, self.model, self.config)

        blend_mask = (
            cv2.GaussianBlur(self.roi_mask.astype(np.float32), (21, 21), 0) / 255.0
        )

        result = frame_rgb.copy()
        result[y_min:y_max, x_min:x_max] = (
            blend_mask[:, :, np.newaxis] * processed_roi
            + (1 - blend_mask[:, :, np.newaxis]) * roi
        )

        return np.clip(result, 0, 255).astype(np.uint8, copy=False)


def process_video(video_clip, output_path, text_mask, model, config):
    video_info = get_video_info(video_clip)
    start_time = time.time()

    y_indices, x_indices = np.where(text_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("No text region found in mask")

    video_width = int(video_clip.w)
    video_height = int(video_clip.h)
    margin = 10
    y_min, y_max = (
        max(0, np.min(y_indices) - margin),
        min(video_height, np.max(y_indices) + margin),
    )
    x_min, x_max = (
        max(0, np.min(x_indices) - margin),
        min(video_width, np.max(x_indices) + margin),
    )

    y_min, y_max, x_min, x_max = (
        int(y_min),
        int(y_max),
        int(x_min),
        int(x_max),
    )

    roi_coords = (y_min, y_max, x_min, x_max)
    roi_mask = text_mask[y_min:y_max, x_min:x_max]

    processor = TextProcessor(model, config, roi_coords, roi_mask)

    frame_count = 0
    total_frames = video_info["total_frames"]
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frames")
    temporary_output_path = f"{output_path}.part.mp4"

    def process_frame_wrapper(frame):
        nonlocal frame_count
        result = processor.process_frame(frame, frame_count)

        frame_count += 1
        progress_bar.update(1)
        return result

    processed_video = video_clip.image_transform(
        lambda frame: process_frame_wrapper(frame)
    )
    try:
        processed_video.write_videofile(
            temporary_output_path,
            codec="libx264",
            audio=True,
            audio_codec="aac",
            fps=video_clip.fps,
            logger=None,
        )
        if not is_readable_video_file(temporary_output_path):
            raise RuntimeError(
                f"MoviePy produced an unreadable output video: {temporary_output_path}"
            )
        os.replace(temporary_output_path, f"{output_path}.mp4")
    finally:
        progress_bar.close()
        processed_video.close()
        if os.path.exists(temporary_output_path):
            os.remove(temporary_output_path)

    end_time = time.time()
    processing_time = end_time - start_time
    processing_info = {
        "video_info": video_info,
        "processing_time": str(timedelta(seconds=int(processing_time))),
    }

    return processing_info


def process_video_on_device(
    video,
    output_dir,
    device,
    roi,
    preview_enabled,
    require_device,
):
    """Process one video in an isolated worker and return its state record."""
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The preview path runs in the already initialized main process.
        pass
    cv2.setNumThreads(2)

    video_name = os.path.basename(video)
    output_video_path = os.path.join(output_dir, os.path.splitext(video_name)[0])
    video_clip = None
    video_info = None
    try:
        if device.startswith("npu:"):
            try:
                torch.npu.set_device(int(device.split(":", 1)[1]))
            except Exception as device_error:
                if require_device:
                    raise RuntimeError(
                        f"Required NPU {device} could not be activated: {device_error}"
                    ) from device_error
                print(
                    f"Cannot activate {device} for {video_name}; "
                    f"using CPU: {device_error}"
                )
                device = "cpu"

        video_clip = VideoFileClip(video)
        video_info = get_video_info(video_clip)
        detector = TextDetector(roi=roi)
        if detector.roi is not None:
            detector.validate_roi(video_clip)
        text_mask = detector.generate_mask(video_clip)
        try:
            lama_model, lama_config = initialize_lama(device=device)
            if preview_enabled and not detector.preview_effect(
                video_clip, text_mask, lama_model, lama_config
            ):
                print(f"Processing cancelled by user: {video_name}")
                return {"status": "cancelled", "video": video}

            processing_info = process_video(
                video_clip,
                output_video_path,
                text_mask,
                lama_model,
                lama_config,
            )
        except Exception as accelerator_error:
            if not device.startswith("npu:") or require_device:
                raise

            # LaMa Cleaner 1.2.5 may request Ascend operators that are missing
            # from a particular CANN kernel package. Reopen the source and
            # retry on CPU so one unsupported operator does not drop a video.
            print(
                f"NPU inference failed for {video_name}; retrying on CPU: "
                f"{accelerator_error}"
            )
            video_clip.close()
            video_clip = VideoFileClip(video)
            lama_model, lama_config = initialize_lama(device="cpu")
            processing_info = process_video(
                video_clip,
                output_video_path,
                text_mask,
                lama_model,
                lama_config,
            )
            device = "cpu fallback"
        return {
            "status": "success",
            "video": video,
            "roi": detector.roi,
            "output_file": f"{output_video_path}.mp4",
            "video_info": processing_info["video_info"],
            "processing_time": processing_info["processing_time"],
            "device": device,
        }
    except Exception as error:
        return {
            "status": "failed",
            "video": video,
            "roi": roi,
            "output_file": f"{output_video_path}.mp4",
            "video_info": video_info,
            "error": str(error),
            "device": device,
        }
    finally:
        if video_clip is not None:
            video_clip.close()
