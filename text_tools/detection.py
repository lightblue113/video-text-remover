"""Text region selection and temporal mask detection."""

import cv2
import numpy as np

from .inpainting import lama_inpaint


class TextDetector:
    def __init__(
        self,
        num_sample_frames=10,
        min_frame_count=7,
        dilation_kernel_size=5,
        roi=None,
    ):
        self.num_sample_frames = num_sample_frames
        self.min_frame_count = min_frame_count
        self.dilation_kernel_size = dilation_kernel_size
        self.roi = tuple(roi) if roi is not None else None

    def validate_roi(self, video_clip):
        if self.roi is None:
            return
        if len(self.roi) != 4:
            raise ValueError("ROI must contain x, y, width, and height")

        x, y, width, height = (int(value) for value in self.roi)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("ROI must have non-negative position and positive size")
        video_width = int(video_clip.w)
        video_height = int(video_clip.h)
        if x + width > video_width or y + height > video_height:
            raise ValueError(
                f"ROI {(x, y, width, height)} exceeds video bounds "
                f"{video_width}x{video_height}"
            )
        self.roi = (x, y, width, height)

    def get_first_valid_frame(self, video_clip, threshold=10):
        total_frames = int(video_clip.fps * video_clip.duration)
        frame_indices = [
            int(i * total_frames / self.num_sample_frames)
            for i in range(self.num_sample_frames)
        ]

        for idx in frame_indices:
            frame = video_clip.get_frame(idx / video_clip.fps)
            if frame.mean() > threshold:
                return frame

        return video_clip.get_frame(0)

    def select_roi(self, video_clip):
        if self.roi is not None:
            self.validate_roi(video_clip)
            return self.roi

        frame = self.get_first_valid_frame(video_clip)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        display_height = 720
        scale_factor = display_height / frame.shape[0]
        display_width = int(frame.shape[1] * scale_factor)
        display_frame = cv2.resize(frame, (display_width, display_height))

        instructions = "Select ROI and press SPACE or ENTER"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            display_frame,
            instructions,
            (10, 30),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        r = cv2.selectROI(display_frame)
        cv2.destroyAllWindows()

        self.roi = (
            int(r[0] / scale_factor),
            int(r[1] / scale_factor),
            int(r[2] / scale_factor),
            int(r[3] / scale_factor),
        )

        self.validate_roi(video_clip)

        return self.roi

    def detect_text_in_frame(self, frame):
        if self.roi is None:
            raise ValueError("ROI hasn't been selected yet. Call select_roi first.")

        roi_frame = frame[
            self.roi[1] : self.roi[1] + self.roi[3],
            self.roi[0] : self.roi[0] + self.roi[2],
        ]
        # MoviePy returns RGB frames; use the matching OpenCV conversion.
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_RGB2GRAY)
        _, binary_frame = cv2.threshold(
            gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        mask[
            self.roi[1] : self.roi[1] + self.roi[3],
            self.roi[0] : self.roi[0] + self.roi[2],
        ] = binary_frame

        return mask

    def generate_mask(self, video_clip):
        if self.roi is None:
            self.select_roi(video_clip)
        else:
            self.validate_roi(video_clip)

        total_frames = int(video_clip.duration * video_clip.fps)
        frame_indices = [
            int(i * total_frames / self.num_sample_frames)
            for i in range(self.num_sample_frames)
        ]
        frames = [video_clip.get_frame(idx / video_clip.fps) for idx in frame_indices]

        masks = [self.detect_text_in_frame(frame) for frame in frames]

        final_mask = sum((mask == 255).astype(np.uint8) for mask in masks)
        final_mask = np.where(final_mask >= self.min_frame_count, 255, 0).astype(
            np.uint8
        )

        kernel = np.ones(
            (self.dilation_kernel_size, self.dilation_kernel_size), np.uint8
        )
        dilated_mask = cv2.dilate(final_mask, kernel, iterations=2)

        return dilated_mask

    def get_roi_coordinates(self, text_mask, margin=50):
        y_indices, x_indices = np.where(text_mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError("No text region found in mask")

        y_min = max(0, np.min(y_indices) - margin)
        y_max = min(text_mask.shape[0], np.max(y_indices) + margin)
        x_min = max(0, np.min(x_indices) - margin)
        x_max = min(text_mask.shape[1], np.max(x_indices) + margin)

        return (y_min, y_max, x_min, x_max)

    def extract_roi_mask(self, text_mask, roi_coords):
        y_min, y_max, x_min, x_max = roi_coords
        return text_mask[y_min:y_max, x_min:x_max]

    def preview_effect(self, video_clip, text_mask, model, config, max_height=720):
        frame = self.get_first_valid_frame(video_clip)

        result = lama_inpaint(frame, text_mask, model, config)

        h, w = result.shape[:2]

        scale_factor = max_height / h
        display_width = int(w * scale_factor)

        # cv2.imshow expects BGR, while lama_inpaint returns RGB for MoviePy.
        display_result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        display_result = cv2.resize(display_result, (display_width, max_height))

        instructions = "Processed Result | Press ENTER to continue, ESC to exit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            display_result,
            instructions,
            (10, 30),
            font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Text Removal Result", display_result)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        return key != 27
