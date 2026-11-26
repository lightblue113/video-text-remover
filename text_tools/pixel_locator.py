"""OpenCV pixel inspection without loading PyTorch or the inpainting model."""

import argparse
import math
from pathlib import Path

import cv2


def load_frame(path, time_seconds=0.0):
    """Read a BGR image or a video frame at the requested time in seconds."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    if not math.isfinite(time_seconds) or time_seconds < 0:
        raise ValueError("--time must be a finite, non-negative number")

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is not None:
        if time_seconds != 0:
            raise ValueError("--time applies only to video inputs")
        return frame

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open image or video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0 and frame_count > 0 and time_seconds >= frame_count / fps:
            raise ValueError("--time is outside the video duration")
        if time_seconds:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
        readable, frame = capture.read()
        if not readable or frame is None:
            raise ValueError(f"Cannot read a frame at {time_seconds:g}s from {path}")
        return frame
    finally:
        capture.release()


class PixelLocator:
    """Map mouse positions on a scaled preview back to source image pixels."""

    def __init__(self, frame, max_width=1280, max_height=720):
        if max_width < 1 or max_height < 1:
            raise ValueError("Display dimensions must be positive")
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        scale = min(1.0, max_width / self.width, max_height / self.height)
        self.display_width = max(1, int(self.width * scale))
        self.display_height = max(1, int(self.height * scale))
        self.preview = cv2.resize(
            frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        self.cursor = None
        self.drag_start = None
        self.roi = None

    def source_coordinates(self, x, y):
        """Return source (x, y), clamping to the displayed image boundaries."""
        x = max(0, min(x, self.display_width - 1))
        y = max(0, min(y, self.display_height - 1))
        return (
            min(self.width - 1, int(x * self.width / self.display_width)),
            min(self.height - 1, int(y * self.height / self.display_height)),
        )

    def on_mouse(self, event, x, y, flags, param):
        """Report clicked pixels and print a CLI-ready ROI after a drag."""
        if event == cv2.EVENT_RBUTTONDOWN:
            self.drag_start = None
            self.roi = None
            return
        inside = 0 <= x < self.display_width and 0 <= y < self.display_height
        if not inside and self.drag_start is None:
            self.cursor = None
            return

        self.cursor = self.source_coordinates(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.cursor
            self.roi = None
            source_x, source_y = self.cursor
            rgb = tuple(int(value) for value in self.frame[source_y, source_x, ::-1])
            print(f"Pixel: x={source_x}, y={source_y}, RGB={rgb}", flush=True)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            start_x, start_y = self.drag_start
            end_x, end_y = self.cursor
            if self.cursor != self.drag_start:
                self.roi = (
                    min(start_x, end_x),
                    min(start_y, end_y),
                    abs(end_x - start_x) + 1,
                    abs(end_y - start_y) + 1,
                )
                print("--roi " + " ".join(str(value) for value in self.roi), flush=True)
            self.drag_start = None

    def render(self):
        """Draw the current source coordinate and selection on a fresh preview."""
        display = self.preview.copy()
        if self.drag_start is not None and self.cursor is not None:
            start_x, start_y = self.drag_start
            end_x, end_y = self.cursor
            rectangle = (
                min(start_x, end_x),
                min(start_y, end_y),
                abs(end_x - start_x) + 1,
                abs(end_y - start_y) + 1,
            )
        else:
            rectangle = self.roi
        if rectangle is not None:
            x, y, width, height = rectangle
            start = (
                int(x * self.display_width / self.width),
                int(y * self.display_height / self.height),
            )
            end = (
                int((x + width - 1) * self.display_width / self.width),
                int((y + height - 1) * self.display_height / self.height),
            )
            cv2.rectangle(display, start, end, (0, 255, 0), 1)

        status = f"Source: {self.width}x{self.height} | Move mouse over the image"
        if self.cursor is not None:
            x, y = self.cursor
            rgb = tuple(int(value) for value in self.frame[y, x, ::-1])
            status = f"x={x}, y={y} | RGB={rgb}"
        if self.roi is not None:
            status += " | --roi " + " ".join(str(value) for value in self.roi)

        # Put text below the source image so it does not cover pixels being inspected.
        display = cv2.copyMakeBorder(display, 0, 60, 0, 0, cv2.BORDER_CONSTANT)
        for index, text in enumerate(
            (
                status,
                "Click: pixel | Drag: ROI | Right-click: reset | Enter/Esc/Q: quit",
            )
        ):
            cv2.putText(
                display,
                text,
                (8, self.display_height + 22 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return display

    def show(self):
        window = "Pixel Locator"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setMouseCallback(window, self.on_mouse)
            while True:
                cv2.imshow(window, self.render())
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect source pixels or drag a text ROI in an image/video frame."
    )
    parser.add_argument("input", help="Image or video file to inspect")
    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Video timestamp in seconds (default: 0)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Maximum image display width (default: 1280)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Maximum image display height (default: 720)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="PATH",
        help="Save the source frame and exit without opening a window",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.time) or args.time < 0:
        parser.error("--time must be a finite, non-negative number")
    if args.max_width < 1 or args.max_height < 1:
        parser.error("--max-width and --max-height must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        frame = load_frame(args.input, args.time)
        print(f"Source resolution: {frame.shape[1]}x{frame.shape[0]}")
        if args.export is not None:
            if not cv2.imwrite(str(args.export), frame):
                raise ValueError(f"Cannot save frame to {args.export}")
            print(f"Saved frame to {args.export}")
            return
        print(
            "Coordinates are zero-based source pixels: x increases right, y increases down."
        )
        print("Click to print a pixel; drag to print --roi X Y WIDTH HEIGHT.")
        PixelLocator(frame, args.max_width, args.max_height).show()
    except (ValueError, cv2.error) as error:
        raise SystemExit(str(error)) from error
