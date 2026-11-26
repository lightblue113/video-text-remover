"""Command-line arguments and batch processing orchestration."""

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from loguru import logger
from moviepy import VideoFileClip

from .detection import TextDetector
from .devices import check_gpu, check_npus
from .inpainting import initialize_lama, lama_inpaint
from .processing import process_video_on_device
from .state import (
    ensure_directory_exists,
    get_unprocessed_videos,
    load_processing_state,
    record_video_state,
)

logger.remove()
logger.add(sys.stderr, level="ERROR", format="{message}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Video Text Remover")
    parser.add_argument(
        "--input", "-i", type=str, default=".", help="Input directory containing videos"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--preview", "-p", action="store_true", help="Preview effect before processing"
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Numeric text ROI in source pixels: X Y WIDTH HEIGHT",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry videos recorded as failed in the state file",
    )
    parser.add_argument(
        "--device",
        choices=("npu", "cuda", "cpu"),
        default="npu",
        help="Inference device (default: npu)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        nargs="+",
        default=list(range(8)),
        metavar="ID",
        help="NPU device indexes when --device npu (default: 0 through 7)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent video workers on CPU/fallback (default: 8)",
    )
    parser.add_argument(
        "--require-device",
        action="store_true",
        help="Fail instead of falling back to CPU if the selected accelerator is unavailable",
    )
    parser.add_argument(
        "--npu-preflight",
        action="store_true",
        help="Run a small LaMa inference on the first selected NPU and exit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the text removal CLI with optional explicit arguments."""
    args = parse_args(argv)

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    # Keep CPU fallback parallel without each worker starting a large,
    # oversubscribed intra-op thread pool.
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)

    if args.npu_preflight:
        preflight_device_id = args.device_id[0]
        npu_ids, _ = check_npus([preflight_device_id])
        if not npu_ids:
            raise SystemExit(
                f"NPU {preflight_device_id} is not available to this process"
            )
        torch.npu.set_device(preflight_device_id)
        lama_model, lama_config = initialize_lama(device=f"npu:{preflight_device_id}")
        sample_frame = np.zeros((64, 485, 3), dtype=np.uint8)
        sample_mask = np.zeros((64, 485), dtype=np.uint8)
        sample_mask[16:48, 180:300] = 255
        sample_result = lama_inpaint(sample_frame, sample_mask, lama_model, lama_config)
        if sample_result.shape != sample_frame.shape:
            raise RuntimeError(
                f"NPU preflight returned {sample_result.shape}, "
                f"expected {sample_frame.shape}"
            )
        print(f"NPU {preflight_device_id} LaMa preflight succeeded")
        sys.exit(0)

    input_dir = args.input
    output_dir = args.output
    preview_enabled = args.preview

    if args.device == "npu":
        npu_ids, npu_names = check_npus(list(dict.fromkeys(args.device_id)))
        if npu_ids:
            for npu_id, npu_name in zip(npu_ids, npu_names):
                print(f"NPU {npu_id} detected: {npu_name}")
            print(f"Using {len(npu_ids)} NPU(s) for processing")
            devices = [f"npu:{npu_id}" for npu_id in npu_ids]
        else:
            if args.require_device:
                raise SystemExit(
                    "NPU was required, but no requested NPU is available to this "
                    "process. Check that Ascend driver device nodes (/dev/davinci*) "
                    "are mounted and that npu-smi can initialize DCMI."
                )
            print(
                "No working NPU detected, using CPU for processing "
                f"with {args.workers} workers"
            )
            devices = ["cpu"] * args.workers
    elif args.device == "cuda":
        has_gpu, use_device, gpu_name = check_gpu()
        if has_gpu:
            print(f"GPU detected: {gpu_name}")
            print("Using GPU for processing")
        else:
            if args.require_device:
                raise SystemExit(
                    "CUDA was required, but no CUDA device is available to this process"
                )
            print(
                f"No GPU detected, using CPU for processing with {args.workers} workers"
            )
            use_device = "cpu"
        devices = [use_device] if use_device != "cpu" else ["cpu"] * args.workers
    else:
        devices = ["cpu"] * args.workers
        print(f"Using CPU for processing with {args.workers} workers")

    if not ensure_directory_exists(output_dir):
        sys.exit(1)

    state = load_processing_state(output_dir)

    if os.path.isdir(input_dir):
        videos = get_unprocessed_videos(
            input_dir,
            output_dir,
            state=state,
            roi=args.roi,
            retry_failed=args.retry_failed,
        )
    else:
        print(f"Input path {input_dir} is not a directory")
        sys.exit(1)

    if not videos:
        print(f"No unprocessed valid video files found in {input_dir}")
        sys.exit(0)

    roi = tuple(args.roi) if args.roi is not None else None
    if roi is None:
        # Let the user select the shared text region once before starting
        # concurrent workers, avoiding multiple simultaneous OpenCV dialogs.
        selection_clip = VideoFileClip(videos[0])
        try:
            roi = TextDetector().select_roi(selection_clip)
        finally:
            selection_clip.close()

    max_workers = min(len(devices), len(videos))
    if preview_enabled:
        # OpenCV preview windows are interactive and should remain serial.
        max_workers = 1

    executor_type = ThreadPoolExecutor if preview_enabled else ProcessPoolExecutor
    executor_kwargs = {}
    if not preview_enabled:
        # Keep PyTorch and MoviePy's FFmpeg readers isolated per worker.
        executor_kwargs["mp_context"] = multiprocessing.get_context("spawn")

    with executor_type(max_workers=max_workers, **executor_kwargs) as executor:
        futures = [
            executor.submit(
                process_video_on_device,
                video,
                output_dir,
                devices[index % len(devices)],
                roi,
                preview_enabled,
                args.require_device,
            )
            for index, video in enumerate(videos)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "cancelled":
                continue

            if result["status"] == "success":
                record_video_state(
                    output_dir,
                    state,
                    result["video"],
                    status="success",
                    roi=result["roi"],
                    output_file=result["output_file"],
                    video_info=result["video_info"],
                    processing_time=result["processing_time"],
                )
                print(
                    f"Successfully processed "
                    f"{os.path.basename(result['video'])} on {result['device']}"
                )
                print(f"  Resolution: {result['video_info']['resolution']}")
                print(f"  Duration: {result['video_info']['duration']}")
                print(f"  FPS: {result['video_info']['fps']}")
                print(f"  Total frames: {result['video_info']['total_frames']}")
                print(f"  Processing time: {result['processing_time']}")
            else:
                record_video_state(
                    output_dir,
                    state,
                    result["video"],
                    status="failed",
                    roi=result["roi"],
                    output_file=result["output_file"],
                    video_info=result["video_info"],
                    error=result["error"],
                )
                print(
                    f"Failed to process {os.path.basename(result['video'])} "
                    f"on {result['device']}: {result['error']}"
                )
