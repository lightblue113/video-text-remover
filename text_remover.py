"""Video text remover entry point and public helper imports."""

from text_tools.cli import main, parse_args
from text_tools.detection import TextDetector
from text_tools.devices import check_gpu, check_npu, check_npus
from text_tools.inpainting import initialize_lama, lama_inpaint
from text_tools.processing import (
    TextProcessor,
    process_video,
    process_video_on_device,
)
from text_tools.state import (
    STATE_FILENAME,
    ensure_directory_exists,
    get_unprocessed_videos,
    load_processing_state,
    record_video_state,
    save_processing_state,
)
from text_tools.video import (
    get_video_info,
    is_readable_video_file,
    is_valid_video_file,
)

__all__ = [
    "STATE_FILENAME",
    "TextDetector",
    "TextProcessor",
    "check_gpu",
    "check_npu",
    "check_npus",
    "ensure_directory_exists",
    "get_unprocessed_videos",
    "get_video_info",
    "initialize_lama",
    "is_readable_video_file",
    "is_valid_video_file",
    "lama_inpaint",
    "load_processing_state",
    "main",
    "parse_args",
    "process_video",
    "process_video_on_device",
    "record_video_state",
    "save_processing_state",
]


if __name__ == "__main__":
    main()
