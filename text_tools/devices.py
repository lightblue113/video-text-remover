"""CPU, CUDA, and Ascend NPU availability checks."""

import glob

import torch
from loguru import logger


def check_gpu():
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        return True, device, gpu_name
    else:
        return False, "cpu", None


def check_npus(device_ids):
    """Return the available requested Ascend NPUs and their display names.

    ``torch_npu`` is imported lazily because importing it initializes the
    Ascend runtime.  This keeps CPU and CUDA-only installations usable while
    allowing the NPU to be selected explicitly (and by default).
    """
    # Don't initialize torch_npu when Ascend devices are not exposed to this
    # process. This is common in CPU-only containers and avoids noisy runtime
    # driver failures during the normal CPU fallback.
    if not glob.glob("/dev/davinci[0-9]*"):
        return [], []

    try:
        import torch_npu  # noqa: F401  # Registers the NPU backend with torch.

        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            return [], []

        device_count = npu.device_count()
        invalid_ids = [
            device_id
            for device_id in device_ids
            if device_id < 0 or device_id >= device_count
        ]
        if invalid_ids:
            raise ValueError(
                f"NPU device IDs {invalid_ids} are out of range; "
                f"available devices: 0-{device_count - 1}"
            )

        return device_ids, [npu.get_device_name(device_id) for device_id in device_ids]
    except ValueError:
        raise
    except Exception as error:
        logger.warning(f"NPU is unavailable: {error}")
        return [], []


def check_npu(device_id=0):
    """Backward-compatible single-NPU probe."""
    device_ids, device_names = check_npus([device_id])
    if not device_ids:
        return False, "npu", None
    return True, f"npu:{device_ids[0]}", device_names[0]
